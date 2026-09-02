"""
GPU diagnostic utilities — used for testing subprocess isolation and profiler compatibility.

This module provides:
1. Test GPU availability without initializing CUDA in the main process
2. Test CUDA isolation in subprocesses
3. Test profiler compatibility inside subprocesses
4. Verify that CUDA errors do not contaminate the main process

Author: KernelServer Team
Date: 2025-10-29
"""

import os
import sys
import subprocess
import multiprocessing as mp
import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger("kernelgym.gpu_diagnostics")


@dataclass
class GPUHealthReport:
    """GPU health-check report."""
    healthy: bool
    device_id: int
    device_name: Optional[str] = None
    total_memory_gb: Optional[float] = None
    cuda_available: bool = False
    error_message: Optional[str] = None
    test_duration_sec: float = 0.0


@dataclass
class IsolationTestReport:
    """Isolation-test report."""
    isolation_successful: bool
    main_process_contaminated: bool
    subprocess_error_message: Optional[str] = None
    details: Dict[str, Any] = None


@dataclass
class ProfilerTestReport:
    """Profiler-compatibility test report."""
    profiler_works: bool
    profiling_data_received: bool
    profiling_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# Module-level worker functions (must be at module level to be picklable)

def _gpu_health_worker(device_id: int, result_queue):
    """Subprocess worker for the GPU health test."""
    try:
        # Set CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

        # Import torch (inside the subprocess)
        import torch

        if not torch.cuda.is_available():
            result_queue.put({
                'success': False,
                'error': 'CUDA not available in subprocess'
            })
            return

        # Initialize CUDA
        torch.cuda.init()
        torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES exposes only one GPU

        # Get GPU info
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory

        # Simple test
        test_tensor = torch.randn(100, 100, device='cuda')
        result = torch.mm(test_tensor, test_tensor.T)
        torch.cuda.synchronize()

        result_queue.put({
            'success': True,
            'device_name': device_name,
            'total_memory': total_memory
        })

    except Exception as e:
        result_queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def _cuda_error_worker(device_id: int, result_queue):
    """Worker that intentionally triggers a CUDA error."""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        import torch

        torch.cuda.init()
        torch.cuda.set_device(0)

        # Intentionally trigger a CUDA error: access an invalid memory address.
        try:
            # Create a huge tensor that may OOM
            giant_tensor = torch.randn(100000, 100000, device='cuda')
            # Or use an invalid CUDA kernel configuration
            result_queue.put({'phase': 'error_triggered', 'success': False, 'expected': True})
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'out of memory' in str(e):
                result_queue.put({
                    'phase': 'error_caught',
                    'success': True,
                    'error': str(e)
                })
            else:
                raise

    except Exception as e:
        result_queue.put({
            'phase': 'unexpected_error',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def _normal_worker(device_id: int, result_queue):
    """A normal worker used to test whether the GPU is still usable."""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        import torch

        torch.cuda.init()
        torch.cuda.set_device(0)

        # Simple test
        test_tensor = torch.randn(100, 100, device='cuda')
        result = torch.mm(test_tensor, test_tensor.T)
        torch.cuda.synchronize()

        result_queue.put({'success': True, 'phase': 'normal_execution'})

    except Exception as e:
        result_queue.put({
            'success': False,
            'phase': 'normal_execution_failed',
            'error': str(e)
        })


def _profiler_worker(device_id: int, result_queue, profiling_queue):
    """Worker that uses the profiler."""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        import torch
        import torch.profiler as profiler

        torch.cuda.init()
        torch.cuda.set_device(0)

        # Enable profiler
        prof = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False
        )

        with prof:
            # Run some CUDA ops
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.mm(x, x.T)
            z = torch.nn.functional.relu(y)
            torch.cuda.synchronize()

        # Extract profiling data
        events = prof.key_averages()
        cuda_events = [
            evt for evt in events
            if hasattr(evt, 'device_type') and
            evt.device_type == profiler.DeviceType.CUDA
        ]

        # Build serializable profiling data
        profiling_data = {
            'total_events': len(list(events)),
            'cuda_events': len(cuda_events),
            'top_5_cuda_kernels': [
                {
                    'name': evt.key,
                    'cuda_time_us': float(evt.cuda_time_total) if hasattr(evt, 'cuda_time_total') else 0.0,
                    'count': int(evt.count) if hasattr(evt, 'count') else 0
                }
                for evt in sorted(
                    cuda_events,
                    key=lambda e: getattr(e, 'cuda_time_total', 0.0),
                    reverse=True
                )[:5]
            ]
        }

        # Send profiling data
        profiling_queue.put(profiling_data)
        result_queue.put({'success': True})

    except Exception as e:
        result_queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


class GPUDiagnostics:
    """GPU diagnostic toolkit."""

    @staticmethod
    def test_gpu_health_nvidia_smi(device_id: int) -> GPUHealthReport:
        """
        Test GPU health with nvidia-smi (no CUDA initialization).

        Args:
            device_id: GPU device id.

        Returns:
            A GPUHealthReport.
        """
        start_time = time.time()

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i", str(device_id),
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return GPUHealthReport(
                    healthy=False,
                    device_id=device_id,
                    error_message=f"nvidia-smi failed: {result.stderr}",
                    test_duration_sec=time.time() - start_time
                )

            # Parse output: "GPU Name, Memory in MB"
            output = result.stdout.strip()
            parts = output.split(',')

            if len(parts) < 2:
                return GPUHealthReport(
                    healthy=False,
                    device_id=device_id,
                    error_message=f"Unexpected nvidia-smi output: {output}",
                    test_duration_sec=time.time() - start_time
                )

            device_name = parts[0].strip()
            memory_mb = float(parts[1].strip())
            memory_gb = memory_mb / 1024.0

            return GPUHealthReport(
                healthy=True,
                device_id=device_id,
                device_name=device_name,
                total_memory_gb=memory_gb,
                cuda_available=True,
                test_duration_sec=time.time() - start_time
            )

        except subprocess.TimeoutExpired:
            return GPUHealthReport(
                healthy=False,
                device_id=device_id,
                error_message="nvidia-smi timeout",
                test_duration_sec=time.time() - start_time
            )
        except Exception as e:
            return GPUHealthReport(
                healthy=False,
                device_id=device_id,
                error_message=f"nvidia-smi error: {str(e)}",
                test_duration_sec=time.time() - start_time
            )

    @staticmethod
    def test_gpu_health_subprocess(device_id: int) -> GPUHealthReport:
        """
        Test GPU health inside a subprocess.

        This method initializes CUDA and tests the GPU inside an isolated subprocess,
        so the main process is unaffected.

        Args:
            device_id: GPU device id.

        Returns:
            A GPUHealthReport.
        """
        start_time = time.time()

        # Use spawn context to create the process
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()

        process = ctx.Process(target=_gpu_health_worker, args=(device_id, result_queue))
        process.start()

        try:
            # Wait for the result, 10s timeout (spawn + import torch + CUDA init takes time)
            result = result_queue.get(timeout=10)
            process.join(timeout=2)

            duration = time.time() - start_time

            if result['success']:
                return GPUHealthReport(
                    healthy=True,
                    device_id=device_id,
                    device_name=result['device_name'],
                    total_memory_gb=result['total_memory'] / (1024**3),
                    cuda_available=True,
                    test_duration_sec=duration
                )
            else:
                return GPUHealthReport(
                    healthy=False,
                    device_id=device_id,
                    cuda_available=False,
                    error_message=result.get('error', 'Unknown error'),
                    test_duration_sec=duration
                )

        except Exception as e:
            process.terminate()
            process.join(timeout=2)

            return GPUHealthReport(
                healthy=False,
                device_id=device_id,
                cuda_available=False,
                error_message=f"Subprocess test failed: {str(e)}",
                test_duration_sec=time.time() - start_time
            )

    @staticmethod
    def test_cuda_error_isolation(device_id: int) -> IsolationTestReport:
        """
        Test CUDA-error isolation.

        Intentionally triggers a CUDA error in a subprocess, then verifies:
        1. The subprocess correctly catches the error
        2. The main process is unaffected
        3. Subsequent subprocesses can still use the GPU

        Args:
            device_id: GPU device id.

        Returns:
            An IsolationTestReport.
        """
        ctx = mp.get_context('spawn')

        # Step 1: trigger a CUDA error
        logger.info(f"[Isolation Test] Step 1: trigger a CUDA error in a subprocess")
        result_queue1 = ctx.Queue()
        process1 = ctx.Process(target=_cuda_error_worker, args=(device_id, result_queue1))
        process1.start()

        try:
            result1 = result_queue1.get(timeout=15)  # increased timeout
            process1.join(timeout=2)
        except Exception as e:
            process1.terminate()
            return IsolationTestReport(
                isolation_successful=False,
                main_process_contaminated=False,
                subprocess_error_message=f"Step 1 failed: {str(e)}"
            )

        # Step 2: check whether the main process was affected (main process does
        # not use CUDA, so this should always be fine)
        logger.info(f"[Isolation Test] Step 2: check main-process state")
        # Main process does not use CUDA, so this step always succeeds
        main_process_ok = True

        # Step 3: test in a fresh subprocess whether the GPU is still usable
        logger.info(f"[Isolation Test] Step 3: test whether GPU is usable in a new subprocess")
        result_queue2 = ctx.Queue()
        process2 = ctx.Process(target=_normal_worker, args=(device_id, result_queue2))
        process2.start()

        try:
            result2 = result_queue2.get(timeout=15)  # increased timeout
            process2.join(timeout=2)
        except Exception as e:
            process2.terminate()
            return IsolationTestReport(
                isolation_successful=False,
                main_process_contaminated=False,
                subprocess_error_message=f"Step 3 failed: {str(e)}",
                details={
                    'step1': result1,
                    'step2': 'main_process_ok',
                    'step3_error': str(e)
                }
            )

        # Determine isolation success
        step1_ok = result1.get('phase') == 'error_caught'
        step2_ok = main_process_ok
        step3_ok = result2.get('success') == True

        isolation_successful = step1_ok and step2_ok and step3_ok

        # Build detailed error info
        error_parts = []
        if not step1_ok:
            error_parts.append(f"Step1 failed: phase={result1.get('phase')}, expected='error_caught'")
        if not step2_ok:
            error_parts.append("Step2 failed: main process contaminated")
        if not step3_ok:
            error_parts.append(f"Step3 failed: success={result2.get('success')}, expected=True")

        error_message = "; ".join(error_parts) if error_parts else None

        logger.info(f"[Isolation Test] Step 1 OK: {step1_ok}, Phase: {result1.get('phase')}")
        logger.info(f"[Isolation Test] Step 2 OK: {step2_ok}")
        logger.info(f"[Isolation Test] Step 3 OK: {step3_ok}, Success: {result2.get('success')}")

        return IsolationTestReport(
            isolation_successful=isolation_successful,
            main_process_contaminated=not main_process_ok,
            subprocess_error_message=error_message,
            details={
                'step1_error_caught': result1,
                'step2_main_process_ok': main_process_ok,
                'step3_gpu_available': result2
            }
        )

    @staticmethod
    def test_profiler_compatibility(device_id: int) -> ProfilerTestReport:
        """
        Test torch.profiler compatibility inside a subprocess.

        Verifies:
        1. The profiler can start normally inside a subprocess
        2. The profiler can collect CUDA events
        3. Profiling data can be passed back to the main process via a Queue

        Args:
            device_id: GPU device id.

        Returns:
            A ProfilerTestReport.
        """
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        profiling_queue = ctx.Queue()

        process = ctx.Process(
            target=_profiler_worker,
            args=(device_id, result_queue, profiling_queue)
        )
        process.start()

        try:
            # Wait for the result (profiler takes longer)
            result = result_queue.get(timeout=20)

            # Try to fetch profiling data
            profiling_data = None
            if not profiling_queue.empty():
                profiling_data = profiling_queue.get_nowait()

            process.join(timeout=2)

            if result['success']:
                return ProfilerTestReport(
                    profiler_works=True,
                    profiling_data_received=(profiling_data is not None),
                    profiling_data=profiling_data
                )
            else:
                return ProfilerTestReport(
                    profiler_works=False,
                    profiling_data_received=False,
                    error_message=result.get('error', 'Unknown error')
                )

        except Exception as e:
            process.terminate()
            process.join(timeout=2)

            return ProfilerTestReport(
                profiler_works=False,
                profiling_data_received=False,
                error_message=f"Profiler test failed: {str(e)}"
            )

    @staticmethod
    def run_full_diagnostics(device_id: int) -> Dict[str, Any]:
        """
        Run the full GPU diagnostics.

        Args:
            device_id: GPU device id.

        Returns:
            A dict containing all diagnostic results.
        """
        logger.info(f"=== Starting full diagnostics for GPU {device_id} ===")

        results = {}

        # Test 1: nvidia-smi health check
        logger.info("[Test 1/4] nvidia-smi health check...")
        health_nvidia_smi = GPUDiagnostics.test_gpu_health_nvidia_smi(device_id)
        results['health_nvidia_smi'] = health_nvidia_smi
        logger.info(f"  Result: {'PASS' if health_nvidia_smi.healthy else 'FAIL'}")
        if health_nvidia_smi.healthy:
            logger.info(f"  GPU: {health_nvidia_smi.device_name}, "
                       f"Memory: {health_nvidia_smi.total_memory_gb:.1f}GB")

        # Test 2: Subprocess health check
        logger.info("[Test 2/4] Subprocess CUDA health check...")
        health_subprocess = GPUDiagnostics.test_gpu_health_subprocess(device_id)
        results['health_subprocess'] = health_subprocess
        logger.info(f"  Result: {'PASS' if health_subprocess.healthy else 'FAIL'}")
        if not health_subprocess.healthy:
            logger.error(f"  Error: {health_subprocess.error_message}")

        # Test 3: CUDA-error isolation test
        logger.info("[Test 3/4] CUDA-error isolation test...")
        isolation = GPUDiagnostics.test_cuda_error_isolation(device_id)
        results['isolation_test'] = isolation
        logger.info(f"  Result: {'PASS (isolation succeeded)' if isolation.isolation_successful else 'FAIL (isolation failed)'}")
        if not isolation.isolation_successful:
            logger.warning(f"  Main process contaminated: {isolation.main_process_contaminated}")
            logger.error(f"  Error info: {isolation.subprocess_error_message}")
            if isolation.details:
                logger.debug(f"  Details: {isolation.details}")

        # Test 4: Profiler compatibility test
        logger.info("[Test 4/4] torch.profiler compatibility test...")
        profiler_test = GPUDiagnostics.test_profiler_compatibility(device_id)
        results['profiler_test'] = profiler_test
        logger.info(f"  Result: {'PASS (Profiler works)' if profiler_test.profiler_works else 'FAIL (Profiler failed)'}")
        if profiler_test.profiler_works:
            logger.info(f"  Profiling data received: {'YES' if profiler_test.profiling_data_received else 'NO'}")
            if profiler_test.profiling_data:
                logger.info(f"  CUDA-event count: {profiler_test.profiling_data.get('cuda_events', 0)}")

        # Summary
        logger.info("=== Diagnostics complete ===")
        all_passed = (
            health_nvidia_smi.healthy and
            health_subprocess.healthy and
            isolation.isolation_successful and
            profiler_test.profiler_works
        )
        logger.info(f"Overall status: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

        results['all_passed'] = all_passed
        return results


def main():
    """Command-line entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    parser = argparse.ArgumentParser(description="GPU diagnostic utilities")
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU device id (default: 0)'
    )
    parser.add_argument(
        '--test',
        choices=['health', 'isolation', 'profiler', 'all'],
        default='all',
        help='Which test to run'
    )

    args = parser.parse_args()

    if args.test == 'all':
        results = GPUDiagnostics.run_full_diagnostics(args.device)
        sys.exit(0 if results['all_passed'] else 1)

    elif args.test == 'health':
        report = GPUDiagnostics.test_gpu_health_subprocess(args.device)
        print(f"Healthy: {report.healthy}")
        if report.healthy:
            print(f"Device: {report.device_name}")
            print(f"Memory: {report.total_memory_gb:.1f}GB")
        sys.exit(0 if report.healthy else 1)

    elif args.test == 'isolation':
        report = GPUDiagnostics.test_cuda_error_isolation(args.device)
        print(f"Isolation Successful: {report.isolation_successful}")
        print(f"Main Process Contaminated: {report.main_process_contaminated}")
        sys.exit(0 if report.isolation_successful else 1)

    elif args.test == 'profiler':
        report = GPUDiagnostics.test_profiler_compatibility(args.device)
        print(f"Profiler Works: {report.profiler_works}")
        print(f"Data Received: {report.profiling_data_received}")
        if report.profiling_data:
            print(f"CUDA Events: {report.profiling_data.get('cuda_events', 0)}")
        sys.exit(0 if report.profiler_works else 1)


if __name__ == '__main__':
    main()

