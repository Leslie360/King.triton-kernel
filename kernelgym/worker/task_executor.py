"""
Task Executor with Subprocess Isolation.

This module provides fully isolated task execution:
1. Each task runs in its own subprocess
2. Uses the spawn context to avoid inheriting CUDA state
3. Each subprocess directly uses its assigned GPU device (cuda:0, cuda:1, ...)
4. CUDA errors affect only the current subprocess, never the main process or
   other subprocesses
5. Supports torch.profiler (enabled inside the subprocess)

Author: KernelGym Team
Date: 2025-10-29
Version: v0.3.3-alpha
"""

import sys
import time
import logging
import traceback
import multiprocessing as mp
import queue
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("kernelgym.task_executor")


@dataclass
class TaskExecutionMetrics:
    """Task-execution metrics."""
    subprocess_spawn_time: float  # subprocess startup time
    task_execution_time: float    # task execution time
    total_time: float             # total time
    profiling_overhead: float = 0.0  # profiling overhead (if enabled)
    success: bool = True
    error_type: Optional[str] = None


class IsolatedTaskExecutor:
    """
    Fully isolated task executor.

    Core features:
    1. Each task runs in a fresh subprocess (spawn mode)
    2. CUDA errors are fully isolated, never affecting the main process
    3. Supports torch.profiler (enabled inside the subprocess)
    4. Complete error handling and timeout mechanism
    5. Automatic GPU-resource cleanup

    Design principles:
    - The main process never uses CUDA
    - The subprocess initializes CUDA after spawn and uses its assigned GPU device
    - The profiler runs inside the subprocess (optional)
    - All data is passed via Queues
    """

    @staticmethod
    def execute_task(
        task_data: Dict[str, Any],
        device_id: int,
        timeout: int = 60,
    ) -> Tuple[Dict[str, Any], TaskExecutionMetrics]:
        """
        Execute a generic task (toolkit + backend) in an isolated subprocess.

        Args:
            task_data: Task payload dict (must include toolkit and backend_adapter).
            device_id: Physical GPU device id (e.g. 0-7).
            timeout: Timeout in seconds.

        Returns:
            (result_dict, metrics): The result dict and execution metrics.

        Raises:
            TimeoutError: Task timed out.
            RuntimeError: Task execution failed.
        """
        start_time = time.time()

        # Create a process with the spawn context
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        spawn_start = time.time()

        process = ctx.Process(
            target=_toolkit_worker,
            args=(task_data, device_id, result_queue),
        )

        process.start()
        spawn_time = time.time() - spawn_start

        try:
            # Wait for the result
            exec_start = time.time()
            result_data = result_queue.get(timeout=timeout)
            exec_time = time.time() - exec_start

            # Fetch profiling data (if any)
            # Wait for the process to end
            process.join(timeout=5)
            if process.is_alive():
                logger.warning("Process did not terminate, forcing kill")
                process.terminate()
                process.join(timeout=2)

            total_time = time.time() - start_time

            # Check the result
            if not result_data.get('success', False):
                # Task failed
                error_type = result_data.get('error_type', 'Unknown')
                error_message = result_data.get('error_message', 'Unknown error')

                metrics = TaskExecutionMetrics(
                    subprocess_spawn_time=spawn_time,
                    task_execution_time=exec_time,
                    total_time=total_time,
                    success=False,
                    error_type=error_type
                )

                raise RuntimeError(f"{error_type}: {error_message}")

            # Task succeeded
            result = result_data['result']

            # Compute profiling overhead
            profiling_overhead = 0.0
            if task_data.get("enable_profiling"):
                profiling_overhead = exec_time * 0.1  # rough estimate

            metrics = TaskExecutionMetrics(
                subprocess_spawn_time=spawn_time,
                task_execution_time=exec_time,
                total_time=total_time,
                profiling_overhead=profiling_overhead,
                success=True
            )

            logger.info(
                f"Task {task_data.get('task_id', 'unknown')} completed: "
                f"spawn={spawn_time:.3f}s, exec={exec_time:.3f}s, total={total_time:.3f}s"
            )

            return result, metrics

        except queue.Empty:
            # Timeout
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()

            total_time = time.time() - start_time
            metrics = TaskExecutionMetrics(
                subprocess_spawn_time=spawn_time,
                task_execution_time=timeout,
                total_time=total_time,
                success=False,
                error_type='TimeoutError'
            )

            raise TimeoutError(
                f"Task {task_data.get('task_id', 'unknown')} timeout after {timeout}s"
            )

        except Exception as e:
            # Other errors
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()

            logger.error(f"Task execution failed: {e}")
            raise

        finally:
            # Ensure the process is cleaned up
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()


# ============================================================================
# Subprocess Worker Functions (module-level so they can be pickled)
# ============================================================================

def _toolkit_worker(
    task_data: Dict[str, Any],
    device_id: int,
    result_queue: mp.Queue,
):
    try:
        import torch
        import torch.cuda
        from kernelgym.backend import get_backend
        from kernelgym.toolkit import get_toolkit

        torch.cuda.init()
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        task_data["device"] = str(device)

        toolkit_name = task_data.get("toolkit")
        backend_adapter = task_data.get("backend_adapter")
        if not toolkit_name:
            raise ValueError("Task payload missing required 'toolkit'")
        if not backend_adapter:
            raise ValueError("Task payload missing required 'backend_adapter'")

        toolkit = get_toolkit(toolkit_name)
        backend = get_backend(backend_adapter)

        result = toolkit.evaluate(task_data, backend=backend)
        result_queue.put({"success": True, "result": result})
    except Exception as e:
        error_info = {
            "success": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        result_queue.put(error_info)
    finally:
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as cleanup_error:
            print(f"[WARNING] GPU cleanup failed: {cleanup_error}", file=sys.stderr)
