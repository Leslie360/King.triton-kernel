"""
Subprocess Worker Pool with CUDA Error Auto-Restart

Core features:
1. Pre-spawn a pool of worker processes that handle multiple tasks
2. torch and CUDA are initialized exactly once at startup
3. **First CUDA error immediately terminates the worker process**
4. The main process automatically respawns a new worker
5. Drastically reduces spawn overhead (from 2.5s per task to near zero)

Author: KernelGym Team
Date: 2025-10-30
Version: v0.3.3-rc
"""

import asyncio
import logging
import multiprocessing as mp
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("kernelgym.subprocess_pool")


def _aggressive_gpu_cleanup(device_id: int):
    """
    Aggressively clean up GPU memory.

    Tries multiple strategies to free memory:
    1. Empty the PyTorch cache
    2. Run Python garbage collection
    3. Reset CUDA peak-memory stats
    4. Clear Triton's cache (if any)
    5. Synchronize CUDA operations

    Args:
        device_id: GPU device id.
    """
    import gc

    import torch

    # 1. Synchronize all CUDA operations
    try:
        torch.cuda.synchronize(device_id)
    except Exception:
        pass

    # 2. Empty the PyTorch cache
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 3. Python garbage collection (release unreferenced tensors)
    gc.collect()

    # 4. Empty the cache again
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 5. Reset memory stats (helps the next allocation)
    try:
        torch.cuda.reset_peak_memory_stats(device_id)
        torch.cuda.reset_accumulated_memory_stats(device_id)
    except Exception:
        pass

    # 6. Clear Triton's cache (if Triton is used)
    try:
        import triton  # noqa: F401 -- probe only: confirms Triton is importable in this process
        # Triton-compiled kernel cache may linger.
        # NOTE: Triton has no public cleanup API, but the cache is freed
        # automatically when the process exits.
    except (ImportError, AttributeError):
        pass

    # 7. Final synchronization
    try:
        torch.cuda.synchronize(device_id)
    except Exception:
        pass


@dataclass
class WorkerMetrics:
    """Worker execution metrics."""

    task_execution_time: float  # task execution time
    total_time: float  # total time (including queue wait)
    success: bool = True
    error_type: str | None = None


class PersistentWorker:
    """
    Persistent worker process.

    Features:
    - One-shot initialization of torch and CUDA at startup
    - Receives tasks via a Queue and returns results
    - **Exits immediately on CUDA error (via a special marker)**
    - The main process detects the exit and respawns a new worker
    """

    def __init__(
        self, worker_id: str, device_id: int, pool_size_info: str = "", max_tasks_per_worker: int = 100
    ):
        """
        Args:
            worker_id: Worker identifier (e.g. "worker_0").
            device_id: GPU device id (e.g. 0-7).
            pool_size_info: Pool-size info used in logging.
            max_tasks_per_worker: Max tasks per worker (prevents VRAM accumulation).
        """
        self.worker_id = worker_id
        self.device_id = device_id
        self.pool_size_info = pool_size_info
        self.max_tasks_per_worker = max_tasks_per_worker
        self.process: mp.Process | None = None

        # Use spawn context to ensure full isolation
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue(maxsize=10)  # bound queue size to avoid memory blowups
        self.result_queue = self.ctx.Queue(maxsize=10)

        self.is_alive_flag = True
        self.tasks_processed = 0
        self.start_time = time.time()

        # Start the worker process
        self._start_worker()

    def _start_worker(self):
        """Start the worker process."""
        logger.info(
            f"[{self.worker_id}] Starting persistent worker for GPU {self.device_id} {self.pool_size_info}"
        )

        self.process = self.ctx.Process(
            target=_persistent_worker_loop,
            args=(self.worker_id, self.device_id, self.task_queue, self.result_queue),
            daemon=False,  # not a daemon: ensure clean shutdown
        )

        self.process.start()

        # Wait for initialization to complete
        try:
            init_msg = self.result_queue.get(timeout=120)  # plenty of time to load torch (raised from 60s)
            if init_msg.get("status") == "READY":
                logger.info(
                    f"[{self.worker_id}] Worker initialized successfully "
                    f"(init_time={init_msg.get('init_time', 0):.2f}s)"
                )
                self.is_alive_flag = True
            else:
                raise RuntimeError(f"Worker failed to initialize: {init_msg}")
        except queue.Empty:
            self.process.terminate()
            self.process.join(timeout=5)
            raise RuntimeError(f"[{self.worker_id}] Worker initialization timeout (>120s)") from None

    def execute_task(self, task_data: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
        """
        Execute a task.

        Args:
            task_data: Task data dict.
            timeout: Timeout in seconds.

        Returns:
            Result dict with success, result/error_type/error_message.

        Raises:
            RuntimeError: Worker is dead or task execution failed.
            TimeoutError: Task timed out.
        """
        if not self.is_alive():
            raise RuntimeError(f"[{self.worker_id}] Worker is not alive")

        # Send the task
        try:
            self.task_queue.put(task_data, timeout=5)
        except queue.Full:
            raise RuntimeError(f"[{self.worker_id}] Task queue is full") from None

        # Wait for the result
        try:
            result = self.result_queue.get(timeout=timeout)

            # Check whether the worker reported a CUDA error and is about to exit
            if result.get("worker_exiting") is True:
                logger.warning(
                    f"[{self.worker_id}] Worker encountered CUDA error and will exit. "
                    f"Error: {result.get('error_type', 'Unknown')}: {result.get('error_message', 'N/A')}"
                )
                self.is_alive_flag = False
                # Mark the process as about to exit; the main process will respawn

            # Update stats
            self.tasks_processed += 1

            # **Critical: check whether we hit the task cap (prevents VRAM accumulation)**
            if self.tasks_processed >= self.max_tasks_per_worker:
                logger.info(
                    f"[{self.worker_id}] Reached max tasks limit ({self.max_tasks_per_worker}). "
                    f"Marking for restart to prevent memory accumulation."
                )
                self.is_alive_flag = False
                # Note: we don't kill the process immediately — let the next check
                # respawn it. This lets us return the current task's result first.

            return result

        except queue.Empty:
            # Timeout
            logger.error(
                f"[{self.worker_id}] Task timeout after {timeout}s, "
                f"task_id={task_data.get('task_id', 'unknown')}"
            )
            # Mark worker as unavailable (possibly stuck)
            self.is_alive_flag = False
            raise TimeoutError(
                f"[{self.worker_id}] Task {task_data.get('task_id', 'unknown')} timeout after {timeout}s"
            ) from None

    def is_alive(self) -> bool:
        """Check whether the worker is alive."""
        return self.is_alive_flag and self.process is not None and self.process.is_alive()

    def shutdown(self, timeout: int = 10):
        """Shut down the worker process."""
        logger.info(f"[{self.worker_id}] Shutting down worker...")

        try:
            # Send the shutdown signal
            self.task_queue.put({"command": "SHUTDOWN"}, timeout=2)

            # Wait for the process to end
            if self.process and self.process.is_alive():
                self.process.join(timeout=timeout)

                # If it hasn't ended, force-terminate
                if self.process.is_alive():
                    logger.warning(f"[{self.worker_id}] Force terminating worker")
                    self.process.terminate()
                    self.process.join(timeout=3)

                    if self.process.is_alive():
                        logger.error(f"[{self.worker_id}] Force killing worker")
                        self.process.kill()
                        # Bounded join: a subprocess stuck in an uninterruptible
                        # CUDA op (D-state) ignores even SIGKILL until the kernel
                        # reclaims it; an unbounded join() here would wedge the
                        # pool's own restart path. Give it 5s, then abandon and
                        # let worker_monitor's process-group kill + GPU reset
                        # reclaim the orphaned CUDA context.
                        self.process.join(timeout=5)
                        if self.process.is_alive():
                            logger.error(
                                f"[{self.worker_id}] Subprocess did not exit 5s after "
                                f"SIGKILL (likely D-state CUDA op); leaving GPU reclaim "
                                f"to worker_monitor process-group kill + GPU reset"
                            )

        except Exception as e:
            logger.error(f"[{self.worker_id}] Error during shutdown: {e}")
            if self.process and self.process.is_alive():
                self.process.kill()

        self.is_alive_flag = False
        logger.info(
            f"[{self.worker_id}] Worker shut down "
            f"(processed {self.tasks_processed} tasks in "
            f"{time.time() - self.start_time:.1f}s)"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get worker stats."""
        return {
            "worker_id": self.worker_id,
            "device_id": self.device_id,
            "is_alive": self.is_alive(),
            "tasks_processed": self.tasks_processed,
            "uptime": time.time() - self.start_time,
            "pid": self.process.pid if self.process else None,
        }


class SubprocessWorkerPool:
    """
    Worker-pool manager.

    Responsibilities:
    1. Manage multiple PersistentWorkers
    2. Dispatch tasks to idle workers
    3. **Automatically respawn workers that hit a CUDA error**
    4. Load balancing
    """

    def __init__(
        self,
        device_id: int,
        pool_size: int = 2,
        worker_prefix: str = "pool_worker",
        max_tasks_per_worker: int = 100,
    ):
        """
        Args:
            device_id: GPU device id.
            pool_size: Number of worker processes (recommend 2-4, adjust to memory).
            worker_prefix: Worker ID prefix.
            max_tasks_per_worker: Max tasks per worker (prevents VRAM accumulation, default 100).
        """
        self.device_id = device_id
        self.pool_size = pool_size
        self.worker_prefix = worker_prefix
        self.max_tasks_per_worker = max_tasks_per_worker

        # Workers list
        self.workers: list[PersistentWorker] = []
        self.idle_workers: list[PersistentWorker] = []
        self.busy_workers: list[PersistentWorker] = []

        # Stats
        self.total_tasks_processed = 0
        self.total_workers_restarted = 0
        self.pool_start_time = time.time()

        # Synchronization lock
        self.lock = asyncio.Lock()

        # Initialize workers
        self._init_workers()

        logger.info(f"[GPU {device_id}] Worker pool initialized with {pool_size} workers")

    def _init_workers(self):
        """Initialize all workers."""
        pool_info = f"(pool_size={self.pool_size}, max_tasks={self.max_tasks_per_worker})"

        for i in range(self.pool_size):
            worker_id = f"{self.worker_prefix}_{self.device_id}_{i}"
            try:
                worker = PersistentWorker(
                    worker_id, self.device_id, pool_info, max_tasks_per_worker=self.max_tasks_per_worker
                )
                self.workers.append(worker)
                self.idle_workers.append(worker)
            except Exception as e:
                logger.error(f"[GPU {self.device_id}] Failed to start worker {worker_id}: {e}")
                # If startup fails, keep trying to start the others.
                # At least one worker must come up.
                if len(self.workers) == 0 and i == self.pool_size - 1:
                    raise RuntimeError(f"[GPU {self.device_id}] Failed to start any worker in pool") from e

    async def execute_task(
        self, task_data: dict[str, Any], timeout: int = 60, max_retries: int = 2
    ) -> dict[str, Any]:
        """
        Execute a task (auto-selects an idle worker).

        Args:
            task_data: Task data.
            timeout: Timeout in seconds.
            max_retries: Max retries (used to retry after a worker restart).
                        Note: timeout errors are NOT retried, to avoid blocking the queue.

        Returns:
            Result dict.
        """
        retry_count = 0
        last_error = None
        is_timeout_error = False  # Track whether the error was a timeout

        while retry_count <= max_retries:
            # Acquire an idle worker
            worker = await self._get_idle_worker(timeout=timeout)

            if worker is None:
                # All workers are busy; wait briefly and retry
                await asyncio.sleep(0.5)
                retry_count += 1
                continue

            try:
                # Run the task in the thread pool so asyncio is not blocked
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, worker.execute_task, task_data, timeout)

                # Task done
                self.total_tasks_processed += 1

                # Check whether the worker needs to be restarted
                if not worker.is_alive():
                    logger.warning(f"[{worker.worker_id}] Worker needs restart after task")
                    await self._restart_worker(worker)

                return result

            except (RuntimeError, TimeoutError) as e:
                # The worker may be dead or may have timed out
                logger.error(f"[{worker.worker_id}] Task execution failed: {e}")
                last_error = e

                # Check whether this is a timeout error
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "Task timeout after" in error_msg:
                    is_timeout_error = True
                    task_id = task_data.get("task_id", "unknown")
                    logger.warning(
                        f"[{worker.worker_id}] Task {task_id} timeout detected "
                        f"(timeout={timeout}s) - will NOT retry to avoid blocking queue"
                    )

                # Try to restart the worker
                await self._restart_worker(worker)

                # Don't retry on timeout — exit immediately to free up the queue
                if is_timeout_error:
                    logger.error(
                        f"[{worker.worker_id}] Task failed due to timeout, "
                        f"not retrying to free up worker queue"
                    )
                    break  # Exit the retry loop immediately

                # Retry (only on non-timeout errors)
                retry_count += 1

            finally:
                # Return the worker to the idle pool (if still alive)
                await self._return_worker(worker)

        # Failed after all retries (or timeout)
        if is_timeout_error:
            task_id = task_data.get("task_id", "unknown")
            raise TimeoutError(
                f"[GPU {self.device_id}] Task {task_id} timeout after {timeout}s. "
                f"Not retried to avoid blocking worker queue."
            )
        else:
            raise RuntimeError(
                f"[GPU {self.device_id}] Task failed after {max_retries} retries. Last error: {last_error}"
            )

    async def _get_idle_worker(self, timeout: int = 60) -> PersistentWorker | None:
        """
        Acquire an idle worker.

        If all workers are busy, waits until one becomes idle.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            async with self.lock:
                # Prune dead workers
                self.idle_workers = [w for w in self.idle_workers if w.is_alive()]

                # Emergency recovery: if the pool has no workers at all, try to create one
                if not self.workers and not self.idle_workers and not self.busy_workers:
                    logger.warning(
                        f"[GPU {self.device_id}] Pool has no workers! Attempting emergency recovery..."
                    )
                    try:
                        # Wait briefly for GPU resources to be released
                        await asyncio.sleep(3.0)

                        emergency_worker = PersistentWorker(
                            f"worker_gpu_{self.device_id}_pool_{self.device_id}_emergency",
                            self.device_id,
                            "(emergency recovery)",
                            max_tasks_per_worker=self.max_tasks_per_worker,
                        )
                        self.workers.append(emergency_worker)
                        self.idle_workers.append(emergency_worker)
                        logger.info(f"[GPU {self.device_id}] Emergency worker created successfully")
                    except Exception as e:
                        logger.error(f"[GPU {self.device_id}] Emergency recovery failed: {e}")

                if self.idle_workers:
                    worker = self.idle_workers.pop(0)
                    self.busy_workers.append(worker)
                    return worker

            # No idle worker; wait briefly
            await asyncio.sleep(0.1)

        # Timeout
        logger.error(f"[GPU {self.device_id}] No idle worker available after {timeout}s")
        return None

    async def _return_worker(self, worker: PersistentWorker):
        """Return a worker to the idle pool."""
        async with self.lock:
            if worker in self.busy_workers:
                self.busy_workers.remove(worker)

            # Only return alive workers to the idle pool
            if worker.is_alive():
                if worker not in self.idle_workers:
                    self.idle_workers.append(worker)

    async def _restart_worker(self, worker: PersistentWorker):
        """
        Restart a worker.

        This function:
        1. Shuts down the old worker process
        2. Removes it from the workers list
        3. Creates a new worker
        4. Adds it to the idle pool
        """
        async with self.lock:
            logger.info(f"[{worker.worker_id}] Restarting worker (processed {worker.tasks_processed} tasks)")

            # Shut down the old worker
            try:
                worker.shutdown(timeout=5)
            except Exception as e:
                logger.error(f"[{worker.worker_id}] Error shutting down: {e}")

            # Remove from lists
            if worker in self.workers:
                self.workers.remove(worker)
            if worker in self.idle_workers:
                self.idle_workers.remove(worker)
            if worker in self.busy_workers:
                self.busy_workers.remove(worker)

            # Wait a moment for GPU resources to be fully released.
            # Under high load, restarting immediately can cause slow CUDA
            # initialization or timeouts.
            time.sleep(2.0)

            # Create a new worker (keep the same ID)
            try:
                new_worker = PersistentWorker(
                    worker.worker_id,
                    self.device_id,
                    f"(pool_size={self.pool_size}, max_tasks={self.max_tasks_per_worker}, restart)",
                    max_tasks_per_worker=self.max_tasks_per_worker,
                )

                self.workers.append(new_worker)
                self.idle_workers.append(new_worker)
                self.total_workers_restarted += 1

                logger.info(
                    f"[{worker.worker_id}] Worker restarted successfully "
                    f"(total restarts: {self.total_workers_restarted})"
                )

            except Exception as e:
                logger.error(
                    f"[{worker.worker_id}] Failed to restart worker: {e}. "
                    f"Pool now has {len(self.workers)} workers"
                )
                # If restart fails, the pool loses one worker but keeps operating

    async def shutdown(self, timeout: int = 30):
        """Shut down the entire worker pool."""
        logger.info(f"[GPU {self.device_id}] Shutting down worker pool...")

        # Shut down all workers
        for worker in self.workers:
            try:
                worker.shutdown(timeout=timeout // len(self.workers) if self.workers else 5)
            except Exception as e:
                logger.error(f"Error shutting down {worker.worker_id}: {e}")

        self.workers.clear()
        self.idle_workers.clear()
        self.busy_workers.clear()

        logger.info(
            f"[GPU {self.device_id}] Worker pool shut down "
            f"(processed {self.total_tasks_processed} tasks, "
            f"restarted {self.total_workers_restarted} workers in "
            f"{time.time() - self.pool_start_time:.1f}s)"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get pool stats."""
        return {
            "device_id": self.device_id,
            "pool_size": self.pool_size,
            "workers_alive": len([w for w in self.workers if w.is_alive()]),
            "idle_workers": len(self.idle_workers),
            "busy_workers": len(self.busy_workers),
            "total_tasks_processed": self.total_tasks_processed,
            "total_workers_restarted": self.total_workers_restarted,
            "uptime": time.time() - self.pool_start_time,
            "workers": [w.get_stats() for w in self.workers],
        }


# ============================================================================
# Worker Loop (runs inside the subprocess)
# ============================================================================


def _persistent_worker_loop(worker_id: str, device_id: int, task_queue: mp.Queue, result_queue: mp.Queue):
    """
    Main loop of the persistent worker.

    Runs inside the subprocess:
    1. One-shot initialization of torch and CUDA at startup
    2. Loop over tasks
    3. **Exit immediately on CUDA error**
    4. Clean up GPU memory after each task
    """
    init_start = time.time()

    try:
        # ====================================================================
        # Step 1: one-shot initialization (executed exactly once!)
        # ====================================================================

        # Import dependencies
        import torch
        import torch.cuda

        from kernelgym.backend import get_backend
        from kernelgym.toolkit import get_toolkit

        # Initialize CUDA
        torch.cuda.init()
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)

        # Warm-up (ensure CUDA is fully initialized)
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()

        toolkit_cache: dict[str, Any] = {}
        backend_cache: dict[str, Any] = {}

        init_time = time.time() - init_start

        # Notify the main process: init succeeded
        result_queue.put({"status": "READY", "init_time": init_time, "device": str(device)})

        # Log
        logger.info(
            "[%s] Initialized successfully (device=%s, init_time=%.2fs)", worker_id, device, init_time
        )

        # ====================================================================
        # Step 2: task-processing loop
        # ====================================================================

        tasks_processed = 0

        while True:
            try:
                # Fetch the task
                task_data = task_queue.get()

                # Check for shutdown command
                if isinstance(task_data, dict) and task_data.get("command") == "SHUTDOWN":
                    logger.info("[%s] Received SHUTDOWN command", worker_id)
                    break

                # Execute the task
                result = _execute_task_in_worker(
                    task_data,
                    device,
                    toolkit_cache,
                    backend_cache,
                    get_toolkit,
                    get_backend,
                )
                # Return the result
                result_queue.put(result)

                tasks_processed += 1

                # GPU-memory cleanup (after every task)
                try:
                    # Force-clear VRAM
                    _aggressive_gpu_cleanup(device_id)
                except Exception as cleanup_error:
                    logger.warning("[%s] GPU cleanup warning: %s", worker_id, cleanup_error)

            except Exception as task_error:
                # Task execution failed
                error_type = type(task_error).__name__
                error_message = str(task_error)

                # **Critical: check whether it's a CUDA error**
                is_cuda_error = (
                    "CUDA" in error_type
                    or "CUDA" in error_message
                    or "cuda" in error_message.lower()
                    or error_type in ["RuntimeError", "CudaError"]
                )
                is_profiler_error = "PROFILER_NO_CUDA_EVENTS" in error_message

                if is_cuda_error or is_profiler_error:
                    # CUDA error / profiler dropout! Prepare to exit
                    logger.error(
                        "[%s] CUDA/profiler error detected! Worker will exit. Error: %s: %s",
                        worker_id,
                        error_type,
                        error_message,
                    )

                    # Return an error result and mark the worker as exiting
                    result_queue.put(
                        {
                            "success": False,
                            "error_type": error_type,
                            "error_message": error_message,
                            "traceback": traceback.format_exc(),
                            "worker_exiting": True,  # critical flag!
                            "cuda_error": is_cuda_error,
                            "profiling_error": is_profiler_error,
                        }
                    )

                    # **Critical: force-clean VRAM before exiting on CUDA error**
                    logger.warning("[%s] Performing aggressive GPU cleanup before exit...", worker_id)
                    try:
                        _aggressive_gpu_cleanup(device_id)
                        logger.info("[%s] GPU cleanup completed", worker_id)
                    except Exception as cleanup_err:
                        logger.error(
                            "[%s] GPU cleanup failed (expected after CUDA error): %s",
                            worker_id,
                            cleanup_err,
                        )

                    # Try a final sync (may fail, but try)
                    try:
                        torch.cuda.synchronize()
                        logger.info("[%s] Final CUDA sync before exit", worker_id)
                    except Exception:
                        pass

                    # Exit the loop immediately
                    break

                else:
                    # Non-CUDA error: return error but keep running
                    logger.warning(
                        "[%s] Task error (non-CUDA): %s: %s",
                        worker_id,
                        error_type,
                        error_message,
                    )

                    result_queue.put(
                        {
                            "success": False,
                            "error_type": error_type,
                            "error_message": error_message,
                            "traceback": traceback.format_exc(),
                            "worker_exiting": False,
                            "cuda_error": False,
                        }
                    )

        # Normal exit — clean VRAM
        logger.info(
            "[%s] Worker exiting normally (processed %s tasks)",
            worker_id,
            tasks_processed,
        )

        # **Critical: clean VRAM on normal exit too**
        logger.info("[%s] Performing final GPU cleanup...", worker_id)
        try:
            _aggressive_gpu_cleanup(device_id)
            logger.info("[%s] Final GPU cleanup completed", worker_id)
        except Exception as cleanup_err:
            logger.error("[%s] Final GPU cleanup failed: %s", worker_id, cleanup_err)

        # **Additionally: try resetting the CUDA context (fully release on exit)**
        try:
            import torch

            # This auto-runs CUDA cleanup on process exit, but we call it
            # explicitly to be sure.
            torch.cuda.synchronize()
            logger.info("[%s] CUDA context synchronized before exit", worker_id)
        except Exception as cuda_cleanup_err:
            logger.error("[%s] CUDA synchronize failed: %s", worker_id, cuda_cleanup_err)

    except Exception as init_error:
        # Initialization failed
        logger.error(
            "[%s] Initialization failed: %s",
            worker_id,
            init_error,
        )
        traceback.print_exc(file=sys.stderr)

        result_queue.put(
            {"status": "INIT_FAILED", "error": str(init_error), "traceback": traceback.format_exc()}
        )


def _execute_task_in_worker(
    task_data: dict[str, Any],
    device: Any,  # torch.device
    toolkit_cache: dict[str, Any],
    backend_cache: dict[str, Any],
    get_toolkit: Any,
    get_backend: Any,
) -> dict[str, Any]:
    """
    Execute a single task inside the worker.

    Args:
        task_data: Task data dict.
        device: torch.device.
        toolkit: KernelBench-integration module.

    Returns:
        Result dict.
    """

    def _has_no_cuda_events(result_obj: Any) -> bool:
        """Detect profiler dropouts where no CUDA events were captured."""
        try:
            metadata = None
            if isinstance(result_obj, dict):
                metadata = result_obj.get("metadata")
            else:
                metadata = getattr(result_obj, "metadata", None)
            if not isinstance(metadata, dict):
                return False
            profiling = metadata.get("profiling")
            if not isinstance(profiling, dict):
                return False
            return profiling.get("profiling_warning") == "no_cuda_events"
        except Exception:
            return False

    try:
        toolkit_name = task_data.get("toolkit")
        backend_adapter = task_data.get("backend_adapter")
        if not toolkit_name:
            raise ValueError("Task payload missing required 'toolkit'")
        if not backend_adapter:
            raise ValueError("Task payload missing required 'backend_adapter'")

        if toolkit_name not in toolkit_cache:
            toolkit_cache[toolkit_name] = get_toolkit(toolkit_name)
        if backend_adapter not in backend_cache:
            backend_cache[backend_adapter] = get_backend(backend_adapter)

        task_data["device"] = str(device)

        toolkit = toolkit_cache[toolkit_name]
        backend = backend_cache[backend_adapter]
        result = toolkit.evaluate(task_data, backend=backend)

        if isinstance(result, dict):
            status = result.get("status")
            error_msg = result.get("error_message")
        else:
            status = getattr(result, "status", None)
            error_msg = getattr(result, "error_message", None)

        if status == "failed" and error_msg:
            if (
                "CUDA" in error_msg
                or "cuda" in error_msg.lower()
                or "illegal memory access" in error_msg.lower()
                or "device-side assert" in error_msg.lower()
            ):
                raise RuntimeError(f"CUDA error detected: {error_msg}")

        if _has_no_cuda_events(result):
            raise RuntimeError("PROFILER_NO_CUDA_EVENTS")

        return {
            "success": True,
            "result": result,
            "worker_exiting": False,
        }

    except Exception:
        # The exception here is caught by the caller and classified as CUDA vs non-CUDA.
        raise
