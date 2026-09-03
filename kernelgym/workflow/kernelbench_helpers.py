"""KernelBench workflow helpers (server-side orchestration utilities)."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

from .kernelbench_types import (
    EvaluationTask,
    KernelEvaluationResult,
    KernelEvaluationTask,
    EvaluationResult,
    ReferenceTimingResult,
    ReferenceTimingTask,
)

_reference_cache: Any = None


class InMemoryReferenceCache:
    """2026-08-29 P0 深修(K3 §20 Q2): reference 分母缓存, 固定分母跨 run 复用.

    A4 本意: 判分不重跑 reference 分母, 且消除 run 级 reference 计时方差。
    根因: gs300 方差 std=23.6, 同一 reference 不同 run 差 42%(机器级抖动)。
    缓存 key = (uuid, reference_code_hash, is_valid) —— 同题同 reference 同 validation 状态复用分母。
    注意: 缓存只在本 server 进程内有效(server 重启缓存清, 分母重新计时)。
    """
    def __init__(self) -> None:
        self._store: dict = {}

    def _key(self, uuid: Optional[str], reference_code: str, is_valid: bool):
        ref_hash = hashlib.md5((reference_code or "").encode()).hexdigest()[:16]
        return (uuid, ref_hash, bool(is_valid))

    def get(self, uuid: Optional[str], reference_code: str, is_valid: bool) -> Optional[float]:
        if not uuid:
            return None
        return self._store.get(self._key(uuid, reference_code, is_valid))

    def put(self, uuid: Optional[str], reference_code: str, is_valid: bool,
            runtime: float) -> None:
        if not uuid or runtime is None:
            return
        self._store[self._key(uuid, reference_code, is_valid)] = float(runtime)


_reference_cache: Any = None


def set_reference_cache(cache: Any) -> None:
    """Register a reference cache provider used by workflow orchestration."""
    global _reference_cache
    _reference_cache = cache


def _get_cached_reference_runtime(
    uuid: Optional[str], reference_code: str, is_valid: bool
) -> Optional[float]:
    if _reference_cache is None:
        return None
    return _reference_cache.get(uuid, reference_code, is_valid)


def _put_reference_cache(uuid: Optional[str], reference_code: str, is_valid: bool,
                         runtime: float) -> None:
    """PUT reference 分母进缓存(2026-08-29 P0 深修): reference 跑完后写, 跨 run 固定分母复用."""
    if _reference_cache is None:
        return
    try:
        _reference_cache.put(uuid, reference_code, is_valid, runtime)
    except Exception as e:
        logger.error("[refcache] put error: %s", e)


def _validate_code(code: str, entry_point: str = "Model") -> Tuple[bool, str]:
    try:
        if not code:
            return False, "Code is required"
        if f"class {entry_point}" not in code:
            return False, f"Code must contain a '{entry_point}' class"
        return True, ""
    except Exception as exc:
        return False, f"Code validation error: {exc}"


def _create_paired_tasks(
    task: EvaluationTask,
) -> Tuple[Optional[ReferenceTimingTask], KernelEvaluationTask]:
    ref_device = task.device_preference or task.device
    kernel_device = task.device

    reference_task: Optional[ReferenceTimingTask] = None
    if task.use_reference_cache and task.uuid:
        cached_runtime = _get_cached_reference_runtime(
            task.uuid, task.reference_code, task.is_valid
        )
        if cached_runtime is None:
            reference_task = ReferenceTimingTask(
                task_id=f"{task.task_id}_ref",
                base_task_id=task.task_id,
                reference_code=task.reference_code,
                toolkit=task.toolkit,
                backend_adapter=task.backend_adapter,
                backend=task.backend,
                num_perf_trials=task.num_perf_trials,
                timeout=task.timeout,
                device=ref_device,
                priority=task.priority,
                entry_point=task.entry_point,
                reference_backend=task.reference_backend,
                device_preference=task.device_preference,
                resources=task.resources,
            )
    else:
        reference_task = ReferenceTimingTask(
            task_id=f"{task.task_id}_ref",
            base_task_id=task.task_id,
            reference_code=task.reference_code,
            toolkit=task.toolkit,
            backend_adapter=task.backend_adapter,
            backend=task.backend,
            num_perf_trials=task.num_perf_trials,
            timeout=task.timeout,
            device=ref_device,
            priority=task.priority,
            entry_point=task.entry_point,
            reference_backend=task.reference_backend,
            device_preference=task.device_preference,
            resources=task.resources,
        )

    kernel_task = KernelEvaluationTask(
        task_id=f"{task.task_id}_kernel",
        base_task_id=task.task_id,
        reference_code=task.reference_code,
        kernel_code=task.kernel_code,
        toolkit=task.toolkit,
        backend_adapter=task.backend_adapter,
        backend=task.backend,
        num_correct_trials=task.num_correct_trials,
        num_perf_trials=task.num_perf_trials,
        timeout=task.timeout,
        device=kernel_device,
        priority=task.priority,
        entry_point=task.entry_point,
        device_preference=task.device_preference,
        enable_profiling=task.enable_profiling,
        enable_triton_detection=task.enable_triton_detection,
        measure_performance=task.measure_performance,
        run_correctness=task.run_correctness,
        run_triton_detection=task.run_triton_detection,
        run_performance=task.run_performance,
        resources=task.resources,
    )

    return reference_task, kernel_task


def _combine_results(
    reference_result: ReferenceTimingResult,
    kernel_result: KernelEvaluationResult,
) -> EvaluationResult:
    if reference_result.base_task_id != kernel_result.base_task_id:
        raise ValueError(
            f"Task ID mismatch: {reference_result.base_task_id} != {kernel_result.base_task_id}"
        )
    return EvaluationResult.from_paired_results(
        reference_result.base_task_id, reference_result, kernel_result
    )
