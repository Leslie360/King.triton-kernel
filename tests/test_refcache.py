"""Tests for kernelgym/workflow/kernelbench_helpers.py.

Covers:
- InMemoryReferenceCache: get/put with key = (uuid, ref_hash, is_valid).
- _create_paired_tasks: ref task created only when cache miss; kernel task
  always created with _kernel suffix.
- _combine_results: base_task_id mismatch raises; otherwise merges.
"""

from __future__ import annotations

import hashlib

import pytest

from kernelgym.schema.result import KernelEvaluationResult, ReferenceTimingResult
from kernelgym.schema.task import EvaluationTask
from kernelgym.workflow.kernelbench_helpers import (
    InMemoryReferenceCache,
    _combine_results,
    _create_paired_tasks,
)


def _ref_hash(reference_code):
    return hashlib.md5((reference_code or "").encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# InMemoryReferenceCache
# ---------------------------------------------------------------------------
def test_cache_put_get_roundtrip():
    cache = InMemoryReferenceCache()
    cache.put("uuid-1", "ref-code-A", True, 12.5)
    assert cache.get("uuid-1", "ref-code-A", True) == 12.5


def test_cache_key_uses_ref_hash():
    # Same reference text -> same hash -> same key, so put with equal content hits.
    cache = InMemoryReferenceCache()
    cache.put("u", "ref-code", True, 9.0)
    assert cache.get("u", "ref-code", True) == 9.0


def test_cache_miss_on_different_reference():
    cache = InMemoryReferenceCache()
    cache.put("u", "ref-code-A", True, 9.0)
    assert cache.get("u", "ref-code-B", True) is None


def test_cache_miss_on_different_uuid():
    cache = InMemoryReferenceCache()
    cache.put("u1", "ref-code-A", True, 9.0)
    assert cache.get("u2", "ref-code-A", True) is None


def test_cache_miss_on_different_is_valid():
    cache = InMemoryReferenceCache()
    cache.put("u", "ref-code-A", True, 9.0)
    assert cache.get("u", "ref-code-A", False) is None


def test_cache_get_none_uuid():
    cache = InMemoryReferenceCache()
    cache.put("u", "ref-code", True, 9.0)
    assert cache.get(None, "ref-code", True) is None


def test_cache_put_none_uuid_noop():
    cache = InMemoryReferenceCache()
    cache.put(None, "ref-code", True, 9.0)
    assert cache.get("u", "ref-code", True) is None


def test_cache_put_none_runtime_noop():
    cache = InMemoryReferenceCache()
    cache.put("u", "ref-code", True, None)
    assert cache.get("u", "ref-code", True) is None


def test_cache_runtime_cast_to_float():
    cache = InMemoryReferenceCache()
    cache.put("u", "ref-code", True, "42.0")
    got = cache.get("u", "ref-code", True)
    assert got == 42.0
    assert isinstance(got, float)


def test_cache_is_valid_forced_bool_in_key():
    cache = InMemoryReferenceCache()
    # is_valid passed as 1/0 should be coerced via bool() so 1 and True collide.
    cache.put("u", "ref-code", 1, 5.0)
    assert cache.get("u", "ref-code", True) == 5.0


# ---------------------------------------------------------------------------
# _create_paired_tasks
# ---------------------------------------------------------------------------
def _base_task(**overrides):
    kwargs = dict(
        task_id="t1",
        reference_code="ref-code-A",
        kernel_code="kernel-code",
        uuid="uuid-1",
        is_valid=True,
    )
    kwargs.update(overrides)
    return EvaluationTask(**kwargs)


def test_create_paired_tasks_cache_miss_creates_ref_task():
    task = _base_task(use_reference_cache=True)
    ref_task, kernel_task = _create_paired_tasks(task)
    assert ref_task is not None
    assert ref_task.task_id == "t1_ref"
    assert ref_task.base_task_id == "t1"
    assert ref_task.reference_code == "ref-code-A"
    assert kernel_task.task_id == "t1_kernel"
    assert kernel_task.base_task_id == "t1"
    assert kernel_task.kernel_code == "kernel-code"


def test_create_paired_tasks_cache_hit_skips_ref_task():
    # Warm the global cache then re-create: reference should be skipped.
    from kernelgym.workflow import kernelbench_helpers as kh

    kh.set_reference_cache(InMemoryReferenceCache())
    kh._put_reference_cache("uuid-1", "ref-code-A", True, 7.5)

    task = _base_task(use_reference_cache=True)
    ref_task, kernel_task = _create_paired_tasks(task)
    assert ref_task is None  # cached runtime available -> no ref task needed
    assert kernel_task.task_id == "t1_kernel"

    # reset global cache
    kh.set_reference_cache(None)


def test_create_paired_tasks_no_uuid_always_creates_ref():
    task = _base_task(use_reference_cache=True, uuid=None)
    ref_task, _ = _create_paired_tasks(task)
    assert ref_task is not None


def test_create_paired_tasks_ref_device_preference():
    task = _base_task(use_reference_cache=False, device_preference="cuda:1",
                      device="cuda:0")
    ref_task, kernel_task = _create_paired_tasks(task)
    assert ref_task.device == "cuda:1"  # uses device_preference
    assert kernel_task.device == "cuda:0"  # kernel uses task.device


# ---------------------------------------------------------------------------
# _combine_results
# ---------------------------------------------------------------------------
def _ref(base="t1", status="completed", runtime=20.0):
    return ReferenceTimingResult(
        task_id="t1_ref", base_task_id=base, reference_runtime=runtime,
        metadata={}, status=status,
    )


def _kern(base="t1", correctness=True, runtime=10.0):
    return KernelEvaluationResult(
        task_id="t1_kernel", base_task_id=base, compiled=True,
        correctness=correctness, decoy_kernel=False, kernel_runtime=runtime,
        metadata={},
    )


def test_combine_results_speedup():
    res = _combine_results(_ref(), _kern())
    assert res.task_id == "t1"
    assert res.speedup == 2.0
    assert res.status == "completed"


def test_combine_results_mismatch_raises():
    with pytest.raises(ValueError):
        _combine_results(_ref(base="a"), _kern(base="b"))


def test_combine_results_failure_passthrough():
    ref = _ref(status="failed", runtime=0.0)
    ref.error_message = "ref died"
    ref.error_code = "RUNTIME_ERROR"
    res = _combine_results(ref, _kern())
    assert res.status == "failed"
    assert res.error_message == "Reference timing failed: ref died"
