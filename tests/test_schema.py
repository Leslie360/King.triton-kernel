"""Tests for kernelgym/schema/.

Covers:
- serialization.make_json_safe / coerce_error_code / serialize_error_code
- task.py to_dict/from_dict round-trips (incl. unknown-field filtering)
- result.py speedup computation + from_paired_results failure passthrough
"""

from __future__ import annotations

import ast
from enum import Enum

from kernelgym.common import ErrorCode
from kernelgym.schema.result import (
    EvaluationResult,
    KernelEvaluationResult,
    ReferenceTimingResult,
)
from kernelgym.schema.serialization import (
    coerce_error_code,
    make_json_safe,
    serialize_error_code,
)
from kernelgym.schema.task import EvaluationTask, KernelEvaluationTask, ReferenceTimingTask


# ---------------------------------------------------------------------------
# make_json_safe
# ---------------------------------------------------------------------------
def test_make_json_safe_primitives():
    assert make_json_safe(None) is None
    assert make_json_safe(1) == 1
    assert make_json_safe(1.5) == 1.5
    assert make_json_safe(True) is True
    assert make_json_safe("x") == "x"


def test_make_json_safe_enum_to_value():
    assert make_json_safe(ErrorCode.TIMEOUT_ERROR) == "TIMEOUT_ERROR"


def test_make_json_safe_dict_enum_values():
    obj = {"status": ErrorCode.RUNTIME_ERROR, "n": 3}
    assert make_json_safe(obj) == {"status": "RUNTIME_ERROR", "n": 3}


def test_make_json_safe_dict_keys_strified():
    obj = {1: "a"}
    assert make_json_safe(obj) == {"1": "a"}


def test_make_json_safe_collections():
    assert make_json_safe([1, (2, 3), {4}]) == [1, [2, 3], [4]]
    assert make_json_safe((1, 2)) == [1, 2]


def test_make_json_safe_ast():
    node = ast.parse("x = 1").body[0]
    assert make_json_safe(node) == "x = 1"


def test_make_json_safe_unknown_to_str():
    class Weird:
        def __str__(self):
            return "weird-obj"

    assert make_json_safe(Weird()) == "weird-obj"


def test_make_json_safe_nested():
    obj = {"meta": {"code": ErrorCode.SYSTEM_ERROR, "extra": [1, None]}}
    assert make_json_safe(obj) == {"meta": {"code": "SYSTEM_ERROR", "extra": [1, None]}}


# ---------------------------------------------------------------------------
# coerce_error_code / serialize_error_code
# ---------------------------------------------------------------------------
def test_coerce_error_code_none():
    assert coerce_error_code(None) is None


def test_coerce_error_code_enum_passthrough():
    assert coerce_error_code(ErrorCode.TIMEOUT_ERROR) is ErrorCode.TIMEOUT_ERROR


def test_coerce_error_code_string_valid():
    assert coerce_error_code("TIMEOUT_ERROR") is ErrorCode.TIMEOUT_ERROR


def test_coerce_error_code_other_enum():
    class Other(str, Enum):
        A = "COMPILATION_ERROR"

    assert coerce_error_code(Other.A) is ErrorCode.COMPILATION_ERROR


def test_coerce_error_code_string_invalid_returns_str():
    assert coerce_error_code("NOT_A_REAL_CODE") == "NOT_A_REAL_CODE"


def test_coerce_error_code_other_enum_unmapped():
    class Other(str, Enum):
        A = "some_custom_value"

    assert coerce_error_code(Other.A) == "some_custom_value"


def test_coerce_error_code_non_mapping():
    assert coerce_error_code(123) == 123


def test_serialize_error_code_enum():
    assert serialize_error_code(ErrorCode.CORRECTNESS_ERROR) == "CORRECTNESS_ERROR"


def test_serialize_error_code_plain():
    assert serialize_error_code("plain") == "plain"
    assert serialize_error_code(None) is None


# ---------------------------------------------------------------------------
# EvaluationTask round-trip
# ---------------------------------------------------------------------------
def test_evaluation_task_roundtrip():
    task = EvaluationTask(
        task_id="t1",
        reference_code="ref",
        kernel_code="kern",
        uuid="abc-123",
        is_valid=True,
        resources={"gpu": 1},
    )
    d = task.to_dict()
    assert d["task_id"] == "t1"
    assert d["uuid"] == "abc-123"
    assert d["is_valid"] is True

    restored = EvaluationTask.from_dict(d)
    assert restored == task


def test_evaluation_task_from_dict_filters_unknown():
    d = EvaluationTask(task_id="t1", reference_code="r", kernel_code="k").to_dict()
    d["unknown_field"] = "should_be_dropped"
    restored = EvaluationTask.from_dict(d)
    assert not hasattr(restored, "unknown_field")
    assert restored.task_id == "t1"


def test_reference_timing_task_roundtrip():
    task = ReferenceTimingTask(task_id="t_ref", base_task_id="t", reference_code="r")
    assert ReferenceTimingTask.from_dict(task.to_dict()) == task


def test_kernel_evaluation_task_roundtrip():
    task = KernelEvaluationTask(
        task_id="t_ker", base_task_id="t", reference_code="r", kernel_code="k"
    )
    assert KernelEvaluationTask.from_dict(task.to_dict()) == task


# ---------------------------------------------------------------------------
# EvaluationResult.from_kernel_exec_result — speedup
# ---------------------------------------------------------------------------
def test_from_kernel_exec_result_speedup():
    class _R:
        compiled = True
        correctness = True
        decoy_kernel = False
        runtime = 10.0
        metadata = {}

    res = EvaluationResult.from_kernel_exec_result("t", _R(), reference_runtime=20.0)
    assert res.speedup == 2.0
    assert res.correctness is True
    assert res.status == "completed"


def test_from_kernel_exec_result_speedup_zero_when_wrong():
    class _R:
        compiled = True
        correctness = False
        decoy_kernel = False
        runtime = 5.0
        metadata = {}

    res = EvaluationResult.from_kernel_exec_result("t", _R(), reference_runtime=20.0)
    assert res.speedup == 0.0


def test_from_kernel_exec_result_speedup_zero_on_bad_runtimes():
    class _R:
        compiled = True
        correctness = True
        decoy_kernel = False
        runtime = 0.0
        metadata = {}

    res = EvaluationResult.from_kernel_exec_result("t", _R(), reference_runtime=20.0)
    assert res.speedup == 0.0

    class _R2:
        compiled = True
        correctness = True
        decoy_kernel = False
        runtime = 10.0
        metadata = {}

    res2 = EvaluationResult.from_kernel_exec_result("t", _R2(), reference_runtime=0.0)
    assert res2.speedup == 0.0


# ---------------------------------------------------------------------------
# EvaluationResult.from_paired_results — speedup + failure passthrough
# ---------------------------------------------------------------------------
def _mk_ref(status="completed", runtime=20.0, err_msg=None, err_code=None):
    return ReferenceTimingResult(
        task_id="t_ref",
        base_task_id="t",
        reference_runtime=runtime,
        metadata={},
        status=status,
        error_message=err_msg,
        error_code=err_code,
    )


def _mk_kernel(status="completed", correctness=True, runtime=10.0, err_msg=None,
               err_code=None, compiled=True):
    return KernelEvaluationResult(
        task_id="t_kernel",
        base_task_id="t",
        compiled=compiled,
        correctness=correctness,
        decoy_kernel=False,
        kernel_runtime=runtime,
        metadata={},
        status=status,
        error_message=err_msg,
        error_code=err_code,
    )


def test_from_paired_results_speedup():
    res = EvaluationResult.from_paired_results("t", _mk_ref(), _mk_kernel())
    assert res.task_id == "t"
    assert res.speedup == 2.0
    assert res.status == "completed"
    assert res.error_message is None
    assert res.error_code is None
    # combined metadata carries both task ids
    assert res.metadata["reference_task_id"] == "t_ref"
    assert res.metadata["kernel_task_id"] == "t_kernel"


def test_from_paired_results_reference_failure_passthrough():
    ref = _mk_ref(status="failed", runtime=0.0, err_msg="boom ref",
                  err_code=ErrorCode.RUNTIME_ERROR)
    res = EvaluationResult.from_paired_results("t", ref, _mk_kernel())
    assert res.status == "failed"
    assert res.error_message == "Reference timing failed: boom ref"
    assert res.error_code is ErrorCode.RUNTIME_ERROR
    # speedup must be 0 because reference failed
    assert res.speedup == 0.0


def test_from_paired_results_kernel_failure_passthrough():
    kern = _mk_kernel(status="failed", correctness=False, runtime=0.0,
                      err_msg="kernel crash", err_code=ErrorCode.COMPILATION_ERROR)
    res = EvaluationResult.from_paired_results("t", _mk_ref(), kern)
    assert res.status == "failed"
    assert res.error_message == "Kernel evaluation failed: kernel crash"
    assert res.error_code is ErrorCode.COMPILATION_ERROR
    assert res.speedup == 0.0


def test_from_paired_results_reference_failure_takes_precedence():
    ref = _mk_ref(status="failed", runtime=0.0, err_msg="ref fail",
                  err_code=ErrorCode.SYSTEM_ERROR)
    kern = _mk_kernel(status="failed", err_msg="kernel fail",
                      err_code=ErrorCode.RUNTIME_ERROR)
    res = EvaluationResult.from_paired_results("t", ref, kern)
    # reference failure checked first
    assert res.status == "failed"
    assert "Reference timing failed" in res.error_message


def test_from_paired_results_no_speedup_when_wrong():
    kern = _mk_kernel(correctness=False)
    res = EvaluationResult.from_paired_results("t", _mk_ref(), kern)
    assert res.speedup == 0.0


def test_evaluation_result_roundtrip():
    res = EvaluationResult(
        task_id="t",
        compiled=True,
        correctness=True,
        decoy_kernel=False,
        reference_runtime=20.0,
        kernel_runtime=10.0,
        speedup=2.0,
        metadata={"a": 1},
        status="completed",
    )
    d = res.to_dict()
    assert d["speedup"] == 2.0
    assert EvaluationResult.from_dict(d) == res


def test_evaluation_result_from_dict_coerces_error_code():
    d = {
        "task_id": "t",
        "compiled": True,
        "correctness": True,
        "decoy_kernel": False,
        "reference_runtime": 1.0,
        "kernel_runtime": 1.0,
        "speedup": 1.0,
        "metadata": {},
        "status": "failed",
        "error_message": "x",
        "error_code": "TIMEOUT_ERROR",
    }
    res = EvaluationResult.from_dict(d)
    assert res.error_code is ErrorCode.TIMEOUT_ERROR
