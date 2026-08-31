"""Tests for kernelgym/common.py enums: values and uniqueness."""

from __future__ import annotations

from kernelgym.common import Backend, ErrorCode, Priority, TaskStatus


def test_taskstatus_values():
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.PROCESSING.value == "processing"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.FAILED.value == "failed"
    assert TaskStatus.TIMEOUT.value == "timeout"


def test_taskstatus_members_unique():
    values = [m.value for m in TaskStatus]
    assert len(values) == len(set(values))
    assert len(TaskStatus) == 5


def test_taskstatus_is_str_enum():
    # Subclassing str makes it JSON-serializable natively.
    assert isinstance(TaskStatus.PENDING, str)
    assert TaskStatus("completed") is TaskStatus.COMPLETED


def test_backend_values():
    assert Backend.CUDA.value == "cuda"
    assert Backend.TRITON.value == "triton"
    assert len(Backend) == 2


def test_priority_values():
    assert Priority.LOW.value == "low"
    assert Priority.NORMAL.value == "normal"
    assert Priority.HIGH.value == "high"
    assert len(Priority) == 3


def test_errorcode_values():
    expected = {
        "VALIDATION_ERROR": "VALIDATION_ERROR",
        "COMPILATION_ERROR": "COMPILATION_ERROR",
        "RUNTIME_ERROR": "RUNTIME_ERROR",
        "CORRECTNESS_ERROR": "CORRECTNESS_ERROR",
        "TIMEOUT_ERROR": "TIMEOUT_ERROR",
        "SYSTEM_ERROR": "SYSTEM_ERROR",
        "RESOURCE_ERROR": "RESOURCE_ERROR",
        "UNKNOWN_ERROR": "UNKNOWN_ERROR",
        "SYNTAX_ERROR": "SYNTAX_ERROR",
        "IMPORT_ERROR": "IMPORT_ERROR",
        "INSTANTIATION_ERROR": "INSTANTIATION_ERROR",
    }
    for name, val in expected.items():
        member = getattr(ErrorCode, name)
        assert member.value == val
    assert len(ErrorCode) == len(expected)


def test_errorcode_members_unique():
    values = [m.value for m in ErrorCode]
    assert len(values) == len(set(values))


def test_errorcode_from_value_roundtrip():
    assert ErrorCode("TIMEOUT_ERROR") is ErrorCode.TIMEOUT_ERROR


def test_common_enums_are_str_subclasses():
    # All four enums are str-valued, so values are directly serializable.
    for enum_cls in (TaskStatus, Backend, Priority, ErrorCode):
        for member in enum_cls:
            assert isinstance(member.value, str)
