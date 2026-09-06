"""Tests for kernelgym/utils/error_classifier.py regex-based classification.

The classifier scans error text (lowercased) with ordered regex groups and
returns the first matching ErrorCode. Order matters:
validation -> compilation -> runtime -> correctness -> timeout -> system
-> resource -> context fallback -> UNKNOWN.
"""

from __future__ import annotations

from kernelgym.common import ErrorCode
from kernelgym.utils.error_classifier import (
    classify_error,
    get_error_category,
    get_error_description,
)


class TestValidation:
    def test_empty_message_unknown(self):
        assert classify_error("") is ErrorCode.UNKNOWN_ERROR
        assert classify_error(None) is ErrorCode.UNKNOWN_ERROR

    def test_validation_failed(self):
        assert classify_error("validation failed: bad code") is ErrorCode.VALIDATION_ERROR

    def test_dangerous_pattern(self):
        assert classify_error("dangerous pattern detected in source") is ErrorCode.VALIDATION_ERROR

    def test_code_must_contain(self):
        assert classify_error("code must contain a Model class") is ErrorCode.VALIDATION_ERROR

    def test_missing_class(self):
        assert classify_error("missing class Model") is ErrorCode.VALIDATION_ERROR

    def test_invalid_entry_point(self):
        assert classify_error("invalid entry point specified") is ErrorCode.VALIDATION_ERROR


class TestCompilation:
    def test_compilation_failed(self):
        assert classify_error("compilation failed") is ErrorCode.COMPILATION_ERROR

    def test_syntax_error(self):
        assert classify_error("syntax error at line 3") is ErrorCode.COMPILATION_ERROR

    def test_nvcc_error(self):
        assert classify_error("nvcc error: identifier undefined") is ErrorCode.COMPILATION_ERROR

    def test_triton_compilation(self):
        assert classify_error("triton compilation failed") is ErrorCode.COMPILATION_ERROR

    def test_kernel_compilation(self):
        assert classify_error("kernel compilation error") is ErrorCode.COMPILATION_ERROR

    def test_build_failed(self):
        assert classify_error("build failed") is ErrorCode.COMPILATION_ERROR

    def test_linker_error(self):
        assert classify_error("linker error: undefined symbol") is ErrorCode.COMPILATION_ERROR


class TestRuntime:
    def test_runtime_error(self):
        assert classify_error("runtime error in kernel") is ErrorCode.RUNTIME_ERROR

    def test_kernel_execution_failed(self):
        assert classify_error("kernel execution failed") is ErrorCode.RUNTIME_ERROR

    def test_out_of_memory(self):
        assert classify_error("out of memory on device") is ErrorCode.RUNTIME_ERROR

    def test_cuda_error(self):
        assert classify_error("cuda error: illegal memory access") is ErrorCode.RUNTIME_ERROR

    def test_execution_failed(self):
        assert classify_error("execution failed") is ErrorCode.RUNTIME_ERROR


class TestCorrectness:
    def test_correctness_check_failed(self):
        assert classify_error("correctness check failed") is ErrorCode.CORRECTNESS_ERROR

    def test_output_mismatch(self):
        assert classify_error("output mismatch detected") is ErrorCode.CORRECTNESS_ERROR

    def test_result_incorrect(self):
        assert classify_error("result incorrect for sample 2") is ErrorCode.CORRECTNESS_ERROR

    def test_assertion_failed(self):
        assert classify_error("assertion failed") is ErrorCode.CORRECTNESS_ERROR

    def test_numerical_error(self):
        assert classify_error("numerical error in output") is ErrorCode.CORRECTNESS_ERROR


class TestTimeout:
    def test_timeout(self):
        assert classify_error("task timeout") is ErrorCode.TIMEOUT_ERROR

    def test_task_timed_out(self):
        assert classify_error("task timed out") is ErrorCode.TIMEOUT_ERROR

    def test_execution_timeout(self):
        assert classify_error("execution timeout reached") is ErrorCode.TIMEOUT_ERROR

    def test_time_limit_exceeded(self):
        assert classify_error("time limit exceeded") is ErrorCode.TIMEOUT_ERROR

    def test_hung_task(self):
        assert classify_error("hung task detected") is ErrorCode.TIMEOUT_ERROR


class TestSystem:
    def test_system_error(self):
        assert classify_error("system error occurred") is ErrorCode.SYSTEM_ERROR

    def test_redis_error(self):
        assert classify_error("redis error: connection refused") is ErrorCode.SYSTEM_ERROR

    def test_database_error(self):
        assert classify_error("database error") is ErrorCode.SYSTEM_ERROR

    def test_connection_failed(self):
        assert classify_error("connection failed") is ErrorCode.SYSTEM_ERROR

    def test_service_unavailable(self):
        assert classify_error("service unavailable") is ErrorCode.SYSTEM_ERROR


class TestResource:
    def test_resource_error(self):
        assert classify_error("resource error") is ErrorCode.RESOURCE_ERROR

    def test_insufficient_memory(self):
        assert classify_error("insufficient memory to schedule") is ErrorCode.RESOURCE_ERROR

    def test_queue_full(self):
        assert classify_error("queue full") is ErrorCode.RESOURCE_ERROR

    def test_no_available_workers(self):
        assert classify_error("no available workers") is ErrorCode.RESOURCE_ERROR

    def test_gpu_unavailable(self):
        assert classify_error("gpu unavailable") is ErrorCode.RESOURCE_ERROR

    def test_disk_space(self):
        assert classify_error("disk space exhausted") is ErrorCode.RESOURCE_ERROR


class TestContextFallback:
    def test_context_validation(self):
        # error text unknown, but context says "validation" -> VALIDATION_ERROR
        assert (
            classify_error("something odd happened", context="validation phase") is ErrorCode.VALIDATION_ERROR
        )

    def test_context_compilation(self):
        assert classify_error("odd error", context="during compilation") is ErrorCode.COMPILATION_ERROR

    def test_context_runtime(self):
        assert classify_error("odd error", context="runtime execution") is ErrorCode.RUNTIME_ERROR

    def test_context_correctness(self):
        assert classify_error("odd error", context="correctness check") is ErrorCode.CORRECTNESS_ERROR

    def test_context_timeout(self):
        assert classify_error("odd error", context="timeout handling") is ErrorCode.TIMEOUT_ERROR

    def test_context_system(self):
        assert classify_error("odd error", context="system setup") is ErrorCode.SYSTEM_ERROR

    def test_context_resource(self):
        assert classify_error("odd error", context="resource allocation") is ErrorCode.RESOURCE_ERROR


class TestUnknown:
    def test_unrecognized_message(self):
        assert classify_error("something totally weird happened here") is ErrorCode.UNKNOWN_ERROR

    def test_unrecognized_with_unmatched_context(self):
        # context doesn't match any keyword and message is unknown
        assert classify_error("odd error", context="post-processing") is ErrorCode.UNKNOWN_ERROR


class TestOrdering:
    def test_compilation_wins_over_runtime_text(self):
        # "runtime" appears but compilation pattern matches first
        assert classify_error("triton compilation failed at runtime") is ErrorCode.COMPILATION_ERROR

    def test_validation_wins_over_compilation(self):
        assert classify_error("code validation error before compile") is ErrorCode.VALIDATION_ERROR

    def test_case_insensitive(self):
        assert classify_error("Compilation Failed On Device") is ErrorCode.COMPILATION_ERROR


class TestDescriptionsAndCategories:
    def test_get_error_description(self):
        assert "validation" in get_error_description(ErrorCode.VALIDATION_ERROR).lower()

    def test_get_error_category(self):
        assert get_error_category(ErrorCode.VALIDATION_ERROR) == "input"
        assert get_error_category(ErrorCode.COMPILATION_ERROR) == "compilation"
        assert get_error_category(ErrorCode.RUNTIME_ERROR) == "runtime"
        assert get_error_category(ErrorCode.CORRECTNESS_ERROR) == "correctness"
        assert get_error_category(ErrorCode.TIMEOUT_ERROR) == "timeout"
        assert get_error_category(ErrorCode.SYSTEM_ERROR) == "system"
        assert get_error_category(ErrorCode.RESOURCE_ERROR) == "resource"
        assert get_error_category(ErrorCode.UNKNOWN_ERROR) == "unknown"
