"""Shared task models for KernelBench workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EvaluationTask:
    task_id: str
    reference_code: str
    kernel_code: str
    toolkit: str = "kernelbench"
    backend_adapter: str = "kernelbench"
    backend: str = "triton"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    entry_point: str = "Model"
    reference_backend: str | None = None
    device_preference: str | None = None
    force_refresh: bool = False
    uuid: str | None = None
    use_reference_cache: bool = False
    is_valid: bool = False
    enable_profiling: bool | None = None
    enable_triton_detection: bool | None = None
    measure_performance: bool | None = None
    run_correctness: bool | None = None
    run_triton_detection: bool | None = None
    run_performance: bool | None = None
    resources: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationTask:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class ReferenceTimingTask:
    task_id: str
    base_task_id: str
    reference_code: str
    toolkit: str = "kernelbench"
    backend_adapter: str = "kernelbench"
    backend: str = "triton"
    num_perf_trials: int = 100
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    entry_point: str = "Model"
    reference_backend: str | None = None
    device_preference: str | None = None
    resources: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceTimingTask:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class KernelEvaluationTask:
    task_id: str
    base_task_id: str
    reference_code: str
    kernel_code: str
    toolkit: str = "kernelbench"
    backend_adapter: str = "kernelbench"
    backend: str = "triton"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    entry_point: str = "Model"
    device_preference: str | None = None
    enable_profiling: bool | None = None
    enable_triton_detection: bool | None = None
    measure_performance: bool | None = None
    run_correctness: bool | None = None
    run_triton_detection: bool | None = None
    run_performance: bool | None = None
    resources: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KernelEvaluationTask:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
