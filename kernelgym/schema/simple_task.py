"""Schema for kernel simple workflow tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class KernelSimpleTask:
    task_id: str
    kernel_code: str
    toolkit: str = "kernel_simple"
    backend_adapter: str = "kernelbench"
    backend: str = "triton"
    entry_point: str = "ModelNew"
    num_perf_trials: int = 100
    num_warmup: int = 3
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    run_correctness: bool | None = None
    run_performance: bool | None = None
    enable_profiling: bool | None = None
    cases_code: str | None = None
    cases: list[Any] | None = None
    resources: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KernelSimpleTask:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
