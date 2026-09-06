"""Core data models for KernelGym refactor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Artifact:
    name: str
    uri: str | None = None
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    unit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    task_id: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    metrics: list[Metric] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    error_message: str | None = None


@dataclass
class TaskSpec:
    """Minimal task spec for scheduler submission."""

    kind: str
    payload: dict[str, Any]
    resources: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskGroup:
    """Container for multiple tasks with optional dependencies."""

    tasks: list[TaskSpec]
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
