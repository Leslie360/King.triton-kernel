"""Workflow controller abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .scheduler import SchedulerAPI


@dataclass
class WorkflowState:
    data: dict[str, Any] = field(default_factory=dict)


class WorkflowController(ABC):
    @abstractmethod
    async def handle_request(self, input_data: dict[str, Any], scheduler: SchedulerAPI) -> dict[str, Any]:
        """Run the workflow and return the final response payload."""

    async def validate_request(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Optional request validation hook."""
        return {"valid": True}

    async def on_task_finished(
        self,
        state: WorkflowState,
        task_id: str,
        result: dict[str, Any],
        scheduler: SchedulerAPI,
    ) -> dict[str, Any] | None:
        """Optional hook for incremental decision making."""
        return None

    async def aggregate(self, state: WorkflowState) -> dict[str, Any]:
        """Aggregate state into a final response payload."""
        return dict(state.data)
