"""Workflow registry and lookup helpers."""

from __future__ import annotations

from kernelgym.core import Registry

from ..core.workflow import WorkflowController
from .kernel_simple import KernelSimpleWorkflowController
from .kernelbench import KernelBenchWorkflowController

_WORKFLOW_REGISTRY = Registry()
_WORKFLOW_REGISTRY.register("kernelbench", KernelBenchWorkflowController)
_WORKFLOW_REGISTRY.register("kernel_simple", KernelSimpleWorkflowController)


def get_workflow_controller(name: str) -> WorkflowController:
    key = (name or "kernelbench").strip().lower()
    return _WORKFLOW_REGISTRY.get(key)()


def register_workflow(name: str, controller_cls: type[WorkflowController]) -> None:
    key = name.strip().lower()
    _WORKFLOW_REGISTRY.register(key, controller_cls)


def list_workflows() -> dict[str, type[WorkflowController]]:
    return _WORKFLOW_REGISTRY.items()
