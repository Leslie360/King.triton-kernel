"""Toolkit abstraction (evaluation logic)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..backend import Backend


class Toolkit(ABC):
    name: str = "unknown"

    @abstractmethod
    def evaluate(self, task: dict[str, Any], backend: Backend, **kwargs: Any) -> dict[str, Any]:
        """Run evaluation logic against a backend."""
