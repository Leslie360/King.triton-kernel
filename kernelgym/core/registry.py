"""Simple registry for workflow/backends/toolkits."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Registry:
    _items: dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, obj: Any) -> None:
        if name in self._items:
            raise KeyError(f"Registry already contains '{name}'")
        self._items[name] = obj

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise KeyError(f"Registry missing '{name}'")
        return self._items[name]

    def list(self) -> Iterable[str]:
        return tuple(self._items.keys())

    def items(self) -> dict[str, Any]:
        return dict(self._items)
