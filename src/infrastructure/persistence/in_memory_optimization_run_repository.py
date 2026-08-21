"""OptimizationRunRepository 의 인메모리 구현."""

from __future__ import annotations

from collections.abc import Sequence

from domain.optimization.identifiers import OptimizationRunId
from domain.optimization.optimization_run import OptimizationRun


class InMemoryOptimizationRunRepository:
    """domain.optimization.ports.OptimizationRunRepository 구현."""

    def __init__(self) -> None:
        self._items: dict[str, OptimizationRun] = {}

    def save(self, run: OptimizationRun) -> None:
        self._items[str(run.id)] = run

    def find_by_id(self, run_id: OptimizationRunId) -> OptimizationRun | None:
        return self._items.get(str(run_id))

    def exists(self, run_id: OptimizationRunId) -> bool:
        return str(run_id) in self._items

    def list_all(self) -> Sequence[OptimizationRun]:
        return tuple(self._items.values())

    def clear(self) -> None:
        self._items.clear()
