"""TrainingRunRepository 의 인메모리 구현."""

from __future__ import annotations

from collections.abc import Sequence

from domain.model.identifiers import TrainingRunId
from domain.model.training_run import TrainingRun


class InMemoryTrainingRunRepository:
    """domain.model.ports.TrainingRunRepository 구현."""

    def __init__(self) -> None:
        self._items: dict[str, TrainingRun] = {}

    def save(self, run: TrainingRun) -> None:
        self._items[str(run.id)] = run

    def find_by_id(self, run_id: TrainingRunId) -> TrainingRun | None:
        return self._items.get(str(run_id))

    def exists(self, run_id: TrainingRunId) -> bool:
        return str(run_id) in self._items

    def list_all(self) -> Sequence[TrainingRun]:
        return tuple(self._items.values())

    def clear(self) -> None:
        self._items.clear()
