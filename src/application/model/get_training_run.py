"""GetTrainingRun / ListTrainingRuns — Job 상태 조회. (CLAUDE.md §11)

학습이 도는 동안 클라이언트가 물어보는 곳이다.

    POST /training-runs/{id}/start  → 즉시 RUNNING
    GET  /training-runs/{id}        → RUNNING … COMPLETED
"""

from __future__ import annotations

from dataclasses import dataclass

from application.model.dto import TrainingCurveView, TrainingRunView
from application.model.support import load_run
from domain.model.ports import TrainingRunRepository


@dataclass(frozen=True, slots=True)
class GetTrainingRunQuery:
    run_id: str


class GetTrainingRun:
    def __init__(self, runs: TrainingRunRepository) -> None:
        self._runs = runs

    def execute(self, query: GetTrainingRunQuery) -> TrainingRunView:
        return TrainingRunView.of(load_run(self._runs, query.run_id))


class GetTrainingCurve:
    def __init__(self, runs: TrainingRunRepository) -> None:
        self._runs = runs

    def execute(self, query: GetTrainingRunQuery) -> TrainingCurveView:
        run = load_run(self._runs, query.run_id)
        return TrainingCurveView.of(str(run.id), run.status.value, run.curve)


class ListTrainingRuns:
    def __init__(self, runs: TrainingRunRepository) -> None:
        self._runs = runs

    def execute(self) -> tuple[TrainingRunView, ...]:
        return tuple(TrainingRunView.of(r) for r in self._runs.list_all())
