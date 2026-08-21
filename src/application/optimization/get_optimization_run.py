"""조회 Use Case."""

from __future__ import annotations

from dataclasses import dataclass

from application.optimization.dto import OptimizationRunView, RooflineView, SelectionView
from application.optimization.support import load_run
from application.shared.errors import ResourceNotFound
from domain.optimization.ports import OptimizationRunRepository


@dataclass(frozen=True, slots=True)
class GetOptimizationRunQuery:
    run_id: str


class GetOptimizationRun:
    def __init__(self, runs: OptimizationRunRepository) -> None:
        self._runs = runs

    def execute(self, query: GetOptimizationRunQuery) -> OptimizationRunView:
        return OptimizationRunView.of(load_run(self._runs, query.run_id))


class GetOptimizationCertificate:
    def __init__(self, runs: OptimizationRunRepository) -> None:
        self._runs = runs

    def execute(self, query: GetOptimizationRunQuery) -> SelectionView:
        run = load_run(self._runs, query.run_id)
        if run.certificate is None:
            raise ResourceNotFound(
                "아직 선택하지 않았다. 판정 기록이 없다.", subject=str(run.id)
            )
        return SelectionView.of(str(run.id), run.certificate)


class GetRooflineProfile:
    def __init__(self, runs: OptimizationRunRepository) -> None:
        self._runs = runs

    def execute(self, query: GetOptimizationRunQuery) -> RooflineView:
        run = load_run(self._runs, query.run_id)
        if run.roofline is None:
            raise ResourceNotFound(
                "병목 프로파일이 없다.", subject=str(run.id)
            )
        return RooflineView.of(str(run.id), run.roofline)


class ListOptimizationRuns:
    def __init__(self, runs: OptimizationRunRepository) -> None:
        self._runs = runs

    def execute(self) -> tuple[OptimizationRunView, ...]:
        return tuple(OptimizationRunView.of(run) for run in self._runs.list_all())
