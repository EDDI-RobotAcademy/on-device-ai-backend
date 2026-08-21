"""Optimization Use Case 들이 공유하는 최소한의 거들기."""

from __future__ import annotations

from application.shared.ports import EventPublisher
from domain.optimization.errors import OptimizationRunNotFound
from domain.optimization.identifiers import OptimizationRunId
from domain.optimization.optimization_run import OptimizationRun
from domain.optimization.ports import OptimizationRunRepository


def load_run(
    repository: OptimizationRunRepository, run_id: str | OptimizationRunId
) -> OptimizationRun:
    identifier = (
        run_id
        if isinstance(run_id, OptimizationRunId)
        else OptimizationRunId.of(run_id)
    )
    run = repository.find_by_id(identifier)
    if run is None:
        raise OptimizationRunNotFound(
            f"최적화 '{identifier}' 가 존재하지 않는다.", subject=str(identifier)
        )
    return run


def commit(
    repository: OptimizationRunRepository,
    run: OptimizationRun,
    publisher: EventPublisher | None = None,
) -> None:
    repository.save(run)
    events = run.pull_events()
    if publisher is not None and events:
        publisher.publish(events)
