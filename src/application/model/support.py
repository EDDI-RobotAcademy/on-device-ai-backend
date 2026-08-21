"""Model Use Case 들이 공유하는 최소한의 거들기."""

from __future__ import annotations

from application.shared.ports import EventPublisher
from domain.model.errors import TrainingRunNotFound
from domain.model.identifiers import TrainingRunId
from domain.model.ports import TrainingRunRepository
from domain.model.training_run import TrainingRun


def load_run(
    repository: TrainingRunRepository, run_id: str | TrainingRunId
) -> TrainingRun:
    identifier = (
        run_id if isinstance(run_id, TrainingRunId) else TrainingRunId.of(run_id)
    )
    run = repository.find_by_id(identifier)
    if run is None:
        raise TrainingRunNotFound(
            f"학습 '{identifier}' 이 존재하지 않는다.", subject=str(identifier)
        )
    return run


def commit(
    repository: TrainingRunRepository,
    run: TrainingRun,
    publisher: EventPublisher | None = None,
) -> None:
    repository.save(run)
    events = run.pull_events()
    if publisher is not None and events:
        publisher.publish(events)
