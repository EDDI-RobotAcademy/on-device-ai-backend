"""Fleet Use Case 들이 공유하는 최소한의 거들기."""

from __future__ import annotations

from application.shared.ports import Clock, EventPublisher
from domain.fleet.errors import FleetNotFound, RolloutNotFound
from domain.fleet.fleet import Fleet
from domain.fleet.identifiers import FleetId, RolloutId
from domain.fleet.ports import FleetRepository, RolloutRepository
from domain.fleet.rollout import Rollout


def load_fleet(repository: FleetRepository, fleet_id: str | FleetId) -> Fleet:
    identifier = fleet_id if isinstance(fleet_id, FleetId) else FleetId.of(fleet_id)
    fleet = repository.find_by_id(identifier)
    if fleet is None:
        raise FleetNotFound(
            f"플릿 '{identifier}' 가 존재하지 않는다.", subject=str(identifier)
        )
    return fleet


def load_rollout(repository: RolloutRepository, rollout_id: str | RolloutId) -> Rollout:
    identifier = (
        rollout_id if isinstance(rollout_id, RolloutId) else RolloutId.of(rollout_id)
    )
    rollout = repository.find_by_id(identifier)
    if rollout is None:
        raise RolloutNotFound(
            f"롤아웃 '{identifier}' 가 존재하지 않는다.", subject=str(identifier)
        )
    return rollout


def commit(repository, aggregate, publisher: EventPublisher | None = None) -> None:  # noqa: ANN001
    repository.save(aggregate)
    events = aggregate.pull_events()
    if publisher is not None and events:
        publisher.publish(events)


def moment(clock: Clock, override: str | None = None) -> str:
    """이 일이 '언제' 일어난 것으로 기록할 것인가.

    모듈 5 와 같은 이유로 열어 둔다 — 운영 기록은 데이터의 시각을 따라야 할 때가 있다.
    """
    if override:
        return override
    return clock.now().replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
