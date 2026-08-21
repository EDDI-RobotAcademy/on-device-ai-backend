"""Domain Event.

설계 결정:
    Event 에 발생 시각을 넣지 않는다.
    Domain 은 시계(now)를 몰라야 테스트가 결정적(deterministic)이 된다.
    발생 시각 부여는 Application Layer 가 Clock 을 통해 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """모든 Domain Event 의 표식(marker).

    주의: 여기에 `name` 같은 property 를 두면 하위 dataclass 가 `name: str` 필드를
    선언할 때 property 객체가 기본값으로 잡혀 정의가 깨진다. 그래서 `event_name` 이다.
    """

    @property
    def event_name(self) -> str:
        return type(self).__name__


class EventRecorder:
    """Aggregate 가 발생시킨 Event 를 모아 두는 Mixin.

    Application Layer 가 `pull_events()` 로 가져가 발행한다.
    가져가면 비워진다 — 같은 Event 를 두 번 발행하지 않기 위해서다.
    """

    __slots__ = ("_events",)

    def __init__(self) -> None:
        self._events: list[DomainEvent] = []

    def _record(self, event: DomainEvent) -> None:
        self._events.append(event)

    def pull_events(self) -> tuple[DomainEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    @property
    def pending_events(self) -> tuple[DomainEvent, ...]:
        """비우지 않고 들여다보기만 한다 (주로 테스트용)."""
        return tuple(self._events)


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """발행 시점에 시각/추적정보를 덧입힌 Event."""

    event: DomainEvent
    occurred_at: str
    correlation_id: str | None = field(default=None)
