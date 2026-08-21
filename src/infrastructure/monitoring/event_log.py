"""Domain Event 를 로그로 내보낸다.

지금은 structlog 로 찍지만, 이 자리에 SNS / EventBridge / Kafka 가 들어와도
Application 코드는 한 줄도 바뀌지 않는다. 그게 Port 를 둔 이유다.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime

import structlog

from domain.shared.events import DomainEvent


class SystemClock:
    """application.shared.ports.Clock 구현."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class StructlogEventPublisher:
    """application.shared.ports.EventPublisher 구현."""

    def __init__(self, clock: SystemClock | None = None) -> None:
        self._clock = clock or SystemClock()
        self._logger = structlog.get_logger("domain.event")

    def publish(self, events: Iterable[DomainEvent]) -> None:
        occurred_at = self._clock.now().isoformat()
        for event in events:
            self._logger.info(
                event.event_name,
                occurred_at=occurred_at,
                **_payload(event),
            )


class RecordingEventPublisher:
    """테스트/실습용. 발행된 Event 를 그대로 모아 둔다."""

    def __init__(self) -> None:
        self.published: list[DomainEvent] = []

    def publish(self, events: Iterable[DomainEvent]) -> None:
        self.published.extend(events)

    def names(self) -> tuple[str, ...]:
        return tuple(e.event_name for e in self.published)

    def clear(self) -> None:
        self.published.clear()


def _payload(event: DomainEvent) -> dict[str, object]:
    if not is_dataclass(event):  # pragma: no cover - DomainEvent 는 모두 dataclass 다
        return {}
    return {key: _stringify(value) for key, value in asdict(event).items()}


def _stringify(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_stringify(v) for v in value]
    return str(value)
