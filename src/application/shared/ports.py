"""Application 이 필요로 하는 기술 협력자.

Domain 이 시계를 모르게 하기 위해 Clock 은 여기에 둔다.
Event 발행도 마찬가지다. Domain 은 Event 를 만들 뿐, 어디로 보내는지 모른다.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

from domain.shared.events import DomainEvent


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


@runtime_checkable
class EventPublisher(Protocol):
    """Domain Event 를 바깥으로 내보낸다.

    지금은 로그로 나가지만, 나중에 SNS/EventBridge 로 바뀌어도
    Application 코드는 바뀌지 않는다.
    """

    def publish(self, events: Iterable[DomainEvent]) -> None: ...
