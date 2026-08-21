"""Bounded Context 전반에서 공유하는 Domain Primitive."""

from domain.shared.errors import (
    DomainException,
    IllegalStateTransition,
    InvariantViolation,
)
from domain.shared.events import DomainEvent, EventRecorder
from domain.shared.identifier import Identifier

__all__ = [
    "DomainException",
    "DomainEvent",
    "EventRecorder",
    "Identifier",
    "IllegalStateTransition",
    "InvariantViolation",
]
