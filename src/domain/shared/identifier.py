"""식별자 Value Object 기반 클래스."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class Identifier:
    """문자열 식별자. 빈 값을 허용하지 않는다는 것 자체가 불변식이다."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise InvariantViolation(
                f"{type(self).__name__} 는 비어 있을 수 없다.",
                subject=type(self).__name__,
            )

    @classmethod
    def of(cls, value: str) -> Self:
        return cls(value.strip())

    def __str__(self) -> str:
        return self.value
