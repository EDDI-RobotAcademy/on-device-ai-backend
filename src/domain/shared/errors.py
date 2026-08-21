"""Domain 예외 계층.

규칙(CLAUDE.md §12):
    Domain 은 HTTPException 을 절대 만들지 않는다.
    HTTP 상태 코드로의 변환은 Interface Layer 의 책임이다.
"""

from __future__ import annotations


class DomainException(Exception):
    """모든 Domain 규칙 위반의 루트."""

    #: Interface Layer 가 응답 본문에 그대로 실어 보내는 안정적인 코드.
    code: str = "DOMAIN_ERROR"

    def __init__(self, message: str, *, subject: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.subject = subject

    def __str__(self) -> str:  # pragma: no cover - 디버깅 편의
        if self.subject:
            return f"[{self.code}] {self.subject}: {self.message}"
        return f"[{self.code}] {self.message}"


class InvariantViolation(DomainException):
    """Value Object / Entity 가 스스로 지켜야 할 불변식이 깨졌다."""

    code = "INVARIANT_VIOLATION"


class IllegalStateTransition(DomainException):
    """Aggregate 의 생명주기상 허용되지 않는 전이다."""

    code = "ILLEGAL_STATE_TRANSITION"


class EntityNotFound(DomainException):
    """식별자로 찾은 Aggregate 가 없다.

    Context 마다 구체 예외(DatasetNotFound, AssessmentNotFound …)를 두되
    이 클래스를 상속한다. Interface Layer 는 이것 하나만 404 로 매핑하면 된다.
    """

    code = "ENTITY_NOT_FOUND"
