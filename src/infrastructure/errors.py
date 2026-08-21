"""Infrastructure 예외."""

from __future__ import annotations


class InfrastructureException(Exception):
    code: str = "INFRASTRUCTURE_ERROR"

    def __init__(self, message: str, *, subject: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.subject = subject


class SourceUnreadable(InfrastructureException):
    """데이터 원본을 읽을 수 없다."""

    code = "SOURCE_UNREADABLE"


class UnsupportedSourceFormat(InfrastructureException):
    code = "UNSUPPORTED_SOURCE_FORMAT"
