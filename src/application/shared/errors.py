"""Application 예외.

Domain 예외와 구분한다. (CLAUDE.md §12)
    DomainException      — 규칙을 어겼다.
    ApplicationException — 규칙 이전에, Use Case 를 진행할 수 없다.
"""

from __future__ import annotations


class ApplicationException(Exception):
    code: str = "APPLICATION_ERROR"

    def __init__(self, message: str, *, subject: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.subject = subject


class ResourceNotFound(ApplicationException):
    code = "RESOURCE_NOT_FOUND"


class ConflictingRequest(ApplicationException):
    code = "CONFLICTING_REQUEST"


class UnsupportedOperation(ApplicationException):
    code = "UNSUPPORTED_OPERATION"
