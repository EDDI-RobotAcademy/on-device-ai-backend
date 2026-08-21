"""Data Quality Context 고유 예외."""

from __future__ import annotations

from domain.shared.errors import (
    DomainException,
    EntityNotFound,
    InvariantViolation,
)


class AssessmentNotFound(EntityNotFound):
    code = "ASSESSMENT_NOT_FOUND"


class AssessmentAlreadyExists(DomainException):
    code = "ASSESSMENT_ALREADY_EXISTS"


class DimensionNotMeasured(DomainException):
    """측정하지 않은 차원을 참조했다."""

    code = "DIMENSION_NOT_MEASURED"


class UnverifiedRemediation(InvariantViolation):
    """조치는 했는데 재측정하지 않았다."""

    code = "UNVERIFIED_REMEDIATION"
