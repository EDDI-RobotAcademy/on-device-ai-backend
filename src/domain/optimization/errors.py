"""Optimization Context 고유 예외."""

from __future__ import annotations

from domain.shared.errors import (
    DomainException,
    EntityNotFound,
    IllegalStateTransition,
    InvariantViolation,
)


class OptimizationRunNotFound(EntityNotFound):
    code = "OPTIMIZATION_RUN_NOT_FOUND"


class ArtifactNotFound(EntityNotFound):
    code = "ARTIFACT_NOT_FOUND"


class ConversionFailed(DomainException):
    """변환이 실패했다. 실패도 기록되어야 한다."""

    code = "CONVERSION_FAILED"


class EquivalenceNotVerified(InvariantViolation):
    """변환은 됐는데 같은 답을 내는지 확인하지 않았다."""

    code = "EQUIVALENCE_NOT_VERIFIED"


class NoCandidateSelected(IllegalStateTransition):
    """후보 없이 선택할 수 없다.

    형식이 틀린 요청이 아니라 **순서가 틀린 요청**이다. 그래서 상태 전이 오류다.
    """

    code = "NO_CANDIDATE_SELECTED"
