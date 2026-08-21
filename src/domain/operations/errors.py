"""Operations Context 고유 예외."""

from __future__ import annotations

from domain.shared.errors import (
    DomainException,
    EntityNotFound,
    IllegalStateTransition,
)


class DeploymentNotFound(EntityNotFound):
    code = "DEPLOYMENT_NOT_FOUND"


class HealthWatchNotFound(EntityNotFound):
    code = "HEALTH_WATCH_NOT_FOUND"


class VersionNotFound(EntityNotFound):
    """돌아갈 곳이 없으면 롤백이 아니다."""

    code = "DEPLOYMENT_VERSION_NOT_FOUND"


class NotDeployable(IllegalStateTransition):
    """모듈 4 의 선택 판정을 통과하지 않은 결과물."""

    code = "NOT_DEPLOYABLE"


class NoObservationRecorded(IllegalStateTransition):
    """관측 없이 판단할 수 없다."""

    code = "NO_OBSERVATION_RECORDED"


class InferenceLogUnusable(DomainException):
    """로그가 남아 있긴 한데 그것으로 답할 수 있는 질문이 없다."""

    code = "INFERENCE_LOG_UNUSABLE"
