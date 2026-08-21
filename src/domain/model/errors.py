"""Model Context 고유 예외."""

from __future__ import annotations

from domain.shared.errors import DomainException, EntityNotFound, InvariantViolation


class TrainingRunNotFound(EntityNotFound):
    code = "TRAINING_RUN_NOT_FOUND"


class TrainingRunAlreadyExists(DomainException):
    code = "TRAINING_RUN_ALREADY_EXISTS"


class ShapeMismatch(InvariantViolation):
    """모델이 기대하는 입력 모양과 데이터가 주는 모양이 다르다."""

    code = "SHAPE_MISMATCH"


class ModelNotTrained(DomainException):
    """아직 학습되지 않은 모델을 평가하거나 배포하려 했다."""

    code = "MODEL_NOT_TRAINED"
