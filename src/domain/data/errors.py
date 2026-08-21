"""Data Context 고유 예외."""

from __future__ import annotations

from domain.shared.errors import (
    DomainException,
    EntityNotFound,
    InvariantViolation,
)


class DatasetNotFound(EntityNotFound):
    code = "DATASET_NOT_FOUND"


class DatasetAlreadyExists(DomainException):
    code = "DATASET_ALREADY_EXISTS"


class SchemaMismatch(InvariantViolation):
    """선언한 Schema 가 실제 데이터와 맞지 않는다."""

    code = "SCHEMA_MISMATCH"


class UnknownField(InvariantViolation):
    """Schema 에 존재하지 않는 필드를 참조했다."""

    code = "UNKNOWN_FIELD"


class DatasetNotReady(DomainException):
    """아직 학습에 쓸 수 있다고 판정되지 않은 Dataset 이다."""

    code = "DATASET_NOT_READY"
