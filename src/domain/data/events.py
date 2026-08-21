"""Data Context 의 Domain Event.

"무슨 일이 있었는가"를 남긴다. 이 기록이 나중에 Observability 의 뿌리가 된다.
(운영 모듈에서 "AI가 언제부터 이상해졌는가"를 되짚을 때 필요한 것이 바로 이 흐름이다.)
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.data.identifiers import DatasetId
from domain.data.inspection import InspectionKind, Verdict
from domain.shared.events import DomainEvent


@dataclass(frozen=True, slots=True)
class DatasetRegistered(DomainEvent):
    dataset_id: DatasetId
    name: str
    collected_from: str


@dataclass(frozen=True, slots=True)
class DatasetProfiled(DomainEvent):
    dataset_id: DatasetId
    row_count: int
    column_count: int


@dataclass(frozen=True, slots=True)
class DataSchemaDeclared(DomainEvent):
    dataset_id: DatasetId
    field_count: int


@dataclass(frozen=True, slots=True)
class InspectionRecorded(DomainEvent):
    dataset_id: DatasetId
    kind: InspectionKind
    verdict: Verdict
    finding_count: int


@dataclass(frozen=True, slots=True)
class LabelSpaceDefined(DomainEvent):
    dataset_id: DatasetId
    class_count: int


@dataclass(frozen=True, slots=True)
class TrainingDataDesigned(DomainEvent):
    dataset_id: DatasetId
    input_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DatasetPartitioned(DomainEvent):
    dataset_id: DatasetId
    strategy: str
    train_count: int
    validation_count: int
    test_count: int


@dataclass(frozen=True, slots=True)
class DatasetCertifiedReady(DomainEvent):
    dataset_id: DatasetId
    verdict: Verdict
    warning_count: int


@dataclass(frozen=True, slots=True)
class DatasetRejected(DomainEvent):
    dataset_id: DatasetId
    blocking_count: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetReopened(DomainEvent):
    dataset_id: DatasetId
    reason: str
