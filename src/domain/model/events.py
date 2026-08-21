"""Model Context 의 Domain Event."""

from __future__ import annotations

from dataclasses import dataclass

from domain.model.identifiers import ModelVersionId, TrainingRunId
from domain.shared.events import DomainEvent
from domain.shared.inspection import Verdict


@dataclass(frozen=True, slots=True)
class TrainingRunPrepared(DomainEvent):
    run_id: TrainingRunId
    dataset_ref: str
    architecture: str


@dataclass(frozen=True, slots=True)
class TrainingRunStarted(DomainEvent):
    run_id: TrainingRunId
    config: str


@dataclass(frozen=True, slots=True)
class EpochCompleted(DomainEvent):
    run_id: TrainingRunId
    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True, slots=True)
class TrainingRunCompleted(DomainEvent):
    run_id: TrainingRunId
    model_version_id: ModelVersionId
    epochs: int
    best_epoch: int


@dataclass(frozen=True, slots=True)
class TrainingRunFailed(DomainEvent):
    run_id: TrainingRunId
    reason: str


@dataclass(frozen=True, slots=True)
class ModelEvaluated(DomainEvent):
    run_id: TrainingRunId
    split: str
    accuracy: float
    macro_recall: float


@dataclass(frozen=True, slots=True)
class ModelAccepted(DomainEvent):
    run_id: TrainingRunId
    model_version_id: ModelVersionId
    verdict: Verdict
    warning_count: int


@dataclass(frozen=True, slots=True)
class ModelRejected(DomainEvent):
    run_id: TrainingRunId
    model_version_id: ModelVersionId
    reasons: tuple[str, ...]
