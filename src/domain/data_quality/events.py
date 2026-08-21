"""Data Quality Context 의 Domain Event."""

from __future__ import annotations

from dataclasses import dataclass

from domain.data_quality.dimensions import QualityDimension
from domain.data_quality.identifiers import AssessmentId
from domain.shared.events import DomainEvent
from domain.shared.inspection import Verdict


@dataclass(frozen=True, slots=True)
class QualityAssessmentStarted(DomainEvent):
    assessment_id: AssessmentId
    dataset_ref: str


@dataclass(frozen=True, slots=True)
class QualityDimensionMeasured(DomainEvent):
    assessment_id: AssessmentId
    dimension: QualityDimension
    score: float
    verdict: Verdict
    finding_count: int


@dataclass(frozen=True, slots=True)
class RemediationRecorded(DomainEvent):
    assessment_id: AssessmentId
    dimension: QualityDimension
    kind: str
    affected_rows: int
    decided_by: str


@dataclass(frozen=True, slots=True)
class QualityGatePassed(DomainEvent):
    assessment_id: AssessmentId
    dataset_ref: str
    overall_score: float
    warning_count: int


@dataclass(frozen=True, slots=True)
class QualityGateBlocked(DomainEvent):
    assessment_id: AssessmentId
    dataset_ref: str
    overall_score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualityAssessmentReopened(DomainEvent):
    assessment_id: AssessmentId
    reason: str
