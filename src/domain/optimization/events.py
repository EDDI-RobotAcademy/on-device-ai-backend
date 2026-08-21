"""Optimization Context 의 Domain Event."""

from __future__ import annotations

from dataclasses import dataclass

from domain.optimization.identifiers import ArtifactId, OptimizationRunId
from domain.shared.events import DomainEvent
from domain.shared.inspection import Verdict


@dataclass(frozen=True, slots=True)
class OptimizationRunStarted(DomainEvent):
    run_id: OptimizationRunId
    model_version_id: str


@dataclass(frozen=True, slots=True)
class BaselineBenchmarked(DomainEvent):
    run_id: OptimizationRunId
    p95_ms: float
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CandidateAdded(DomainEvent):
    run_id: OptimizationRunId
    artifact_id: ArtifactId
    label: str
    p95_ms: float
    size_bytes: int
    accuracy: float


@dataclass(frozen=True, slots=True)
class ConversionRejected(DomainEvent):
    run_id: OptimizationRunId
    label: str
    reason: str


@dataclass(frozen=True, slots=True)
class ModelSelected(DomainEvent):
    run_id: OptimizationRunId
    artifact_id: ArtifactId
    label: str
    verdict: Verdict


@dataclass(frozen=True, slots=True)
class SelectionBlocked(DomainEvent):
    run_id: OptimizationRunId
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OptimizationRunReopened(DomainEvent):
    run_id: OptimizationRunId
    reason: str
