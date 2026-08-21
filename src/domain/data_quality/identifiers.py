"""Data Quality Context 의 식별자."""

from __future__ import annotations

from dataclasses import dataclass

from domain.shared.identifier import Identifier


@dataclass(frozen=True, slots=True)
class AssessmentId(Identifier):
    """QualityAssessment Aggregate 의 식별자."""
