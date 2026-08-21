"""QualityAssessmentRepository 의 인메모리 구현."""

from __future__ import annotations

from collections.abc import Sequence

from domain.data_quality.assessment import QualityAssessment
from domain.data_quality.identifiers import AssessmentId


class InMemoryAssessmentRepository:
    """domain.data_quality.ports.QualityAssessmentRepository 구현."""

    def __init__(self) -> None:
        self._items: dict[str, QualityAssessment] = {}

    def save(self, assessment: QualityAssessment) -> None:
        self._items[str(assessment.id)] = assessment

    def find_by_id(self, assessment_id: AssessmentId) -> QualityAssessment | None:
        return self._items.get(str(assessment_id))

    def exists(self, assessment_id: AssessmentId) -> bool:
        return str(assessment_id) in self._items

    def find_by_dataset(self, dataset_ref: str) -> Sequence[QualityAssessment]:
        return tuple(
            a for a in self._items.values() if a.dataset_ref == dataset_ref
        )

    def list_all(self) -> Sequence[QualityAssessment]:
        return tuple(self._items.values())

    def clear(self) -> None:
        self._items.clear()
