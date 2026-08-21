"""GetAssessment / ListAssessments — 조회."""

from __future__ import annotations

from dataclasses import dataclass

from application.data_quality.dto import AssessmentView
from application.data_quality.support import load_assessment
from domain.data_quality.ports import QualityAssessmentRepository


@dataclass(frozen=True, slots=True)
class GetAssessmentQuery:
    assessment_id: str


class GetAssessment:
    def __init__(self, repository: QualityAssessmentRepository) -> None:
        self._repository = repository

    def execute(self, query: GetAssessmentQuery) -> AssessmentView:
        return AssessmentView.of(load_assessment(self._repository, query.assessment_id))


class ListAssessments:
    def __init__(self, repository: QualityAssessmentRepository) -> None:
        self._repository = repository

    def execute(self, dataset_ref: str | None = None) -> tuple[AssessmentView, ...]:
        items = (
            self._repository.find_by_dataset(dataset_ref)
            if dataset_ref
            else self._repository.list_all()
        )
        return tuple(AssessmentView.of(a) for a in items)
