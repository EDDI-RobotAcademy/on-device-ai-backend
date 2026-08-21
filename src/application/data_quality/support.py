"""Data Quality Use Case 들이 공유하는 최소한의 거들기."""

from __future__ import annotations

from application.shared.ports import EventPublisher
from domain.data_quality.assessment import QualityAssessment
from domain.data_quality.errors import AssessmentNotFound
from domain.data_quality.identifiers import AssessmentId
from domain.data_quality.ports import QualityAssessmentRepository


def load_assessment(
    repository: QualityAssessmentRepository, assessment_id: str | AssessmentId
) -> QualityAssessment:
    identifier = (
        assessment_id
        if isinstance(assessment_id, AssessmentId)
        else AssessmentId.of(assessment_id)
    )
    assessment = repository.find_by_id(identifier)
    if assessment is None:
        raise AssessmentNotFound(
            f"품질 평가 '{identifier}' 가 존재하지 않는다.", subject=str(identifier)
        )
    return assessment


def commit(
    repository: QualityAssessmentRepository,
    assessment: QualityAssessment,
    publisher: EventPublisher | None = None,
) -> None:
    repository.save(assessment)
    events = assessment.pull_events()
    if publisher is not None and events:
        publisher.publish(events)
