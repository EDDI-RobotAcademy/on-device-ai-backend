"""MeasureValidity — 이상치를 잰다. (실습 2-3)"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data_quality.dto import DimensionView
from application.data_quality.support import commit, load_assessment
from application.shared.ports import EventPublisher
from domain.data_quality.ports import OutlierMeasurer, QualityAssessmentRepository
from domain.data_quality.validity import ValidityPolicy


@dataclass(frozen=True, slots=True)
class MeasureValidityCommand:
    assessment_id: str
    policy: ValidityPolicy = field(default_factory=ValidityPolicy)


class MeasureValidity:
    def __init__(
        self,
        repository: QualityAssessmentRepository,
        measurer: OutlierMeasurer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._measurer = measurer
        self._publisher = publisher

    def execute(self, command: MeasureValidityCommand) -> DimensionView:
        assessment = load_assessment(self._repository, command.assessment_id)
        measurement = self._measurer.measure(assessment.target)
        result = command.policy.evaluate(measurement)
        assessment.record_dimension(result)
        commit(self._repository, assessment, self._publisher)
        return DimensionView.of(str(assessment.id), result)
