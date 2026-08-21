"""MeasureUniqueness — 중복을 잰다. (실습 2-7)"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data_quality.dto import DimensionView
from application.data_quality.support import commit, load_assessment
from application.shared.ports import EventPublisher
from domain.data_quality.ports import DuplicateMeasurer, QualityAssessmentRepository
from domain.data_quality.uniqueness import UniquenessPolicy


@dataclass(frozen=True, slots=True)
class MeasureUniquenessCommand:
    assessment_id: str
    policy: UniquenessPolicy = field(default_factory=UniquenessPolicy)


class MeasureUniqueness:
    def __init__(
        self,
        repository: QualityAssessmentRepository,
        measurer: DuplicateMeasurer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._measurer = measurer
        self._publisher = publisher

    def execute(self, command: MeasureUniquenessCommand) -> DimensionView:
        assessment = load_assessment(self._repository, command.assessment_id)
        measurement = self._measurer.measure(assessment.target)
        result = command.policy.evaluate(measurement)
        assessment.record_dimension(result)
        commit(self._repository, assessment, self._publisher)
        return DimensionView.of(str(assessment.id), result)
