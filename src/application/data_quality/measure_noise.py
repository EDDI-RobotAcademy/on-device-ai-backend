"""MeasureNoise — 잡음을 잰다. (실습 2-6)"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data_quality.dto import DimensionView
from application.data_quality.support import commit, load_assessment
from application.shared.ports import EventPublisher
from domain.data_quality.noise import NoisePolicy
from domain.data_quality.ports import NoiseMeasurer, QualityAssessmentRepository


@dataclass(frozen=True, slots=True)
class MeasureNoiseCommand:
    assessment_id: str
    policy: NoisePolicy = field(default_factory=NoisePolicy)


class MeasureNoise:
    def __init__(
        self,
        repository: QualityAssessmentRepository,
        measurer: NoiseMeasurer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._measurer = measurer
        self._publisher = publisher

    def execute(self, command: MeasureNoiseCommand) -> DimensionView:
        assessment = load_assessment(self._repository, command.assessment_id)
        measurement = self._measurer.measure(assessment.target)
        result = command.policy.evaluate(measurement)
        assessment.record_dimension(result)
        commit(self._repository, assessment, self._publisher)
        return DimensionView.of(str(assessment.id), result)
