"""MeasureLabelQuality — 라벨 오류를 잰다. (실습 2-4)

라벨 일관성 규칙은 데이터에서 나오지 않는다. 현장에서 받아 적어 넣는다.
규칙이 비어 있으면 Policy 가 그 사실 자체를 CRITICAL 로 잡는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data_quality.dto import DimensionView
from application.data_quality.support import commit, load_assessment
from application.shared.errors import UnsupportedOperation
from application.shared.ports import EventPublisher
from domain.data_quality.label_quality import (
    LabelConsistencyRule,
    LabelQualityPolicy,
)
from domain.data_quality.ports import LabelErrorMeasurer, QualityAssessmentRepository


@dataclass(frozen=True, slots=True)
class MeasureLabelQualityCommand:
    assessment_id: str
    rules: tuple[LabelConsistencyRule, ...] = field(default_factory=tuple)
    policy: LabelQualityPolicy = field(default_factory=LabelQualityPolicy)


class MeasureLabelQuality:
    def __init__(
        self,
        repository: QualityAssessmentRepository,
        measurer: LabelErrorMeasurer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._measurer = measurer
        self._publisher = publisher

    def execute(self, command: MeasureLabelQualityCommand) -> DimensionView:
        assessment = load_assessment(self._repository, command.assessment_id)
        if not assessment.target.has_label:
            raise UnsupportedOperation(
                "라벨 필드가 없다. 라벨 품질을 논할 수 없다.",
                subject=assessment.dataset_ref,
            )
        measurement = self._measurer.measure(assessment.target, command.rules)
        result = command.policy.evaluate(measurement, command.rules)
        assessment.record_dimension(result)
        commit(self._repository, assessment, self._publisher)
        return DimensionView.of(str(assessment.id), result)
