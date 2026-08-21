"""MeasureBalance — 클래스 불균형을 잰다. (실습 2-5)"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data_quality.dto import DimensionView
from application.data_quality.support import commit, load_assessment
from application.shared.errors import UnsupportedOperation
from application.shared.ports import EventPublisher
from domain.data_quality.balance import BalancePolicy
from domain.data_quality.ports import ClassBalanceMeasurer, QualityAssessmentRepository


@dataclass(frozen=True, slots=True)
class MeasureBalanceCommand:
    assessment_id: str
    policy: BalancePolicy = field(default_factory=BalancePolicy)


class MeasureBalance:
    def __init__(
        self,
        repository: QualityAssessmentRepository,
        measurer: ClassBalanceMeasurer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._measurer = measurer
        self._publisher = publisher

    def execute(self, command: MeasureBalanceCommand) -> DimensionView:
        assessment = load_assessment(self._repository, command.assessment_id)
        if not assessment.target.has_label:
            raise UnsupportedOperation(
                "라벨 필드가 없다. 클래스 균형을 논할 수 없다.",
                subject=assessment.dataset_ref,
            )
        measurement = self._measurer.measure(assessment.target)
        result = command.policy.evaluate(measurement)
        assessment.record_dimension(result)
        commit(self._repository, assessment, self._publisher)
        return DimensionView.of(str(assessment.id), result)
