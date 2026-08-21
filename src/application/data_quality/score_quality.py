"""ScoreQuality — 감이 아니라 숫자로 말한다. (실습 2-8)

두 가지를 낸다.

    1. 종합 점수    여섯 축의 가중 평균
    2. 학습 영향    품질을 '표본 수'라는 언어로 환산한 값

2번이 없으면 점수는 그냥 점수다.
"COMPLETENESS 42점"보다 "학습 가능 표본이 8,640에서 6,100으로 줄었다"가
현장에서 훨씬 강하게 작동한다.

네 축의 측정값을 여기서 한 번 더 잰다.
각 축을 측정한 시점이 서로 다를 수 있으므로, 환산은 같은 시점 값으로 해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data_quality.dto import (
    DimensionView,
    QualityScoreView,
    TrainingImpactView,
)
from application.data_quality.support import commit, load_assessment
from application.shared.errors import UnsupportedOperation
from application.shared.ports import EventPublisher
from domain.data_quality.comparison import estimate_training_impact
from domain.data_quality.gate import QualityGatePolicy
from domain.data_quality.label_quality import LabelConsistencyRule
from domain.data_quality.ports import (
    ClassBalanceMeasurer,
    DuplicateMeasurer,
    LabelErrorMeasurer,
    MissingValueMeasurer,
    QualityAssessmentRepository,
)


@dataclass(frozen=True, slots=True)
class ScoreQualityCommand:
    assessment_id: str
    policy: QualityGatePolicy = field(default_factory=QualityGatePolicy)
    label_rules: tuple[LabelConsistencyRule, ...] = field(default_factory=tuple)


class ScoreQuality:
    def __init__(
        self,
        repository: QualityAssessmentRepository,
        missing_measurer: MissingValueMeasurer,
        duplicate_measurer: DuplicateMeasurer,
        balance_measurer: ClassBalanceMeasurer,
        label_measurer: LabelErrorMeasurer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._missing = missing_measurer
        self._duplicates = duplicate_measurer
        self._balance = balance_measurer
        self._labels = label_measurer
        self._publisher = publisher

    def execute(self, command: ScoreQualityCommand) -> QualityScoreView:
        assessment = load_assessment(self._repository, command.assessment_id)
        if not assessment.results:
            raise UnsupportedOperation(
                "측정한 축이 하나도 없다. 점수를 낼 근거가 없다.",
                subject=str(assessment.id),
            )

        target = assessment.target
        impact = estimate_training_impact(
            missing=self._missing.measure(target),
            duplicates=self._duplicates.measure(target),
            balance=self._balance.measure(target),
            labels=self._labels.measure(target, command.label_rules),
        )
        assessment.record_training_impact(impact)
        commit(self._repository, assessment, self._publisher)

        overall = command.policy.overall_score(assessment.results)
        return QualityScoreView(
            assessment_id=str(assessment.id),
            dataset_ref=assessment.dataset_ref,
            overall_score=overall.value,
            grade=overall.grade.value,
            dimensions=tuple(
                DimensionView.of(str(assessment.id), assessment.results[d])
                for d in assessment.measured_dimensions
            ),
            impact=TrainingImpactView.of(impact),
        )
