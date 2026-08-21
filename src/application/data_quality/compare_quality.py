"""CompareQuality — 망가진 데이터와 정상 데이터를 직접 비교한다. (실습 2-9)

두 개의 평가를 나란히 놓는다.
"좋아졌다"가 아니라 **어느 축이 몇 점 올랐고, 학습 가능 표본이 몇 개 바뀌었는가**를 낸다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data_quality.dto import QualityComparisonView
from application.data_quality.support import load_assessment
from application.shared.ports import EventPublisher
from domain.data_quality.comparison import QualityComparison
from domain.data_quality.gate import QualityGatePolicy
from domain.data_quality.ports import QualityAssessmentRepository


@dataclass(frozen=True, slots=True)
class CompareQualityCommand:
    before_assessment_id: str
    after_assessment_id: str
    before_label: str = "before"
    after_label: str = "after"
    policy: QualityGatePolicy = field(default_factory=QualityGatePolicy)


class CompareQuality:
    def __init__(
        self,
        repository: QualityAssessmentRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._publisher = publisher

    def execute(self, command: CompareQualityCommand) -> QualityComparisonView:
        before = load_assessment(self._repository, command.before_assessment_id)
        after = load_assessment(self._repository, command.after_assessment_id)

        comparison = QualityComparison(
            before=before.snapshot(command.policy, command.before_label),
            after=after.snapshot(command.policy, command.after_label),
        )
        return QualityComparisonView.of(comparison)
