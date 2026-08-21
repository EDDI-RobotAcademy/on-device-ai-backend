"""EvaluateQualityGate — 학습을 시작해도 되는지 판정한다. (실습 2-10)

모듈 1의 CertifyDatasetReadiness 와 같은 자리에 있지만 묻는 것이 다르다.

    모듈 1  이 데이터가 무엇인지 아는가
    모듈 2  이 데이터가 쓸 만한가

둘 다 통과해야 모델 학습으로 넘어간다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data_quality.dto import AssessmentView, QualityGateView
from application.data_quality.support import commit, load_assessment
from application.shared.ports import EventPublisher
from domain.data_quality.gate import QualityGatePolicy
from domain.data_quality.ports import QualityAssessmentRepository


@dataclass(frozen=True, slots=True)
class EvaluateQualityGateCommand:
    assessment_id: str
    policy: QualityGatePolicy = field(default_factory=QualityGatePolicy)


class EvaluateQualityGate:
    def __init__(
        self,
        repository: QualityAssessmentRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._publisher = publisher

    def execute(self, command: EvaluateQualityGateCommand) -> QualityGateView:
        assessment = load_assessment(self._repository, command.assessment_id)
        certificate = assessment.pass_through_gate(command.policy)
        commit(self._repository, assessment, self._publisher)
        return QualityGateView.of(str(assessment.id), certificate)


@dataclass(frozen=True, slots=True)
class ReopenAssessmentCommand:
    assessment_id: str
    reason: str


class ReopenAssessment:
    def __init__(
        self,
        repository: QualityAssessmentRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._publisher = publisher

    def execute(self, command: ReopenAssessmentCommand) -> AssessmentView:
        assessment = load_assessment(self._repository, command.assessment_id)
        assessment.reopen(command.reason)
        commit(self._repository, assessment, self._publisher)
        return AssessmentView.of(assessment)
