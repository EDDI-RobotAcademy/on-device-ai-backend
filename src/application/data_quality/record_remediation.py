"""RecordRemediation — 데이터를 고쳤다는 기록을 남긴다. (실습 2-9)

이 Use Case 는 데이터를 고치지 않는다. **고쳤다는 사실을 기록할 뿐이다.**
실제 수정은 데이터 파이프라인이 하고, 여기에는 그 근거만 남는다.

기록한 축은 '미검증' 상태가 되고, 다시 측정해야 게이트를 통과할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.data_quality.dto import AssessmentView
from application.data_quality.support import commit, load_assessment
from application.shared.ports import EventPublisher
from domain.data_quality.ports import QualityAssessmentRepository
from domain.data_quality.remediation import RemediationAction


@dataclass(frozen=True, slots=True)
class RecordRemediationCommand:
    assessment_id: str
    action: RemediationAction


class RecordRemediation:
    def __init__(
        self,
        repository: QualityAssessmentRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._publisher = publisher

    def execute(self, command: RecordRemediationCommand) -> AssessmentView:
        assessment = load_assessment(self._repository, command.assessment_id)
        assessment.record_remediation(command.action)
        commit(self._repository, assessment, self._publisher)
        return AssessmentView.of(assessment)
