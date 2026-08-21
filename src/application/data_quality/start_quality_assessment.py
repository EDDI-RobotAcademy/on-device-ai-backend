"""StartQualityAssessment — 품질 평가를 시작한다. (실습 2-1)

여기서 확인하는 것: **모듈 1을 통과했다고 깨끗한 것이 아니다.**

Dataset 이 READY 라는 것은 "구조와 계약이 섰다"는 뜻이지
"내용이 깨끗하다"는 뜻이 아니다. 그래서 품질 평가는 별도로 시작한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.data.support import load_dataset
from application.data_quality.dto import AssessmentView
from application.data_quality.support import commit
from application.data_quality.target_mapper import assessment_target_from
from application.shared.errors import ConflictingRequest
from application.shared.ports import EventPublisher
from domain.data.ports import DatasetRepository
from domain.data_quality.assessment import QualityAssessment
from domain.data_quality.identifiers import AssessmentId
from domain.data_quality.ports import QualityAssessmentRepository


@dataclass(frozen=True, slots=True)
class StartQualityAssessmentCommand:
    assessment_id: str
    dataset_id: str


class StartQualityAssessment:
    def __init__(
        self,
        assessments: QualityAssessmentRepository,
        datasets: DatasetRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._assessments = assessments
        self._datasets = datasets
        self._publisher = publisher

    def execute(self, command: StartQualityAssessmentCommand) -> AssessmentView:
        assessment_id = AssessmentId.of(command.assessment_id)
        if self._assessments.exists(assessment_id):
            raise ConflictingRequest(
                f"품질 평가 '{assessment_id}' 는 이미 존재한다.",
                subject=str(assessment_id),
            )

        dataset = load_dataset(self._datasets, command.dataset_id)
        target = assessment_target_from(dataset)

        assessment = QualityAssessment.start(assessment_id, target)
        commit(self._assessments, assessment, self._publisher)
        return AssessmentView.of(assessment)
