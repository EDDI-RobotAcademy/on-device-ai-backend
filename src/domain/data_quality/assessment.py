"""QualityAssessment — Data Quality Context 의 Aggregate Root.

이 객체가 지키는 것은 **"고쳤다는 주장은 측정으로만 확인된다"**는 규칙이다.

    측정 → (문제 발견) → 조치 기록 → **재측정** → 게이트

조치를 기록하고 재측정하지 않으면 게이트를 통과할 수 없다.
현장에서 가장 흔한 사고가 "고쳤다고 하고 확인하지 않는 것"이기 때문이다.

절대 하지 않는 것:
    - 파일을 읽지 않는다
    - 점수를 스스로 계산하지 않는다 (Policy 가 한다)
    - 지금 몇 시인지 묻지 않는다
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.data_quality import events as domain_events
from domain.data_quality.comparison import QualitySnapshot, TrainingImpact
from domain.data_quality.dimensions import DimensionResult, QualityDimension
from domain.data_quality.gate import QualityCertificate, QualityGatePolicy
from domain.data_quality.identifiers import AssessmentId
from domain.data_quality.remediation import RemediationAction
from domain.data_quality.target import AssessmentTarget
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from domain.shared.events import EventRecorder
from domain.shared.inspection import Verdict


class AssessmentStatus(Enum):
    OPEN = "OPEN"
    """시작했다. 아직 아무 축도 측정하지 않았다."""

    MEASURING = "MEASURING"
    """측정 중이다."""

    REMEDIATING = "REMEDIATING"
    """조치를 기록했고, 재측정을 기다린다."""

    PASSED = "PASSED"
    BLOCKED = "BLOCKED"


class QualityAssessment(EventRecorder):
    """한 Dataset 에 대한 한 번의 품질 평가."""

    __slots__ = (
        "_id",
        "_target",
        "_status",
        "_results",
        "_remediations",
        "_unverified",
        "_impact",
        "_certificate",
    )

    def __init__(self, assessment_id: AssessmentId, target: AssessmentTarget) -> None:
        super().__init__()
        self._id = assessment_id
        self._target = target
        self._status = AssessmentStatus.OPEN
        self._results: dict[QualityDimension, DimensionResult] = {}
        self._remediations: list[RemediationAction] = []
        self._unverified: set[QualityDimension] = set()
        self._impact: TrainingImpact | None = None
        self._certificate: QualityCertificate | None = None

    # -- 생성 --------------------------------------------------------------
    @classmethod
    def start(
        cls, assessment_id: AssessmentId, target: AssessmentTarget
    ) -> QualityAssessment:
        assessment = cls(assessment_id, target)
        assessment._record(
            domain_events.QualityAssessmentStarted(
                assessment_id=assessment_id, dataset_ref=target.dataset_ref
            )
        )
        return assessment

    # -- 조회 --------------------------------------------------------------
    @property
    def id(self) -> AssessmentId:
        return self._id

    @property
    def target(self) -> AssessmentTarget:
        return self._target

    @property
    def dataset_ref(self) -> str:
        return self._target.dataset_ref

    @property
    def status(self) -> AssessmentStatus:
        return self._status

    @property
    def results(self) -> dict[QualityDimension, DimensionResult]:
        return dict(self._results)

    def result_of(self, dimension: QualityDimension) -> DimensionResult | None:
        return self._results.get(dimension)

    @property
    def remediations(self) -> tuple[RemediationAction, ...]:
        return tuple(self._remediations)

    @property
    def unverified_dimensions(self) -> frozenset[QualityDimension]:
        """조치는 했는데 재측정하지 않은 축."""
        return frozenset(self._unverified)

    @property
    def impact(self) -> TrainingImpact | None:
        return self._impact

    @property
    def certificate(self) -> QualityCertificate | None:
        return self._certificate

    @property
    def is_passed(self) -> bool:
        return self._status is AssessmentStatus.PASSED

    @property
    def measured_dimensions(self) -> tuple[QualityDimension, ...]:
        return tuple(sorted(self._results, key=lambda d: d.value))

    # -- 행위 --------------------------------------------------------------
    def record_dimension(self, result: DimensionResult) -> None:
        """한 축의 평가 결과를 붙인다. 같은 축을 다시 측정하면 덮어쓴다."""
        self._guard_mutable(f"{result.dimension.value} 측정 기록")
        self._results[result.dimension] = result
        self._unverified.discard(result.dimension)
        if self._status is AssessmentStatus.OPEN:
            self._status = AssessmentStatus.MEASURING
        elif self._status is AssessmentStatus.REMEDIATING and not self._unverified:
            self._status = AssessmentStatus.MEASURING
        self._record(
            domain_events.QualityDimensionMeasured(
                assessment_id=self._id,
                dimension=result.dimension,
                score=result.score.value,
                verdict=result.verdict,
                finding_count=len(result.findings),
            )
        )

    def record_training_impact(self, impact: TrainingImpact) -> None:
        """품질을 표본 수 언어로 환산한 값을 붙인다. (실습 2-8, 2-9)"""
        self._guard_mutable("학습 영향 기록")
        if not self._results:
            raise IllegalStateTransition(
                "측정 없이 학습 영향을 계산할 수 없다.", subject=str(self._id)
            )
        self._impact = impact

    def record_remediation(self, action: RemediationAction) -> None:
        """데이터를 고쳤다는 기록을 남긴다.

        조치한 축은 '미검증' 상태가 되고, 다시 측정해야 게이트를 통과할 수 있다.
        """
        self._guard_mutable("조치 기록")
        if action.dimension not in self._results:
            raise IllegalStateTransition(
                f"{action.dimension.value} 를 측정하지 않은 채로 조치를 기록할 수 없다. "
                "무엇을 고쳤는지 확인할 방법이 없다.",
                subject=str(self._id),
            )
        self._remediations.append(action)
        self._unverified.add(action.dimension)
        self._status = AssessmentStatus.REMEDIATING
        self._record(
            domain_events.RemediationRecorded(
                assessment_id=self._id,
                dimension=action.dimension,
                kind=action.kind.value,
                affected_rows=action.affected_rows,
                decided_by=action.decided_by,
            )
        )

    def pass_through_gate(self, policy: QualityGatePolicy) -> QualityCertificate:
        """게이트를 통과시킬지 판정한다. (실습 2-10)"""
        if not self._results:
            raise IllegalStateTransition(
                "아무것도 측정하지 않은 평가는 판정할 수 없다.", subject=str(self._id)
            )
        certificate = policy.evaluate(
            self.dataset_ref,
            self._results,
            unverified_dimensions=self.unverified_dimensions,
        )
        self._certificate = certificate
        if certificate.is_ready:
            self._status = AssessmentStatus.PASSED
            self._record(
                domain_events.QualityGatePassed(
                    assessment_id=self._id,
                    dataset_ref=self.dataset_ref,
                    overall_score=certificate.overall_score.value,
                    warning_count=len(certificate.warning_findings),
                )
            )
        else:
            self._status = AssessmentStatus.BLOCKED
            self._record(
                domain_events.QualityGateBlocked(
                    assessment_id=self._id,
                    dataset_ref=self.dataset_ref,
                    overall_score=certificate.overall_score.value,
                    reasons=certificate.blocking_reasons,
                )
            )
        return certificate

    def reopen(self, reason: str) -> None:
        if self._status not in (AssessmentStatus.PASSED, AssessmentStatus.BLOCKED):
            raise IllegalStateTransition(
                "판정되지 않은 평가는 reopen 대상이 아니다.", subject=str(self._id)
            )
        if not reason.strip():
            raise InvariantViolation(
                "판정을 되돌리려면 이유를 남겨야 한다.", subject="reason"
            )
        self._status = (
            AssessmentStatus.REMEDIATING if self._unverified else AssessmentStatus.MEASURING
        )
        self._certificate = None
        self._record(
            domain_events.QualityAssessmentReopened(
                assessment_id=self._id, reason=reason.strip()
            )
        )

    # -- 파생 --------------------------------------------------------------
    def snapshot(self, policy: QualityGatePolicy, label: str) -> QualitySnapshot:
        """비교(실습 2-9)를 위한 한 시점의 상태."""
        return QualitySnapshot.of(
            label=label,
            overall=policy.overall_score(self._results),
            results=self._results,
            impact=self._impact,
        )

    @property
    def latest_verdict(self) -> Verdict | None:
        return self._certificate.verdict if self._certificate else None

    # -- 내부 --------------------------------------------------------------
    def _guard_mutable(self, action: str) -> None:
        if self._status in (AssessmentStatus.PASSED, AssessmentStatus.BLOCKED):
            raise IllegalStateTransition(
                f"{self._status.value} 상태에서는 '{action}' 을 할 수 없다. "
                "reopen(reason) 으로 판정을 되돌린 뒤 수정한다.",
                subject=str(self._id),
            )

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return (
            f"QualityAssessment(id={self._id}, dataset={self.dataset_ref!r}, "
            f"status={self._status.value})"
        )


@dataclass(frozen=True, slots=True)
class AssessmentSummary:
    """조회용 요약 (Aggregate 를 그대로 내보내지 않기 위한 것)."""

    assessment_id: str
    dataset_ref: str
    status: str
    measured: tuple[str, ...]
    overall_score: float | None
