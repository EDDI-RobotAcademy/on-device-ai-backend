"""QualityAssessment Aggregate — 측정 → 조치 → 재측정 → 게이트."""

from __future__ import annotations

import pytest

from domain.data_quality.assessment import AssessmentStatus, QualityAssessment
from domain.data_quality.dimensions import (
    ALL_DIMENSIONS,
    DimensionResult,
    QualityDimension,
    QualityScore,
)
from domain.data_quality.gate import QualityGatePolicy
from domain.data_quality.identifiers import AssessmentId
from domain.data_quality.remediation import RemediationAction, RemediationKind
from domain.data_quality.target import AssessmentTarget
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from domain.shared.inspection import Finding, Severity, Verdict

TARGET = AssessmentTarget(
    dataset_ref="plant-power-2026-04",
    uri="plant_power.csv",
    feature_fields=("active_power_kw", "voltage_v"),
    label_field="condition",
    time_field="timestamp",
    group_field="batch_id",
)


def new_assessment() -> QualityAssessment:
    return QualityAssessment.start(AssessmentId.of("qa-1"), TARGET)


def perfect(dimension: QualityDimension) -> DimensionResult:
    return DimensionResult(dimension=dimension, score=QualityScore.perfect())


def broken(dimension: QualityDimension, score: float = 30.0) -> DimensionResult:
    return DimensionResult(
        dimension=dimension,
        score=QualityScore(score),
        findings=(Finding("X", "치명적 오염", Severity.CRITICAL),),
    )


def remediation(dimension: QualityDimension) -> RemediationAction:
    return RemediationAction(
        kind=RemediationKind.DROP_ROWS,
        dimension=dimension,
        target="temperature_c",
        affected_rows=120,
        rationale="센서 단절 구간이라 보간할 수 없다. 해당 구간을 제외한다.",
        decided_by="데이터팀 · 설비운영팀 합의",
    )


class TestLifecycle:
    def test_시작하면_이벤트가_남는다(self) -> None:
        assessment = new_assessment()
        assert assessment.status is AssessmentStatus.OPEN
        assert [e.event_name for e in assessment.pending_events] == [
            "QualityAssessmentStarted"
        ]

    def test_측정하면_상태가_바뀐다(self) -> None:
        assessment = new_assessment()
        assessment.record_dimension(perfect(QualityDimension.COMPLETENESS))
        assert assessment.status is AssessmentStatus.MEASURING
        assert assessment.measured_dimensions == (QualityDimension.COMPLETENESS,)

    def test_같은_축을_다시_측정하면_덮어쓴다(self) -> None:
        assessment = new_assessment()
        assessment.record_dimension(broken(QualityDimension.NOISE))
        assert assessment.result_of(QualityDimension.NOISE).verdict is Verdict.FAILED

        assessment.record_dimension(perfect(QualityDimension.NOISE))
        assert assessment.result_of(QualityDimension.NOISE).verdict is Verdict.PASSED

    def test_아무것도_측정하지_않고_판정할_수_없다(self) -> None:
        with pytest.raises(IllegalStateTransition, match="아무것도 측정하지 않은"):
            new_assessment().pass_through_gate(QualityGatePolicy())

    def test_측정_없이_학습_영향을_계산할_수_없다(self) -> None:
        from domain.data_quality.comparison import TrainingImpact

        impact = TrainingImpact(
            total_rows=100,
            distinct_rows=100,
            rows_with_missing=0,
            conflicting_rows=0,
            baseline_accuracy=0.9,
            minority_count=10,
            accuracy_ceiling=1.0,
        )
        with pytest.raises(IllegalStateTransition, match="측정 없이"):
            new_assessment().record_training_impact(impact)


class TestRemediation:
    def test_측정하지_않은_축은_고칠_수_없다(self) -> None:
        assessment = new_assessment()
        assessment.record_dimension(perfect(QualityDimension.NOISE))

        with pytest.raises(IllegalStateTransition, match="측정하지 않은 채로"):
            assessment.record_remediation(remediation(QualityDimension.COMPLETENESS))

    def test_조치하면_그_축은_미검증이_된다(self) -> None:
        assessment = new_assessment()
        assessment.record_dimension(broken(QualityDimension.COMPLETENESS))
        assessment.record_remediation(remediation(QualityDimension.COMPLETENESS))

        assert assessment.status is AssessmentStatus.REMEDIATING
        assert assessment.unverified_dimensions == frozenset(
            {QualityDimension.COMPLETENESS}
        )

    def test_재측정하면_미검증이_풀린다(self) -> None:
        assessment = new_assessment()
        assessment.record_dimension(broken(QualityDimension.COMPLETENESS))
        assessment.record_remediation(remediation(QualityDimension.COMPLETENESS))
        assessment.record_dimension(perfect(QualityDimension.COMPLETENESS))

        assert assessment.unverified_dimensions == frozenset()
        assert assessment.status is AssessmentStatus.MEASURING

    def test_고쳤다는_주장만으로는_게이트를_통과할_수_없다(self) -> None:
        assessment = new_assessment()
        for dimension in ALL_DIMENSIONS:
            assessment.record_dimension(perfect(dimension))
        assessment.record_remediation(remediation(QualityDimension.COMPLETENESS))

        certificate = assessment.pass_through_gate(QualityGatePolicy())
        assert certificate.is_ready is False
        assert any(
            f.code == "GATE_REMEDIATION_UNVERIFIED"
            for f in certificate.blocking_findings
        )


class TestGate:
    def _all_perfect(self, assessment: QualityAssessment) -> None:
        for dimension in ALL_DIMENSIONS:
            assessment.record_dimension(perfect(dimension))

    def test_전부_통과하면_PASSED(self) -> None:
        assessment = new_assessment()
        self._all_perfect(assessment)
        certificate = assessment.pass_through_gate(QualityGatePolicy())

        assert certificate.verdict is Verdict.PASSED
        assert certificate.overall_score.value == pytest.approx(100.0)
        assert assessment.status is AssessmentStatus.PASSED

    def test_측정하지_않은_축이_있으면_막힌다(self) -> None:
        assessment = new_assessment()
        assessment.record_dimension(perfect(QualityDimension.NOISE))
        certificate = assessment.pass_through_gate(QualityGatePolicy())

        assert certificate.is_ready is False
        assert len(certificate.missing_dimensions) == 5
        # 측정하지 않은 축을 0점으로 치지 않는다.
        assert certificate.overall_score.value == pytest.approx(100.0)

    def test_차단_축은_점수와_무관하게_막는다(self) -> None:
        assessment = new_assessment()
        self._all_perfect(assessment)
        assessment.record_dimension(
            DimensionResult(
                dimension=QualityDimension.LABEL_QUALITY,
                score=QualityScore(99.0),  # 점수는 높다
                findings=(Finding("X", "라벨 모순", Severity.CRITICAL),),
            )
        )
        certificate = assessment.pass_through_gate(QualityGatePolicy())

        assert certificate.is_ready is False
        assert any("LABEL_QUALITY" in r for r in certificate.blocking_reasons)

    def test_차단_축이_아니면_점수로_판단한다(self) -> None:
        assessment = new_assessment()
        self._all_perfect(assessment)
        assessment.record_dimension(
            DimensionResult(
                dimension=QualityDimension.NOISE,
                score=QualityScore(80.0),
                findings=(Finding("X", "잡음", Severity.CRITICAL),),
            )
        )
        certificate = assessment.pass_through_gate(QualityGatePolicy())

        # NOISE 는 차단 축이 아니고 점수도 기준(50) 위이므로 통과한다.
        assert certificate.is_ready is True
        assert certificate.blocking_findings  # 그래도 근거는 남는다

    def test_판정_후에는_수정할_수_없다(self) -> None:
        assessment = new_assessment()
        self._all_perfect(assessment)
        assessment.pass_through_gate(QualityGatePolicy())

        with pytest.raises(IllegalStateTransition, match="reopen"):
            assessment.record_dimension(perfect(QualityDimension.NOISE))

    def test_reopen_은_이유를_요구한다(self) -> None:
        assessment = new_assessment()
        self._all_perfect(assessment)
        assessment.pass_through_gate(QualityGatePolicy())

        with pytest.raises(InvariantViolation, match="이유를 남겨야"):
            assessment.reopen("   ")

        assessment.reopen("전압 계측 배선 교체 후 재수집")
        assert assessment.status is AssessmentStatus.MEASURING
        assert assessment.certificate is None


class TestGatePolicy:
    def test_가중치_합은_1이어야_한다(self) -> None:
        with pytest.raises(InvariantViolation, match="합이"):
            QualityGatePolicy(weights={QualityDimension.NOISE: 0.5})

    def test_필수_축에_가중치가_없으면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="가중치가 없는"):
            QualityGatePolicy(
                weights={QualityDimension.NOISE: 1.0},
                required_dimensions=frozenset(ALL_DIMENSIONS),
            )
