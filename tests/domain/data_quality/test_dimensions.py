"""품질 축의 Value Object 와 점수 규칙.

파일도 pandas 도 없다. 측정값을 손으로 만들어 Policy 만 시험한다.
"""

from __future__ import annotations

import pytest

from domain.data_quality.balance import BalancePolicy, ClassBalanceMeasurement
from domain.data_quality.comparison import (
    TrainingImpact,
    estimate_training_impact,
)
from domain.data_quality.completeness import (
    CompletenessPolicy,
    FieldMissingness,
    MissingValueMeasurement,
)
from domain.data_quality.dimensions import (
    DimensionResult,
    QualityDimension,
    QualityGrade,
    QualityScore,
    deduct,
)
from domain.data_quality.label_quality import (
    LabelConsistencyRule,
    LabelErrorMeasurement,
    LabelQualityPolicy,
)
from domain.data_quality.noise import FieldNoise, NoiseMeasurement, NoisePolicy
from domain.data_quality.remediation import RemediationAction, RemediationKind
from domain.data_quality.target import AssessmentTarget
from domain.data_quality.uniqueness import DuplicateMeasurement, UniquenessPolicy
from domain.data_quality.validity import (
    FieldOutliers,
    OutlierMeasurement,
    ValidityPolicy,
)
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Severity, Verdict


def codes(result: DimensionResult) -> set[str]:
    return {f.code for f in result.findings}


class TestQualityScore:
    def test_범위를_벗어난_점수는_만들_수_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            QualityScore(-1.0)
        with pytest.raises(InvariantViolation):
            QualityScore(100.1)

    def test_등급은_점수에서_유도된다(self) -> None:
        assert QualityScore(95.0).grade is QualityGrade.A
        assert QualityScore(85.0).grade is QualityGrade.B
        assert QualityScore(75.0).grade is QualityGrade.C
        assert QualityScore(65.0).grade is QualityGrade.D
        assert QualityScore(10.0).grade is QualityGrade.F

    def test_감점은_0_아래로_내려가지_않는다(self) -> None:
        assert QualityScore.from_deductions([80.0, 50.0]).value == 0.0

    def test_감점_공식은_선형이다(self) -> None:
        assert deduct(0.0, tolerance=0.1, cap=0.5, weight=40.0) == 0.0
        assert deduct(0.3, tolerance=0.1, cap=0.5, weight=40.0) == pytest.approx(20.0)
        assert deduct(0.9, tolerance=0.1, cap=0.5, weight=40.0) == pytest.approx(40.0)

    def test_음수_가중치는_거부한다(self) -> None:
        with pytest.raises(InvariantViolation):
            deduct(1.0, tolerance=0.0, cap=1.0, weight=-1.0)


class TestAssessmentTarget:
    def test_입력_필드가_없으면_품질을_논할_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="입력 필드가 없다"):
            AssessmentTarget(dataset_ref="ds", uri="a.csv", feature_fields=())

    def test_어느_Dataset_인지_없으면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="되돌릴 수 없다"):
            AssessmentTarget(dataset_ref="  ", uri="a.csv", feature_fields=("v",))

    def test_물리_범위가_뒤집혀_있으면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="뒤집혀"):
            AssessmentTarget(
                dataset_ref="ds",
                uri="a.csv",
                feature_fields=("v",),
                physical_ranges={"v": (10.0, 1.0)},
            )


class TestCompleteness:
    def _measurement(self, **overrides: object) -> MissingValueMeasurement:
        base: dict[str, object] = dict(field_name="temperature_c", total_count=10_000)
        base.update(overrides)
        return MissingValueMeasurement(fields=(FieldMissingness(**base),))  # type: ignore[arg-type]

    def test_깨끗하면_만점이다(self) -> None:
        result = CompletenessPolicy().evaluate(self._measurement())
        assert result.score.value == 100.0
        assert result.verdict is Verdict.PASSED

    def test_결측이_많으면_점수가_내려간다(self) -> None:
        result = CompletenessPolicy().evaluate(self._measurement(missing_count=1_500))
        assert result.score.value < 90
        assert "MISSING_RATIO_HIGH" in codes(result)

    def test_연속_결측은_보간_대상이_아니다(self) -> None:
        result = CompletenessPolicy().evaluate(
            self._measurement(missing_count=500, longest_missing_run=500)
        )
        assert "MISSING_RUN_LONG" in codes(result)
        assert result.verdict is Verdict.FAILED

    def test_뭉쳐_있는_반복값은_은폐_결측이_아니다(self) -> None:
        result = CompletenessPolicy().evaluate(
            self._measurement(
                repeated_value=0.0,
                repeated_value_count=1_000,
                repeated_value_mean_run=25.0,
            )
        )
        assert "MISSING_HIDDEN" not in codes(result)

    def test_흩어진_반복값은_은폐_결측이다(self) -> None:
        result = CompletenessPolicy().evaluate(
            self._measurement(
                repeated_value=0.0,
                repeated_value_count=1_000,
                repeated_value_mean_run=1.02,
            )
        )
        assert "MISSING_HIDDEN" in codes(result)
        assert result.verdict is Verdict.FAILED

    def test_점수는_가장_나쁜_필드로_낸다(self) -> None:
        measurement = MissingValueMeasurement(
            fields=(
                FieldMissingness(field_name="a", total_count=1000),
                FieldMissingness(field_name="b", total_count=1000),
                FieldMissingness(
                    field_name="c", total_count=1000, missing_count=300
                ),
            )
        )
        result = CompletenessPolicy().evaluate(measurement)
        # 3개 중 2개가 완벽해도 평균으로 희석되지 않는다.
        assert result.score.value < 65


class TestValidity:
    def test_물리_범위_이탈은_치명적이다(self) -> None:
        result = ValidityPolicy().evaluate(
            OutlierMeasurement(
                fields=(
                    FieldOutliers(
                        field_name="current_a",
                        total_count=1000,
                        out_of_physical_range_count=1,
                    ),
                )
            )
        )
        assert "VALIDITY_OUT_OF_RANGE" in codes(result)
        assert result.verdict is Verdict.FAILED

    def test_masking_은_z_score_와_MAD_의_차이로_잡는다(self) -> None:
        result = ValidityPolicy().evaluate(
            OutlierMeasurement(
                fields=(
                    FieldOutliers(
                        field_name="v",
                        total_count=10_000,
                        z_outlier_count=5,
                        mad_outlier_count=300,
                    ),
                )
            )
        )
        assert "VALIDITY_ZSCORE_MASKED" in codes(result)

    def test_이상치_비율만으로는_막지_않는다(self) -> None:
        result = ValidityPolicy().evaluate(
            OutlierMeasurement(
                fields=(
                    FieldOutliers(
                        field_name="v", total_count=1000, mad_outlier_count=50
                    ),
                )
            )
        )
        assert result.verdict is Verdict.PASSED_WITH_WARNINGS


class TestLabelQuality:
    def _rule(self) -> LabelConsistencyRule:
        return LabelConsistencyRule(
            label="FAULT",
            field_name="active_power_kw",
            expected_max=30.0,
            description="보호 계전기 동작 시 부하 차단",
        )

    def test_규칙이_없으면_그_사실이_치명적이다(self) -> None:
        result = LabelQualityPolicy().evaluate(
            LabelErrorMeasurement(total_labeled=1000), rules=()
        )
        assert "LABEL_NO_CONSISTENCY_RULE" in codes(result)
        assert result.verdict is Verdict.FAILED

    def test_라벨_모순은_단_한_건도_허용하지_않는다(self) -> None:
        result = LabelQualityPolicy().evaluate(
            LabelErrorMeasurement(
                total_labeled=10_000, conflicting_duplicate_count=1
            ),
            rules=(self._rule(),),
        )
        assert "LABEL_CONFLICT" in codes(result)

    def test_정확도_상한을_계산한다(self) -> None:
        measurement = LabelErrorMeasurement(
            total_labeled=1000,
            rule_violations={"r": 20},
            conflicting_duplicate_count=10,
        )
        assert measurement.accuracy_ceiling() == pytest.approx(0.97)

    def test_조건_없는_규칙은_만들_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="조건이 없다"):
            LabelConsistencyRule(label="X", field_name="v", description="설명")


class TestBalance:
    def test_baseline_정확도가_높으면_지표를_바꿔야_한다(self) -> None:
        measurement = ClassBalanceMeasurement(
            class_counts={"NORMAL": 9_970, "FAULT": 30}
        )
        assert measurement.baseline_accuracy == pytest.approx(0.997)
        assert "BALANCE_BASELINE_TOO_HIGH" in codes(
            BalancePolicy().evaluate(measurement)
        )

    def test_소수_클래스가_없으면_분류가_아니다(self) -> None:
        result = BalancePolicy().evaluate(
            ClassBalanceMeasurement(class_counts={"NORMAL": 100})
        )
        assert result.score.value == 0.0
        assert result.findings[0].severity is Severity.CRITICAL

    def test_평가_집합_기대치를_계산한다(self) -> None:
        measurement = ClassBalanceMeasurement(
            class_counts={"NORMAL": 9_900, "FAULT": 100}, test_split_ratio=0.2
        )
        assert measurement.expected_minority_in_test == pytest.approx(20.0)


class TestNoise:
    def test_SNR_은_신호와_잡음의_비율이다(self) -> None:
        assert FieldNoise("v", signal_power=100.0, noise_power=1.0).snr_db == (
            pytest.approx(20.0)
        )

    def test_과하게_매끈하면_막는다(self) -> None:
        result = NoisePolicy().evaluate(
            NoiseMeasurement(
                fields=(
                    FieldNoise(
                        field_name="v",
                        signal_power=100.0,
                        noise_power=0.0001,
                        high_frequency_ratio=0.0,
                    ),
                )
            )
        )
        assert "NOISE_OVERSMOOTHED" in codes(result)

    def test_백색잡음_수준의_반전율은_통과한다(self) -> None:
        result = NoisePolicy().evaluate(
            NoiseMeasurement(
                fields=(
                    FieldNoise(
                        field_name="v",
                        signal_power=100.0,
                        noise_power=0.5,
                        high_frequency_ratio=0.01,
                        reversal_ratio=0.67,
                    ),
                )
            )
        )
        assert "NOISE_JITTER" not in codes(result)


class TestUniqueness:
    def test_부풀림_비율을_계산한다(self) -> None:
        measurement = DuplicateMeasurement(
            total_rows=10_000, exact_duplicate_count=2_000
        )
        assert measurement.distinct_row_count == 8_000
        assert measurement.inflation_ratio == pytest.approx(1.25)

    def test_라벨_모순은_치명적이다(self) -> None:
        result = UniquenessPolicy().evaluate(
            DuplicateMeasurement(total_rows=1000, conflicting_label_count=2)
        )
        assert "UNIQUENESS_LABEL_CONFLICT" in codes(result)
        assert result.verdict is Verdict.FAILED


class TestRemediation:
    def test_근거_없는_조치는_기록할_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="근거가 없다"):
            RemediationAction(
                kind=RemediationKind.DROP_ROWS,
                dimension=QualityDimension.COMPLETENESS,
                target="temperature_c",
                affected_rows=100,
                rationale="ok",
                decided_by="팀",
            )

    def test_영향_행_수는_음수일_수_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            RemediationAction(
                kind=RemediationKind.DROP_ROWS,
                dimension=QualityDimension.COMPLETENESS,
                target="temperature_c",
                affected_rows=-1,
                rationale="센서 단절 구간 제거",
                decided_by="데이터팀",
            )


class TestTrainingImpact:
    def test_네_축의_측정값을_표본_수로_환산한다(self) -> None:
        impact = estimate_training_impact(
            missing=MissingValueMeasurement(
                fields=(
                    FieldMissingness(
                        field_name="v", total_count=1000, missing_count=50
                    ),
                )
            ),
            duplicates=DuplicateMeasurement(
                total_rows=1000,
                exact_duplicate_count=100,
                conflicting_label_count=20,
            ),
            balance=ClassBalanceMeasurement(
                class_counts={"NORMAL": 950, "FAULT": 50}
            ),
            labels=LabelErrorMeasurement(
                total_labeled=1000, rule_violations={"r": 10}
            ),
        )
        assert isinstance(impact, TrainingImpact)
        assert impact.distinct_rows == 900
        assert impact.usable_rows == 880
        assert impact.baseline_accuracy == pytest.approx(0.95)
        assert impact.accuracy_ceiling == pytest.approx(0.99)
