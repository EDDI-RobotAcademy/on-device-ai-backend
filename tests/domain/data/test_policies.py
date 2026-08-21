"""Domain Policy — 측정값을 받아 판정한다.

Policy 테스트는 파일도 pandas 도 필요 없다. 측정값을 손으로 만들어 넣으면 된다.
이것이 "측정과 판정을 분리한다"의 실질적인 이득이다.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from domain.data.inspection import InspectionKind, Severity, Verdict
from domain.data.labeling import LabelAgreementMeasurement, LabelPolicy
from domain.data.partition import (
    PartitionMeasurement,
    PartitionPlan,
    PartitionPolicy,
    SplitRatio,
    SplitStrategy,
)
from domain.data.profile import ColumnProfile, DatasetProfile, FieldType
from domain.data.representativeness import (
    FieldDistributionShift,
    RepresentativenessMeasurement,
    RepresentativenessPolicy,
)
from domain.data.schema import DataSchema, FieldRole, FieldSpec, ValueRange
from domain.data.signal import (
    ImageIntegrityMeasurement,
    SensorChannelMeasurement,
    SignalPlausibilityPolicy,
)
from domain.data.time_axis import (
    SamplingInterval,
    TimeAxisMeasurement,
    TimeAxisPolicy,
)
from tests.support.scenario import condition_label_space


def codes(report) -> set[str]:  # noqa: ANN001
    return {f.code for f in report.findings}


class TestSchemaInspection:
    def _profile(self, **overrides: object) -> DatasetProfile:
        defaults = dict(minimum=100.0, maximum=200.0)
        defaults.update(overrides)
        return DatasetProfile(
            row_count=100,
            columns=(
                ColumnProfile("voltage_v", FieldType.REAL, 100, 0, 90, **defaults),  # type: ignore[arg-type]
                ColumnProfile("note", FieldType.TEXT, 100, 0, 5),
            ),
        )

    def _schema(self) -> DataSchema:
        return DataSchema(
            fields=(
                FieldSpec(
                    "voltage_v",
                    FieldType.REAL,
                    FieldRole.FEATURE,
                    value_range=ValueRange(300.0, 440.0),
                ),
                FieldSpec("missing_field", FieldType.REAL, FieldRole.FEATURE),
            )
        )

    def test_선언과_현실의_차이를_전부_드러낸다(self) -> None:
        report = self._schema().inspect(self._profile())

        assert report.kind is InspectionKind.SCHEMA
        assert report.verdict is Verdict.FAILED
        # 물리 범위 아래 / 요구한 필드 없음 / 모르는 열 있음
        assert codes(report) == {
            "BELOW_PHYSICAL_RANGE",
            "FIELD_MISSING",
            "UNDECLARED_FIELD",
        }

    def test_모르는_열은_경고일_뿐_학습을_막지_않는다(self) -> None:
        schema = DataSchema(
            fields=(FieldSpec("voltage_v", FieldType.REAL, FieldRole.FEATURE),)
        )
        report = schema.inspect(self._profile())
        assert codes(report) == {"UNDECLARED_FIELD"}
        assert report.verdict is Verdict.PASSED_WITH_WARNINGS

    def test_정수로_관측된_열을_실수로_선언하는_것은_허용한다(self) -> None:
        profile = DatasetProfile(
            row_count=10,
            columns=(ColumnProfile("current_a", FieldType.INTEGER, 10, 0, 9),),
        )
        schema = DataSchema(
            fields=(FieldSpec("current_a", FieldType.REAL, FieldRole.FEATURE),)
        )
        assert schema.inspect(profile).verdict is Verdict.PASSED


class TestSignalPlausibilityPolicy:
    def test_센서_고착은_학습을_막는다(self) -> None:
        policy = SignalPlausibilityPolicy()
        report = policy.inspect_sensors(
            (
                SensorChannelMeasurement(
                    field_name="voltage_v",
                    total_count=1000,
                    longest_constant_run=600,
                ),
            )
        )
        assert "SIGNAL_STUCK" in codes(report)
        assert report.verdict is Verdict.FAILED

    def test_짧은_연속은_정상_운전으로_본다(self) -> None:
        policy = SignalPlausibilityPolicy(min_constant_run_length=10)
        report = policy.inspect_sensors(
            (
                SensorChannelMeasurement(
                    field_name="voltage_v", total_count=1000, longest_constant_run=8
                ),
            )
        )
        assert report.verdict is Verdict.PASSED

    def test_물리_범위_이탈은_단_한_건이라도_막는다(self) -> None:
        report = SignalPlausibilityPolicy().inspect_sensors(
            (
                SensorChannelMeasurement(
                    field_name="current_a", total_count=10_000, out_of_range_count=1
                ),
            )
        )
        assert "SIGNAL_OUT_OF_RANGE" in codes(report)
        assert report.verdict is Verdict.FAILED

    def test_포화는_경고다_값은_남아_있지만_잘렸다(self) -> None:
        report = SignalPlausibilityPolicy().inspect_sensors(
            (
                SensorChannelMeasurement(
                    field_name="active_power_kw",
                    total_count=1000,
                    saturated_count=50,
                ),
            )
        )
        assert codes(report) == {"SIGNAL_SATURATED"}
        assert report.verdict is Verdict.PASSED_WITH_WARNINGS

    def test_흐림_기준은_Policy_가_정한다_측정기가_아니라(self) -> None:
        measurement = ImageIntegrityMeasurement(
            total_images=100,
            focus_scores=tuple([10.0] * 20 + [500.0] * 80),
        )
        관대 = SignalPlausibilityPolicy(min_focus_score=5.0)
        엄격 = SignalPlausibilityPolicy(min_focus_score=50.0)

        assert "SIGNAL_DEFOCUSED" not in codes(관대.inspect_images(measurement))
        assert "SIGNAL_DEFOCUSED" in codes(엄격.inspect_images(measurement))

    def test_읽히지_않는_이미지는_치명적이다(self) -> None:
        report = SignalPlausibilityPolicy().inspect_images(
            ImageIntegrityMeasurement(
                total_images=100, unreadable_count=1, focus_scores=(500.0,)
            )
        )
        assert "SIGNAL_UNREADABLE" in codes(report)
        assert report.verdict is Verdict.FAILED


class TestTimeAxisPolicy:
    def _policy(self, **overrides: object) -> TimeAxisPolicy:
        base = dict(expected_interval=SamplingInterval(10.0))
        base.update(overrides)
        return TimeAxisPolicy(**base)  # type: ignore[arg-type]

    def _measurement(self, **overrides: object) -> TimeAxisMeasurement:
        base = dict(
            field_name="timestamp",
            record_count=8641,
            first=datetime(2026, 3, 2, 0, 0, 0),
            last=datetime(2026, 3, 3, 0, 0, 0),
            median_interval_seconds=10.0,
        )
        base.update(overrides)
        return TimeAxisMeasurement(**base)  # type: ignore[arg-type]

    def test_깨끗한_시간축은_통과한다(self) -> None:
        assert self._policy().inspect(self._measurement()).verdict is Verdict.PASSED

    def test_역순은_치명적이다(self) -> None:
        report = self._policy().inspect(self._measurement(out_of_order_count=1))
        assert "TIME_OUT_OF_ORDER" in codes(report)
        assert report.verdict is Verdict.FAILED

    def test_중복_시각은_치명적이다(self) -> None:
        report = self._policy().inspect(
            self._measurement(duplicate_timestamp_count=40)
        )
        assert "TIME_DUPLICATED" in codes(report)

    def test_공백은_경고다_사실을_알고_넘어갈_수_있다(self) -> None:
        report = self._policy().inspect(
            self._measurement(gap_count=2, longest_gap_seconds=900.0)
        )
        assert codes(report) == {"TIME_GAP"}
        assert report.verdict is Verdict.PASSED_WITH_WARNINGS

    def test_수집_주기가_약속과_다르면_경고한다(self) -> None:
        report = self._policy().inspect(
            self._measurement(median_interval_seconds=1.0, record_count=86401)
        )
        assert "TIME_INTERVAL_MISMATCH" in codes(report)


class TestLabelPolicy:
    def test_정의되지_않은_라벨이_있으면_막는다(self) -> None:
        report = LabelPolicy().inspect(
            condition_label_space(),
            LabelAgreementMeasurement(
                class_counts={"NORMAL": 900, "OVERLOAD": 100, "FAULT": 50, "???": 3},
                reviewed_sample_count=100,
                disagreement_count=1,
            ),
        )
        assert "LABEL_UNDEFINED" in codes(report)
        assert report.verdict is Verdict.FAILED

    def test_교차_검토가_한_건도_없으면_막는다(self) -> None:
        report = LabelPolicy().inspect(
            condition_label_space(),
            LabelAgreementMeasurement(
                class_counts={"NORMAL": 900, "OVERLOAD": 100, "FAULT": 50}
            ),
        )
        assert "LABEL_NO_CROSS_REVIEW" in codes(report)

    def test_작업자_판단이_갈리면_막는다(self) -> None:
        report = LabelPolicy(min_agreement_ratio=0.9).inspect(
            condition_label_space(),
            LabelAgreementMeasurement(
                class_counts={"NORMAL": 900, "OVERLOAD": 100, "FAULT": 50},
                annotator_count=2,
                reviewed_sample_count=200,
                disagreement_count=40,  # 일치율 0.8
            ),
        )
        assert "LABEL_DISAGREEMENT" in codes(report)
        assert report.verdict is Verdict.FAILED

    def test_불균형은_경고다_막지는_않는다(self) -> None:
        report = LabelPolicy(max_imbalance_ratio=10.0).inspect(
            condition_label_space(),
            LabelAgreementMeasurement(
                class_counts={"NORMAL": 9000, "OVERLOAD": 100, "FAULT": 50},
                annotator_count=2,
                reviewed_sample_count=200,
                disagreement_count=2,
            ),
        )
        assert codes(report) == {"LABEL_IMBALANCED"}
        assert report.verdict is Verdict.PASSED_WITH_WARNINGS

    def test_교차_검토가_없으면_일치율은_1이_아니라_0이다(self) -> None:
        measurement = LabelAgreementMeasurement(class_counts={"A": 1, "B": 1})
        assert measurement.agreement_ratio == 0.0


class TestPartitionPolicy:
    def _plan(self) -> PartitionPlan:
        return PartitionPlan(
            strategy=SplitStrategy.TIME_ORDERED,
            ratio=SplitRatio.of(0.7, 0.15, 0.15),
            time_field="timestamp",
        )

    def test_그룹_누수는_치명적이다(self) -> None:
        report = PartitionPolicy().inspect(
            self._plan(),
            PartitionMeasurement(700, 150, 150, overlapping_group_count=12),
        )
        assert "PARTITION_GROUP_LEAKAGE" in codes(report)
        assert report.verdict is Verdict.FAILED

    def test_시간_누수는_치명적이다(self) -> None:
        report = PartitionPolicy().inspect(
            self._plan(),
            PartitionMeasurement(700, 150, 150, time_overlap_seconds=86_340.0),
        )
        assert "PARTITION_TIME_LEAKAGE" in codes(report)

    def test_클래스_구성비가_크게_다르면_경고한다(self) -> None:
        report = PartitionPolicy(max_class_ratio_gap=0.1).inspect(
            self._plan(),
            PartitionMeasurement(
                700,
                150,
                150,
                class_distribution={
                    "train": {"NORMAL": 690, "FAULT": 10},
                    "test": {"NORMAL": 100, "FAULT": 50},
                },
            ),
        )
        assert "PARTITION_CLASS_SKEW" in codes(report)

    def test_빈_분할은_치명적이다(self) -> None:
        report = PartitionPolicy().inspect(
            self._plan(), PartitionMeasurement(1000, 0, 0)
        )
        assert "PARTITION_EMPTY_SPLIT" in codes(report)
        assert report.verdict is Verdict.FAILED


class TestRepresentativenessPolicy:
    def test_PSI_임계에_따라_경고와_차단이_갈린다(self) -> None:
        policy = RepresentativenessPolicy(min_observed_sample_count=1)

        drifting = policy.inspect(
            RepresentativenessMeasurement(
                reference_label="3월",
                observed_label="7월",
                field_shifts=(FieldDistributionShift("temperature_c", psi=0.15),),
                observed_sample_count=1000,
            )
        )
        assert codes(drifting) == {"REPR_DISTRIBUTION_DRIFTING"}
        assert drifting.verdict is Verdict.PASSED_WITH_WARNINGS

        shifted = policy.inspect(
            RepresentativenessMeasurement(
                reference_label="3월",
                observed_label="7월",
                field_shifts=(FieldDistributionShift("temperature_c", psi=0.9),),
                observed_sample_count=1000,
            )
        )
        assert "REPR_DISTRIBUTION_SHIFTED" in codes(shifted)
        assert shifted.verdict is Verdict.FAILED

    def test_학습_범위_밖으로_나가면_막는다(self) -> None:
        report = RepresentativenessPolicy().inspect(
            RepresentativenessMeasurement(
                reference_label="3월",
                observed_label="7월",
                field_shifts=(
                    FieldDistributionShift(
                        "temperature_c", psi=0.01, coverage_ratio=0.1
                    ),
                ),
                observed_sample_count=1000,
            )
        )
        assert "REPR_COVERAGE_GAP" in codes(report)

    def test_표본이_너무_적으면_이상없음을_믿지_않는다(self) -> None:
        report = RepresentativenessPolicy(min_observed_sample_count=100).inspect(
            RepresentativenessMeasurement(
                reference_label="3월", observed_label="7월", observed_sample_count=10
            )
        )
        assert "REPR_SAMPLE_TOO_SMALL" in codes(report)

    def test_경고_임계가_심각_임계보다_커질_수_없다(self) -> None:
        with pytest.raises(Exception, match="경고 임계"):
            RepresentativenessPolicy(
                psi_warning_threshold=0.5, psi_critical_threshold=0.25
            )

    def test_심각도_구분이_유지된다(self) -> None:
        finding = RepresentativenessPolicy().inspect(
            RepresentativenessMeasurement(
                reference_label="a",
                observed_label="b",
                unseen_category_ratio=0.4,
                observed_sample_count=1000,
            )
        ).findings[0]
        assert finding.severity is Severity.CRITICAL
