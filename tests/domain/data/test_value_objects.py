"""Domain Value Object 의 불변식.

Value Object 는 잘못된 상태로 *만들어질 수 없어야* 한다.
"만들어 놓고 나중에 검사한다"는 순간, 검사를 빠뜨린 경로가 반드시 생긴다.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from domain.data.identifiers import DatasetId
from domain.data.inspection import (
    Finding,
    InspectionKind,
    InspectionReport,
    Severity,
    Verdict,
)
from domain.data.labeling import LabelDefinition, LabelSpace
from domain.data.partition import PartitionPlan, SplitRatio, SplitStrategy
from domain.data.profile import ColumnProfile, DatasetProfile, FieldType
from domain.data.schema import DataSchema, FieldRole, FieldSpec, ValueRange
from domain.data.source import DataSourceDescriptor, Modality, SourceFormat
from domain.data.time_axis import SamplingInterval, TimeAxisMeasurement
from domain.shared.errors import InvariantViolation


class TestDataSourceDescriptor:
    def test_이미지_모달리티는_CSV_형식에_담길_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="담을 수 없다"):
            DataSourceDescriptor(
                uri="x.csv",
                format=SourceFormat.CSV,
                modality=Modality.IMAGE,
                collected_from="LINE-3",
            )

    def test_수집_현장이_없으면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="추적이 불가능"):
            DataSourceDescriptor(
                uri="x.csv",
                format=SourceFormat.CSV,
                modality=Modality.TIME_SERIES,
                collected_from="   ",
            )


class TestDatasetId:
    def test_빈_식별자는_거부한다(self) -> None:
        with pytest.raises(InvariantViolation):
            DatasetId.of("  ")


class TestColumnProfile:
    def test_결측_수가_전체를_넘을_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="범위를 벗어났다"):
            ColumnProfile(
                name="v",
                inferred_type=FieldType.REAL,
                total_count=10,
                missing_count=11,
                distinct_count=0,
            )

    def test_결측_비율과_상수_여부를_스스로_계산한다(self) -> None:
        column = ColumnProfile(
            name="meter_id",
            inferred_type=FieldType.CATEGORY,
            total_count=100,
            missing_count=20,
            distinct_count=1,
        )
        assert column.missing_ratio == pytest.approx(0.2)
        assert column.present_count == 80
        assert column.is_constant is True
        assert column.is_all_missing is False


class TestDatasetProfile:
    def test_중복된_열_이름을_거부한다(self) -> None:
        column = ColumnProfile("v", FieldType.REAL, 10, 0, 10)
        with pytest.raises(InvariantViolation, match="중복된 열"):
            DatasetProfile(row_count=10, columns=(column, column))


class TestDataSchema:
    def test_시간축이_둘이면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="시간축은 하나"):
            DataSchema(
                fields=(
                    FieldSpec("t1", FieldType.TIMESTAMP, FieldRole.TIME_INDEX),
                    FieldSpec("t2", FieldType.TIMESTAMP, FieldRole.TIME_INDEX),
                )
            )

    def test_시간축인데_타입이_TIMESTAMP_가_아니면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="TIMESTAMP 가 아니다"):
            FieldSpec("t", FieldType.CATEGORY, FieldRole.TIME_INDEX)

    def test_수치형이_아닌_필드에_물리_범위를_줄_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="물리 범위를 가질 수 없다"):
            FieldSpec(
                "code",
                FieldType.CATEGORY,
                FieldRole.FEATURE,
                value_range=ValueRange(0.0, 1.0),
            )

    def test_범위의_최소가_최대보다_클_수_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            ValueRange(minimum=10.0, maximum=1.0)


class TestInspectionReport:
    def test_판정은_저장되지_않고_findings_에서_유도된다(self) -> None:
        clean = InspectionReport(kind=InspectionKind.SCHEMA)
        assert clean.verdict is Verdict.PASSED
        assert clean.passed is True

        warned = InspectionReport(
            kind=InspectionKind.SCHEMA,
            findings=(Finding("W", "경고", Severity.WARNING),),
        )
        assert warned.verdict is Verdict.PASSED_WITH_WARNINGS
        assert warned.passed is True

        failed = InspectionReport(
            kind=InspectionKind.SCHEMA,
            findings=(
                Finding("W", "경고", Severity.WARNING),
                Finding("C", "치명", Severity.CRITICAL),
            ),
        )
        assert failed.verdict is Verdict.FAILED
        assert failed.passed is False
        assert len(failed.blocking_findings) == 1

    def test_보고서는_만들어진_뒤_바뀌지_않는다(self) -> None:
        report = InspectionReport(
            kind=InspectionKind.SCHEMA,
            findings=(Finding("C", "치명", Severity.CRITICAL),),
        )
        with pytest.raises(FrozenInstanceError):
            report.findings = ()  # type: ignore[misc]
        # verdict 는 필드가 아니라 findings 에서 유도되는 값이므로
        # "합격으로 고쳐 두는" 경로 자체가 존재하지 않는다.
        assert report.verdict is Verdict.FAILED


class TestSamplingInterval:
    def test_주기는_0보다_커야_한다(self) -> None:
        with pytest.raises(InvariantViolation):
            SamplingInterval(seconds=0.0)

    def test_주파수로부터_주기를_만들_수_있다(self) -> None:
        interval = SamplingInterval.from_hertz(0.1)
        assert interval.seconds == pytest.approx(10.0)
        assert interval.expected_count(100.0) == 11


class TestTimeAxisMeasurement:
    def test_마지막_시각이_첫_시각보다_앞설_수_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            TimeAxisMeasurement(
                field_name="t",
                record_count=2,
                first=datetime(2026, 3, 2, 12, 0),
                last=datetime(2026, 3, 2, 11, 0),
                median_interval_seconds=10.0,
            )

    def test_커버리지는_약속_주기를_기준으로_계산한다(self) -> None:
        measurement = TimeAxisMeasurement(
            field_name="t",
            record_count=50,
            first=datetime(2026, 3, 2, 0, 0, 0),
            last=datetime(2026, 3, 2, 0, 16, 30),  # 990초 → 100개가 있어야 한다
            median_interval_seconds=10.0,
        )
        assert measurement.coverage_ratio(SamplingInterval(10.0)) == pytest.approx(0.5)


class TestLabelDefinition:
    def test_판단_기준이_없는_라벨은_만들_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="판단 기준이 없다"):
            LabelDefinition(name="NG", meaning="", decided_by="품질팀")

    def test_누가_정했는지_없으면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="누가 정했는지"):
            LabelDefinition(
                name="NG", meaning="표면 균열이 1mm 이상인 경우", decided_by=""
            )

    def test_클래스가_하나뿐이면_분류_문제가_아니다(self) -> None:
        with pytest.raises(InvariantViolation, match="분류 문제가 성립하지 않는다"):
            LabelSpace(
                field_name="condition",
                definitions=(
                    LabelDefinition("OK", "이상 없음이 확인된 상태", "품질팀"),
                ),
            )


class TestSplitRatio:
    def test_비율의_합이_1이_아니면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="합이"):
            SplitRatio.of(0.7, 0.2, 0.2)

    def test_시간_분할인데_시간_필드가_없으면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="시간 필드가 없다"):
            PartitionPlan(
                strategy=SplitStrategy.TIME_ORDERED,
                ratio=SplitRatio.of(0.7, 0.15, 0.15),
            )

    def test_그룹_분할인데_그룹_필드가_없으면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="그룹 필드가 없다"):
            PartitionPlan(
                strategy=SplitStrategy.GROUP_HOLDOUT,
                ratio=SplitRatio.of(0.7, 0.15, 0.15),
            )
