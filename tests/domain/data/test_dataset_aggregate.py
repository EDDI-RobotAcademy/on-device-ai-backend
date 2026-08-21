"""Dataset Aggregate — 순서와 계약을 지킨다. (실습 1-10 의 뼈대)"""

from __future__ import annotations

import pytest

from domain.data.dataset import Dataset, DatasetStatus
from domain.data.errors import SchemaMismatch, UnknownField
from domain.data.identifiers import DatasetId
from domain.data.inspection import (
    Finding,
    InspectionKind,
    InspectionReport,
    Severity,
    Verdict,
)
from domain.data.labeling import LabelDefinition, LabelSpace
from domain.data.partition import (
    PartitionMeasurement,
    PartitionPlan,
    SplitRatio,
    SplitStrategy,
)
from domain.data.profile import ColumnProfile, DatasetProfile, FieldType
from domain.data.readiness import ReadinessPolicy
from domain.data.schema import DataSchema, FieldRole, FieldSpec
from domain.data.source import DataSourceDescriptor, Modality, SourceFormat
from domain.data.time_axis import SamplingInterval
from domain.data.training_spec import TrainingDataSpec, WindowSpec
from domain.shared.errors import IllegalStateTransition, InvariantViolation

SOURCE = DataSourceDescriptor(
    uri="plant_power.csv",
    format=SourceFormat.CSV,
    modality=Modality.TIME_SERIES,
    collected_from="LINE-3 / PM-MAIN-01",
)

SCHEMA = DataSchema(
    fields=(
        FieldSpec("timestamp", FieldType.TIMESTAMP, FieldRole.TIME_INDEX),
        FieldSpec("batch_id", FieldType.CATEGORY, FieldRole.GROUP),
        FieldSpec("active_power_kw", FieldType.REAL, FieldRole.FEATURE),
        FieldSpec("condition", FieldType.CATEGORY, FieldRole.LABEL),
    )
)

PROFILE = DatasetProfile(
    row_count=1000,
    columns=(
        ColumnProfile("timestamp", FieldType.TIMESTAMP, 1000, 0, 1000),
        ColumnProfile("batch_id", FieldType.CATEGORY, 1000, 0, 10),
        ColumnProfile("active_power_kw", FieldType.REAL, 1000, 0, 900, 10.0, 300.0),
        ColumnProfile("condition", FieldType.CATEGORY, 1000, 0, 3),
    ),
)

LABEL_SPACE = LabelSpace(
    field_name="condition",
    definitions=(
        LabelDefinition("NORMAL", "정격 범위 안에서 운전 중", "설비운영팀"),
        LabelDefinition("FAULT", "보호 계전기가 동작한 상태", "보전팀"),
    ),
)


def new_dataset() -> Dataset:
    return Dataset.register(DatasetId.of("ds-1"), "3라인 전력", SOURCE)


def prepared() -> Dataset:
    dataset = new_dataset()
    dataset.attach_profile(PROFILE)
    dataset.declare_schema(SCHEMA)
    return dataset


def clean(kind: InspectionKind) -> InspectionReport:
    return InspectionReport(kind=kind)


def blocking(kind: InspectionKind) -> InspectionReport:
    return InspectionReport(
        kind=kind,
        findings=(Finding("X", "치명적 문제", Severity.CRITICAL),),
    )


class TestLifecycle:
    def test_등록하면_이벤트가_남는다(self) -> None:
        dataset = new_dataset()
        assert dataset.status is DatasetStatus.REGISTERED
        assert [e.event_name for e in dataset.pending_events] == ["DatasetRegistered"]

    def test_열어보지_않고는_스키마를_선언할_수_없다(self) -> None:
        with pytest.raises(IllegalStateTransition, match="먼저 프로파일링"):
            new_dataset().declare_schema(SCHEMA)

    def test_행이_0개인_프로파일은_거부한다(self) -> None:
        empty = DatasetProfile(
            row_count=0,
            columns=(ColumnProfile("timestamp", FieldType.TIMESTAMP, 0, 0, 0),),
        )
        with pytest.raises(InvariantViolation, match="데이터가 없다"):
            new_dataset().attach_profile(empty)

    def test_스키마_선언은_즉시_현실과_대조한다(self) -> None:
        dataset = new_dataset()
        dataset.attach_profile(PROFILE)
        report = dataset.declare_schema(SCHEMA)

        assert report.kind is InspectionKind.SCHEMA
        assert dataset.status is DatasetStatus.INSPECTED
        assert dataset.report_of(InspectionKind.SCHEMA) is report

    def test_스키마_없이_다른_검사를_기록할_수_없다(self) -> None:
        dataset = new_dataset()
        dataset.attach_profile(PROFILE)
        with pytest.raises(IllegalStateTransition, match="스키마 없이"):
            dataset.record_inspection(clean(InspectionKind.TIME_AXIS))

    def test_같은_종류의_검사는_덮어쓴다_재검사가_가능해야_한다(self) -> None:
        dataset = prepared()
        dataset.record_inspection(blocking(InspectionKind.TIME_AXIS))
        assert dataset.report_of(InspectionKind.TIME_AXIS).verdict is Verdict.FAILED

        dataset.record_inspection(clean(InspectionKind.TIME_AXIS))
        assert dataset.report_of(InspectionKind.TIME_AXIS).verdict is Verdict.PASSED


class TestLabelAndTrainingSpec:
    def test_라벨_역할이_아닌_필드를_라벨_공간으로_지정할_수_없다(self) -> None:
        dataset = prepared()
        wrong = LabelSpace(
            field_name="batch_id",
            definitions=LABEL_SPACE.definitions,
        )
        with pytest.raises(SchemaMismatch, match="LABEL 이 아니다"):
            dataset.define_label_space(wrong)

    def test_스키마에_없는_필드를_라벨로_지정할_수_없다(self) -> None:
        dataset = prepared()
        unknown = LabelSpace(field_name="없는열", definitions=LABEL_SPACE.definitions)
        with pytest.raises(UnknownField):
            dataset.define_label_space(unknown)

    def test_라벨_정의_없이_학습_설계를_할_수_없다(self) -> None:
        dataset = prepared()
        spec = TrainingDataSpec(
            schema=SCHEMA,
            feature_fields=("active_power_kw",),
            label_field="condition",
            window=WindowSpec(30, 30, SamplingInterval(10.0)),
        )
        with pytest.raises(IllegalStateTransition, match="LabelSpace"):
            dataset.design_training_data(spec)

    def test_다른_스키마를_참조하는_학습_설계는_거부한다(self) -> None:
        dataset = prepared()
        dataset.define_label_space(LABEL_SPACE)
        other = DataSchema(
            fields=(
                FieldSpec("active_power_kw", FieldType.REAL, FieldRole.FEATURE),
                FieldSpec("condition", FieldType.CATEGORY, FieldRole.LABEL),
            )
        )
        spec = TrainingDataSpec(
            schema=other,
            feature_fields=("active_power_kw",),
            label_field="condition",
        )
        with pytest.raises(SchemaMismatch, match="스키마가 이 Dataset"):
            dataset.design_training_data(spec)


class TestCertification:
    def _full(self, dataset: Dataset) -> None:
        for kind in (
            InspectionKind.SIGNAL_PLAUSIBILITY,
            InspectionKind.LABEL_SPACE,
            InspectionKind.TRAINING_SPEC,
            InspectionKind.PARTITION,
            InspectionKind.REPRESENTATIVENESS,
        ):
            dataset.record_inspection(clean(kind))

    def test_검사가_빠지면_판정에서_막힌다(self) -> None:
        dataset = prepared()
        certificate = dataset.certify(ReadinessPolicy())

        assert certificate.verdict is Verdict.FAILED
        assert dataset.status is DatasetStatus.REJECTED
        assert InspectionKind.PARTITION in certificate.missing_kinds
        assert any("검사 누락" in reason for reason in certificate.reasons())

    def test_모두_통과하면_READY_가_된다(self) -> None:
        dataset = prepared()
        self._full(dataset)
        certificate = dataset.certify(ReadinessPolicy())

        assert certificate.verdict is Verdict.PASSED
        assert dataset.status is DatasetStatus.READY
        assert dataset.is_ready is True

    def test_경고는_기본적으로_통과시키되_기록에_남는다(self) -> None:
        dataset = prepared()
        self._full(dataset)
        dataset.record_inspection(
            InspectionReport(
                kind=InspectionKind.LABEL_SPACE,
                findings=(Finding("W", "불균형", Severity.WARNING),),
            )
        )
        certificate = dataset.certify(ReadinessPolicy(allow_warnings=True))

        assert certificate.verdict is Verdict.PASSED_WITH_WARNINGS
        assert certificate.is_ready is True
        assert len(certificate.warning_findings) == 1

    def test_안전_라인에서는_경고도_막을_수_있다(self) -> None:
        dataset = prepared()
        self._full(dataset)
        dataset.record_inspection(
            InspectionReport(
                kind=InspectionKind.LABEL_SPACE,
                findings=(Finding("W", "불균형", Severity.WARNING),),
            )
        )
        certificate = dataset.certify(ReadinessPolicy(allow_warnings=False))
        assert certificate.verdict is Verdict.FAILED

    def test_READY_상태의_Dataset_은_몰래_바뀌지_않는다(self) -> None:
        dataset = prepared()
        self._full(dataset)
        dataset.certify(ReadinessPolicy())

        with pytest.raises(IllegalStateTransition, match="reopen"):
            dataset.record_inspection(clean(InspectionKind.TIME_AXIS))

    def test_reopen_은_이유를_요구한다(self) -> None:
        dataset = prepared()
        self._full(dataset)
        dataset.certify(ReadinessPolicy())

        with pytest.raises(InvariantViolation, match="이유를 남겨야"):
            dataset.reopen("  ")

        dataset.reopen("전압 센서 교체 후 재수집")
        assert dataset.status is DatasetStatus.INSPECTED
        assert dataset.certificate is None
        assert "DatasetReopened" in [e.event_name for e in dataset.pending_events]

    def test_스키마도_없는_Dataset_은_판정_대상이_아니다(self) -> None:
        dataset = new_dataset()
        dataset.attach_profile(PROFILE)
        with pytest.raises(IllegalStateTransition, match="판정할 수 없다"):
            dataset.certify(ReadinessPolicy())


class TestEvents:
    def test_이벤트는_한_번만_발행된다(self) -> None:
        dataset = prepared()
        first = dataset.pull_events()
        assert [e.event_name for e in first] == [
            "DatasetRegistered",
            "DatasetProfiled",
            "DataSchemaDeclared",
            "InspectionRecorded",
        ]
        assert dataset.pull_events() == ()

    def test_분할_적용은_계획과_실측을_함께_남긴다(self) -> None:
        dataset = prepared()
        plan = PartitionPlan(
            strategy=SplitStrategy.TIME_ORDERED,
            ratio=SplitRatio.of(0.7, 0.15, 0.15),
            time_field="timestamp",
        )
        measurement = PartitionMeasurement(700, 150, 150)
        dataset.apply_partition(plan, measurement, clean(InspectionKind.PARTITION))

        assert dataset.partition is not None
        assert dataset.partition.plan.strategy is SplitStrategy.TIME_ORDERED
        assert dataset.partition.measurement.train_count == 700
        assert "DatasetPartitioned" in [e.event_name for e in dataset.pending_events]
