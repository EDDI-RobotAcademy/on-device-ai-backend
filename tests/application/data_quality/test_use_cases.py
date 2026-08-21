"""Data Quality Use Case — 번역과 조립.

두 Context 사이의 번역(Anti-Corruption Layer)이 제대로 도는지가 핵심이다.
여기서도 pandas 를 쓰지 않는다. 가짜 측정기를 끼운다.
"""

from __future__ import annotations

import pytest

from application.data_quality.evaluate_quality_gate import EvaluateQualityGateCommand
from application.data_quality.get_assessment import GetAssessmentQuery
from application.data_quality.measure_balance import MeasureBalanceCommand
from application.data_quality.measure_completeness import MeasureCompletenessCommand
from application.data_quality.measure_label_quality import MeasureLabelQualityCommand
from application.data_quality.start_quality_assessment import (
    StartQualityAssessmentCommand,
)
from application.data_quality.target_mapper import assessment_target_from
from application.shared.errors import ConflictingRequest, UnsupportedOperation
from domain.data.dataset import Dataset
from domain.data.identifiers import DatasetId
from domain.data.profile import ColumnProfile, DatasetProfile, FieldType
from domain.data.schema import DataSchema, FieldRole, FieldSpec, ValueRange
from domain.data.source import DataSourceDescriptor, Modality, SourceFormat
from domain.data_quality.balance import ClassBalanceMeasurement
from domain.data_quality.completeness import (
    FieldMissingness,
    MissingValueMeasurement,
)
from domain.data_quality.errors import AssessmentNotFound
from domain.data_quality.label_quality import (
    LabelConsistencyRule,
    LabelErrorMeasurement,
)
from infrastructure.config.container import DataQualityContainer
from infrastructure.monitoring.event_log import RecordingEventPublisher
from infrastructure.persistence.in_memory_assessment_repository import (
    InMemoryAssessmentRepository,
)
from infrastructure.persistence.in_memory_dataset_repository import (
    InMemoryDatasetRepository,
)

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
        FieldSpec(
            "active_power_kw",
            FieldType.REAL,
            FieldRole.FEATURE,
            value_range=ValueRange(0.0, 400.0),
        ),
        FieldSpec("voltage_v", FieldType.REAL, FieldRole.FEATURE),
        FieldSpec("condition", FieldType.CATEGORY, FieldRole.LABEL),
    )
)

PROFILE = DatasetProfile(
    row_count=1000,
    columns=(
        ColumnProfile("timestamp", FieldType.TIMESTAMP, 1000, 0, 1000),
        ColumnProfile("batch_id", FieldType.CATEGORY, 1000, 0, 10),
        ColumnProfile("active_power_kw", FieldType.REAL, 1000, 0, 900, 10.0, 300.0),
        ColumnProfile("voltage_v", FieldType.REAL, 1000, 0, 900, 370.0, 390.0),
        ColumnProfile("condition", FieldType.CATEGORY, 1000, 0, 2),
    ),
)


class StubMissingMeasurer:
    def __init__(self) -> None:
        self.received: object | None = None

    def measure(self, target):  # noqa: ANN001, ANN201
        self.received = target
        return MissingValueMeasurement(
            fields=tuple(
                FieldMissingness(field_name=name, total_count=1000)
                for name in target.feature_fields
            )
        )


class StubBalanceMeasurer:
    def measure(self, target):  # noqa: ANN001, ANN201
        return ClassBalanceMeasurement(
            class_counts={"NORMAL": 900, "FAULT": 100}
        )


class StubLabelMeasurer:
    def __init__(self) -> None:
        self.rules: tuple[LabelConsistencyRule, ...] = ()

    def measure(self, target, rules=()):  # noqa: ANN001, ANN201
        self.rules = rules
        return LabelErrorMeasurement(total_labeled=1000)


def make_dataset(with_spec: bool = False) -> Dataset:
    dataset = Dataset.register(DatasetId.of("ds-1"), "3라인 전력", SOURCE)
    dataset.attach_profile(PROFILE)
    dataset.declare_schema(SCHEMA)
    if with_spec:
        from domain.data.labeling import LabelDefinition, LabelSpace
        from domain.data.time_axis import SamplingInterval
        from domain.data.training_spec import TrainingDataSpec, WindowSpec

        dataset.define_label_space(
            LabelSpace(
                field_name="condition",
                definitions=(
                    LabelDefinition("NORMAL", "정격 범위 안에서 운전 중", "설비운영팀"),
                    LabelDefinition("FAULT", "보호 계전기가 동작한 상태", "보전팀"),
                ),
            )
        )
        dataset.design_training_data(
            TrainingDataSpec(
                schema=SCHEMA,
                feature_fields=("active_power_kw",),  # voltage 를 뺐다
                label_field="condition",
                window=WindowSpec(30, 30, SamplingInterval(10.0)),
            )
        )
    return dataset


@pytest.fixture
def wiring():  # noqa: ANN201
    datasets = InMemoryDatasetRepository()
    publisher = RecordingEventPublisher()
    container = DataQualityContainer(
        datasets=datasets,
        assessments=InMemoryAssessmentRepository(),
        publisher=publisher,
        missing_measurer=StubMissingMeasurer(),
        balance_measurer=StubBalanceMeasurer(),
        label_measurer=StubLabelMeasurer(),
    )
    return container, datasets, publisher


class TestTargetMapper:
    def test_Dataset_을_우리_쪽_언어로_번역한다(self) -> None:
        target = assessment_target_from(make_dataset())

        assert target.dataset_ref == "ds-1"
        assert target.uri == "plant_power.csv"
        assert target.label_field == "condition"
        assert target.time_field == "timestamp"
        assert target.group_field == "batch_id"
        assert target.range_of("active_power_kw") == (0.0, 400.0)

    def test_학습_설계가_있으면_그_입력_필드를_쓴다(self) -> None:
        """품질은 '모델이 실제로 보게 될 열'에 대해 따져야 의미가 있다."""
        without = assessment_target_from(make_dataset())
        with_spec = assessment_target_from(make_dataset(with_spec=True))

        assert without.feature_fields == ("active_power_kw", "voltage_v")
        assert with_spec.feature_fields == ("active_power_kw",)

    def test_스키마가_없으면_번역할_수_없다(self) -> None:
        dataset = Dataset.register(DatasetId.of("ds-2"), "x", SOURCE)
        with pytest.raises(UnsupportedOperation, match="스키마가 없다"):
            assessment_target_from(dataset)


class TestStartAssessment:
    def test_Dataset_을_읽어_평가를_시작한다(self, wiring) -> None:
        container, datasets, publisher = wiring
        datasets.save(make_dataset())

        view = container.start_quality_assessment().execute(
            StartQualityAssessmentCommand(assessment_id="qa-1", dataset_id="ds-1")
        )
        assert view.assessment_id == "qa-1"
        assert view.dataset_ref == "ds-1"
        assert view.status == "OPEN"
        assert publisher.names() == ("QualityAssessmentStarted",)

    def test_같은_평가를_두_번_시작할_수_없다(self, wiring) -> None:
        container, datasets, _ = wiring
        datasets.save(make_dataset())
        container.start_quality_assessment().execute(
            StartQualityAssessmentCommand(assessment_id="qa-1", dataset_id="ds-1")
        )
        with pytest.raises(ConflictingRequest):
            container.start_quality_assessment().execute(
                StartQualityAssessmentCommand(assessment_id="qa-1", dataset_id="ds-1")
            )

    def test_없는_Dataset_은_시작할_수_없다(self, wiring) -> None:
        from domain.data.errors import DatasetNotFound

        container, _, _ = wiring
        with pytest.raises(DatasetNotFound):
            container.start_quality_assessment().execute(
                StartQualityAssessmentCommand(assessment_id="qa-1", dataset_id="없음")
            )

    def test_없는_평가를_조회하면_도메인_예외다(self, wiring) -> None:
        container, _, _ = wiring
        with pytest.raises(AssessmentNotFound):
            container.get_assessment().execute(GetAssessmentQuery(assessment_id="x"))


class TestPortWiring:
    def _started(self, wiring):  # noqa: ANN001, ANN202
        container, datasets, publisher = wiring
        datasets.save(make_dataset(with_spec=True))
        container.start_quality_assessment().execute(
            StartQualityAssessmentCommand(assessment_id="qa-1", dataset_id="ds-1")
        )
        return container, publisher

    def test_측정기에_번역된_Target_이_전달된다(self, wiring) -> None:
        container, _ = self._started(wiring)
        container.measure_completeness().execute(
            MeasureCompletenessCommand(assessment_id="qa-1")
        )
        received = container.missing_measurer.received
        assert received is not None
        assert received.dataset_ref == "ds-1"
        assert received.feature_fields == ("active_power_kw",)

    def test_라벨_규칙은_Use_Case_를_통해_측정기까지_간다(self, wiring) -> None:
        container, _ = self._started(wiring)
        rule = LabelConsistencyRule(
            label="FAULT",
            field_name="active_power_kw",
            expected_max=30.0,
            description="보호 계전기 동작 시 부하 차단",
        )
        container.measure_label_quality().execute(
            MeasureLabelQualityCommand(assessment_id="qa-1", rules=(rule,))
        )
        assert container.label_measurer.rules == (rule,)

    def test_라벨이_없는_Target_에는_균형을_묻지_않는다(self, wiring) -> None:
        container, datasets, _ = wiring
        dataset = Dataset.register(DatasetId.of("ds-3"), "라벨 없음", SOURCE)
        dataset.attach_profile(
            DatasetProfile(
                row_count=100,
                columns=(
                    ColumnProfile("active_power_kw", FieldType.REAL, 100, 0, 90),
                ),
            )
        )
        dataset.declare_schema(
            DataSchema(
                fields=(
                    FieldSpec("active_power_kw", FieldType.REAL, FieldRole.FEATURE),
                )
            )
        )
        datasets.save(dataset)
        container.start_quality_assessment().execute(
            StartQualityAssessmentCommand(assessment_id="qa-3", dataset_id="ds-3")
        )
        with pytest.raises(UnsupportedOperation, match="라벨 필드가 없다"):
            container.measure_balance().execute(
                MeasureBalanceCommand(assessment_id="qa-3")
            )


class TestGateOrchestration:
    def test_판정은_Use_Case_가_계산하지_않는다(self, wiring) -> None:
        container, datasets, publisher = wiring
        datasets.save(make_dataset(with_spec=True))
        container.start_quality_assessment().execute(
            StartQualityAssessmentCommand(assessment_id="qa-1", dataset_id="ds-1")
        )
        container.measure_completeness().execute(
            MeasureCompletenessCommand(assessment_id="qa-1")
        )
        view = container.evaluate_quality_gate().execute(
            EvaluateQualityGateCommand(assessment_id="qa-1")
        )

        assert view.is_ready is False
        assert len(view.missing_dimensions) == 5
        assert "QualityGateBlocked" in publisher.names()

    def test_점수를_내려면_측정이_먼저다(self, wiring) -> None:
        from application.data_quality.score_quality import ScoreQualityCommand

        container, datasets, _ = wiring
        datasets.save(make_dataset(with_spec=True))
        container.start_quality_assessment().execute(
            StartQualityAssessmentCommand(assessment_id="qa-1", dataset_id="ds-1")
        )
        with pytest.raises(UnsupportedOperation, match="측정한 축이 하나도 없다"):
            container.score_quality().execute(
                ScoreQualityCommand(assessment_id="qa-1")
            )
