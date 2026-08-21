"""Use Case 검증 — 조립과 순서.

여기서는 pandas 를 쓰지 않는다. 가짜 측정기를 끼운다.
Use Case 가 정말로 Port 만 보고 있다면 이렇게 갈아끼워도 동작해야 한다.
동작하지 않는다면 Application 이 기술에 묶여 있다는 뜻이다.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from application.data.certify_dataset_readiness import (
    CertifyDatasetReadinessCommand,
)
from application.data.declare_data_schema import DeclareDataSchemaCommand
from application.data.define_label_space import DefineLabelSpaceCommand
from application.data.get_dataset import GetDatasetQuery
from application.data.inspect_signal_plausibility import (
    InspectSignalPlausibilityCommand,
)
from application.data.inspect_time_axis import InspectTimeAxisCommand
from application.data.profile_dataset import ProfileDatasetCommand
from application.data.register_dataset import RegisterDatasetCommand
from application.shared.errors import ConflictingRequest, UnsupportedOperation
from domain.data.errors import DatasetNotFound
from domain.data.labeling import LabelAgreementMeasurement
from domain.data.profile import ColumnProfile, DatasetProfile, FieldType
from domain.data.readiness import ReadinessPolicy
from domain.data.schema import DataSchema, FieldRole, FieldSpec
from domain.data.signal import SensorChannelMeasurement
from domain.data.source import Modality, SourceFormat
from domain.data.time_axis import SamplingInterval, TimeAxisMeasurement, TimeAxisPolicy
from infrastructure.config.container import DataContainer
from infrastructure.monitoring.event_log import RecordingEventPublisher
from infrastructure.persistence.in_memory_dataset_repository import (
    InMemoryDatasetRepository,
)

SCHEMA = DataSchema(
    fields=(
        FieldSpec("timestamp", FieldType.TIMESTAMP, FieldRole.TIME_INDEX),
        FieldSpec("active_power_kw", FieldType.REAL, FieldRole.FEATURE),
        FieldSpec("condition", FieldType.CATEGORY, FieldRole.LABEL),
    )
)

PROFILE = DatasetProfile(
    row_count=500,
    columns=(
        ColumnProfile("timestamp", FieldType.TIMESTAMP, 500, 0, 500),
        ColumnProfile("active_power_kw", FieldType.REAL, 500, 0, 480, 10.0, 300.0),
        ColumnProfile("condition", FieldType.CATEGORY, 500, 0, 2),
    ),
)


# ---------------------------------------------------------------------------
# 가짜 측정기 — 파일도 pandas 도 없다
# ---------------------------------------------------------------------------
class StubProfiler:
    def __init__(self) -> None:
        self.calls = 0

    def profile(self, source):  # noqa: ANN001, ANN201
        self.calls += 1
        return PROFILE


class StubSensorMeasurer:
    def __init__(self, longest_constant_run: int = 0) -> None:
        self.longest_constant_run = longest_constant_run
        self.received_schema: DataSchema | None = None

    def measure(self, source, schema):  # noqa: ANN001, ANN201
        self.received_schema = schema
        return (
            SensorChannelMeasurement(
                field_name="active_power_kw",
                total_count=500,
                longest_constant_run=self.longest_constant_run,
            ),
        )


class StubImageMeasurer:
    def measure(self, source):  # noqa: ANN001, ANN201
        raise AssertionError("시계열 데이터에 이미지 측정기가 불렸다")


class StubTimeAxisMeasurer:
    def __init__(self) -> None:
        self.field: str | None = None

    def measure(self, source, time_field):  # noqa: ANN001, ANN201
        self.field = time_field
        return TimeAxisMeasurement(
            field_name=time_field,
            record_count=500,
            first=datetime(2026, 3, 2, 0, 0, 0),
            last=datetime(2026, 3, 2, 1, 23, 10),
            median_interval_seconds=10.0,
        )


class StubLabelMeasurer:
    def measure(self, source, label_field):  # noqa: ANN001, ANN201
        return LabelAgreementMeasurement(
            class_counts={"NORMAL": 400, "FAULT": 100},
            annotator_count=2,
            reviewed_sample_count=100,
            disagreement_count=2,
        )


@pytest.fixture
def stub_container() -> DataContainer:
    return DataContainer(
        repository=InMemoryDatasetRepository(),
        publisher=RecordingEventPublisher(),
        profiler=StubProfiler(),
        sensor_measurer=StubSensorMeasurer(),
        image_measurer=StubImageMeasurer(),
        time_axis_measurer=StubTimeAxisMeasurer(),
        label_measurer=StubLabelMeasurer(),
    )


def register(container: DataContainer, dataset_id: str = "ds") -> None:
    container.register_dataset().execute(
        RegisterDatasetCommand(
            dataset_id=dataset_id,
            name="테스트 데이터",
            uri="memory://none",
            source_format=SourceFormat.CSV,
            modality=Modality.TIME_SERIES,
            collected_from="LINE-9",
        )
    )


class TestRegisterDataset:
    def test_같은_식별자를_두_번_등록할_수_없다(self, stub_container) -> None:
        register(stub_container)
        with pytest.raises(ConflictingRequest, match="이미 등록"):
            register(stub_container)

    def test_없는_Dataset_을_조회하면_도메인_예외다(self, stub_container) -> None:
        with pytest.raises(DatasetNotFound):
            stub_container.get_dataset().execute(GetDatasetQuery(dataset_id="없음"))


class TestPortWiring:
    def test_프로파일러는_Use_Case_를_통해서만_불린다(self, stub_container) -> None:
        register(stub_container)
        assert stub_container.profiler.calls == 0

        stub_container.profile_dataset().execute(ProfileDatasetCommand(dataset_id="ds"))
        assert stub_container.profiler.calls == 1

    def test_센서_측정기에_확정된_스키마가_그대로_전달된다(self, stub_container) -> None:
        register(stub_container)
        stub_container.profile_dataset().execute(ProfileDatasetCommand(dataset_id="ds"))
        stub_container.declare_data_schema().execute(
            DeclareDataSchemaCommand(dataset_id="ds", schema=SCHEMA)
        )
        stub_container.inspect_signal_plausibility().execute(
            InspectSignalPlausibilityCommand(dataset_id="ds")
        )
        assert stub_container.sensor_measurer.received_schema == SCHEMA

    def test_시간축_필드는_스키마에서_결정된다_요청이_아니라(self, stub_container) -> None:
        register(stub_container)
        stub_container.profile_dataset().execute(ProfileDatasetCommand(dataset_id="ds"))
        stub_container.declare_data_schema().execute(
            DeclareDataSchemaCommand(dataset_id="ds", schema=SCHEMA)
        )
        stub_container.inspect_time_axis().execute(
            InspectTimeAxisCommand(
                dataset_id="ds",
                policy=TimeAxisPolicy(expected_interval=SamplingInterval(10.0)),
            )
        )
        assert stub_container.time_axis_measurer.field == "timestamp"

    def test_모달리티에_따라_다른_측정기를_고른다(self, stub_container) -> None:
        """시계열에 이미지 측정기가 불리면 StubImageMeasurer 가 터진다."""
        register(stub_container)
        stub_container.profile_dataset().execute(ProfileDatasetCommand(dataset_id="ds"))
        stub_container.declare_data_schema().execute(
            DeclareDataSchemaCommand(dataset_id="ds", schema=SCHEMA)
        )
        view = stub_container.inspect_signal_plausibility().execute(
            InspectSignalPlausibilityCommand(dataset_id="ds")
        )
        assert view.verdict == "PASSED"


class TestOrdering:
    def test_스키마_없이_신호_검사를_요청하면_막는다(self, stub_container) -> None:
        register(stub_container)
        stub_container.profile_dataset().execute(ProfileDatasetCommand(dataset_id="ds"))
        with pytest.raises(UnsupportedOperation, match="스키마가 없다"):
            stub_container.inspect_signal_plausibility().execute(
                InspectSignalPlausibilityCommand(dataset_id="ds")
            )

    def test_라벨_정의는_측정보다_먼저_확정된다(self, stub_container) -> None:
        """정의 없이 측정하면 무엇과 비교할지 알 수 없다."""
        from application.data.support import load_dataset
        from domain.data.labeling import LabelDefinition, LabelSpace

        register(stub_container)
        stub_container.profile_dataset().execute(ProfileDatasetCommand(dataset_id="ds"))
        stub_container.declare_data_schema().execute(
            DeclareDataSchemaCommand(dataset_id="ds", schema=SCHEMA)
        )
        space = LabelSpace(
            field_name="condition",
            definitions=(
                LabelDefinition("NORMAL", "정격 범위 안에서 운전 중", "설비운영팀"),
                LabelDefinition("FAULT", "보호 계전기 동작", "보전팀"),
            ),
        )
        view = stub_container.define_label_space().execute(
            DefineLabelSpaceCommand(dataset_id="ds", label_space=space)
        )
        assert view.verdict == "PASSED"
        assert load_dataset(stub_container.repository, "ds").label_space == space


class TestEventPublication:
    def test_이벤트가_나갔다면_상태는_이미_저장되어_있다(self, stub_container) -> None:
        """Event 만 나가고 상태가 남지 않는 상황을 만들지 않는다."""
        from domain.data.identifiers import DatasetId

        register(stub_container)
        assert stub_container.publisher.names() == ("DatasetRegistered",)
        assert stub_container.repository.find_by_id(DatasetId.of("ds")) is not None

    def test_같은_이벤트를_두_번_발행하지_않는다(self, stub_container) -> None:
        register(stub_container)
        stub_container.profile_dataset().execute(ProfileDatasetCommand(dataset_id="ds"))
        names = stub_container.publisher.names()
        assert names == ("DatasetRegistered", "DatasetProfiled")


class TestCertification:
    def test_판정은_Use_Case_가_계산하지_않는다(self, stub_container) -> None:
        """Application 은 Dataset 에게 시키기만 한다."""
        register(stub_container)
        stub_container.profile_dataset().execute(ProfileDatasetCommand(dataset_id="ds"))
        stub_container.declare_data_schema().execute(
            DeclareDataSchemaCommand(dataset_id="ds", schema=SCHEMA)
        )
        view = stub_container.certify_dataset_readiness().execute(
            CertifyDatasetReadinessCommand(
                dataset_id="ds", policy=ReadinessPolicy()
            )
        )
        assert view.verdict == "FAILED"
        assert "PARTITION" in view.missing_kinds
