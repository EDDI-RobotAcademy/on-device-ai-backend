"""실습 시나리오 빌더.

여기 있는 스키마와 라벨 정의는 실습 1-3, 1-6 에서 학생이 직접 작성하게 되는 것과 같다.
테스트에서는 매번 반복하지 않기 위해 한 곳에 모아 둔다.
"""

from __future__ import annotations

from pathlib import Path

from application.data.declare_data_schema import DeclareDataSchemaCommand
from application.data.define_label_space import DefineLabelSpaceCommand
from application.data.profile_dataset import ProfileDatasetCommand
from application.data.register_dataset import RegisterDatasetCommand
from domain.data.labeling import LabelDefinition, LabelPolicy, LabelSpace
from domain.data.profile import FieldType
from domain.data.schema import DataSchema, FieldRole, FieldSpec, ValueRange
from domain.data.source import DataSourceDescriptor, Modality, SourceFormat
from infrastructure.config.container import DataContainer

TIME_FIELD = "timestamp"
LABEL_FIELD = "condition"
GROUP_FIELD = "batch_id"
SAMPLE_INTERVAL_SECONDS = 10.0

FEATURE_FIELDS: tuple[str, ...] = (
    "active_power_kw",
    "reactive_power_kvar",
    "current_a",
    "voltage_v",
    "temperature_c",
    "spindle_rpm",
)


def power_source(path: Path, collected_from: str = "LINE-3 / PM-MAIN-01") -> DataSourceDescriptor:
    return DataSourceDescriptor(
        uri=str(path),
        format=SourceFormat.CSV,
        modality=Modality.TIME_SERIES,
        collected_from=collected_from,
    )


def image_source(path: Path) -> DataSourceDescriptor:
    return DataSourceDescriptor(
        uri=str(path),
        format=SourceFormat.IMAGE_DIRECTORY,
        modality=Modality.IMAGE,
        collected_from="DIECAST-CELL-A / 상부 카메라",
    )


def power_schema() -> DataSchema:
    """실습 1-3 에서 확정하는 스키마.

    물리 범위(value_range)는 데이터가 아니라 설비 사양서에서 가져온 값이다.
    """
    return DataSchema(
        fields=(
            FieldSpec(TIME_FIELD, FieldType.TIMESTAMP, FieldRole.TIME_INDEX),
            FieldSpec("meter_id", FieldType.CATEGORY, FieldRole.METADATA),
            FieldSpec(GROUP_FIELD, FieldType.CATEGORY, FieldRole.GROUP),
            FieldSpec("product_code", FieldType.CATEGORY, FieldRole.METADATA),
            FieldSpec(
                "active_power_kw",
                FieldType.REAL,
                FieldRole.FEATURE,
                unit="kW",
                value_range=ValueRange(0.0, 400.0),
            ),
            FieldSpec(
                "reactive_power_kvar",
                FieldType.REAL,
                FieldRole.FEATURE,
                unit="kvar",
                value_range=ValueRange(0.0, 400.0),
            ),
            FieldSpec(
                "current_a",
                FieldType.REAL,
                FieldRole.FEATURE,
                unit="A",
                value_range=ValueRange(0.0, 600.0),
            ),
            FieldSpec(
                "voltage_v",
                FieldType.REAL,
                FieldRole.FEATURE,
                unit="V",
                value_range=ValueRange(300.0, 440.0),
            ),
            FieldSpec(
                "temperature_c",
                FieldType.REAL,
                FieldRole.FEATURE,
                unit="℃",
                required=False,
                value_range=ValueRange(-20.0, 120.0),
            ),
            FieldSpec(
                "spindle_rpm",
                FieldType.REAL,
                FieldRole.FEATURE,
                unit="rpm",
                value_range=ValueRange(0.0, 4000.0),
            ),
            FieldSpec(LABEL_FIELD, FieldType.CATEGORY, FieldRole.LABEL),
            FieldSpec(
                "condition_review",
                FieldType.CATEGORY,
                FieldRole.METADATA,
                required=False,
            ),
        )
    )


def condition_label_space() -> LabelSpace:
    """실습 1-6 에서 확정하는 라벨 정의.

    이름만 나열하면 LabelDefinition 이 생성 자체를 거부한다.
    '무엇을 보고 그렇게 판단했는가'가 없으면 라벨이 아니다.
    """
    return LabelSpace(
        field_name=LABEL_FIELD,
        definitions=(
            LabelDefinition(
                name="NORMAL",
                meaning="정격 부하 범위 안에서 운전 중이며 알람이 없는 상태.",
                decided_by="설비운영팀 표준 SOP-PWR-03",
                examples=("주간 정상 생산 구간",),
            ),
            LabelDefinition(
                name="OVERLOAD",
                meaning="유효전력이 정격의 110% 를 5초 이상 초과한 상태. 즉시 고장은 아니다.",
                decided_by="설비운영팀 표준 SOP-PWR-03",
                examples=("금형 교체 직후 초기 부하 구간",),
            ),
            LabelDefinition(
                name="FAULT",
                meaning="보호 계전기가 동작했거나 설비가 정지한 상태.",
                decided_by="보전팀 고장이력 시스템",
                examples=("2026-03-02 14:12 트립 이력",),
            ),
        ),
    )


def strict_label_policy() -> LabelPolicy:
    return LabelPolicy(
        min_agreement_ratio=0.9,
        min_samples_per_class=30,
        max_unlabeled_ratio=0.0,
        max_imbalance_ratio=10.0,
        require_cross_review=True,
    )


# ---------------------------------------------------------------------------
# 단계 진행 헬퍼
# ---------------------------------------------------------------------------
def register(
    container: DataContainer,
    dataset_id: str,
    path: Path,
    *,
    name: str = "3라인 주회로 전력",
    collected_from: str = "LINE-3 / PM-MAIN-01",
) -> None:
    container.register_dataset().execute(
        RegisterDatasetCommand(
            dataset_id=dataset_id,
            name=name,
            uri=str(path),
            source_format=SourceFormat.CSV,
            modality=Modality.TIME_SERIES,
            collected_from=collected_from,
        )
    )


def register_images(container: DataContainer, dataset_id: str, path: Path) -> None:
    container.register_dataset().execute(
        RegisterDatasetCommand(
            dataset_id=dataset_id,
            name="다이캐스팅 부품 표면 이미지",
            uri=str(path),
            source_format=SourceFormat.IMAGE_DIRECTORY,
            modality=Modality.IMAGE,
            collected_from="DIECAST-CELL-A / 상부 카메라",
        )
    )


def profile(container: DataContainer, dataset_id: str):  # noqa: ANN201
    return container.profile_dataset().execute(
        ProfileDatasetCommand(dataset_id=dataset_id)
    )


def declare_schema(container: DataContainer, dataset_id: str, schema: DataSchema | None = None):  # noqa: ANN201
    return container.declare_data_schema().execute(
        DeclareDataSchemaCommand(
            dataset_id=dataset_id, schema=schema or power_schema()
        )
    )


def define_labels(container: DataContainer, dataset_id: str):  # noqa: ANN201
    return container.define_label_space().execute(
        DefineLabelSpaceCommand(
            dataset_id=dataset_id,
            label_space=condition_label_space(),
            policy=strict_label_policy(),
        )
    )


def through_schema(container: DataContainer, dataset_id: str, path: Path) -> None:
    """등록 → 프로파일 → 스키마 선언까지 진행한다."""
    register(container, dataset_id, path)
    profile(container, dataset_id)
    declare_schema(container, dataset_id)


def time_series_readiness_policy(**overrides) -> "ReadinessPolicy":  # noqa: ANN003
    """시계열 데이터의 학습 착수 기준.

    기본 기준에 TIME_AXIS 를 더한다.
    시간축이 없는 이미지 데이터셋에는 요구할 수 없는 검사이므로 기본값이 아니다.
    판정 기준도 데이터 종류에 따라 달라진다.
    """
    from domain.data.inspection import InspectionKind
    from domain.data.readiness import DEFAULT_REQUIRED_KINDS, ReadinessPolicy

    base = dict(
        required_kinds=DEFAULT_REQUIRED_KINDS | {InspectionKind.TIME_AXIS},
        allow_warnings=True,
        max_warning_count=10,
    )
    base.update(overrides)
    return ReadinessPolicy(**base)


def run_full_inspection(
    container: DataContainer,
    dataset_id: str,
    path: Path,
    observed_path: Path,
) -> None:
    """실습 1-1 ~ 1-9 를 순서대로 한 번에 수행한다. (실습 1-10 에서 사용)

    실제 실습에서는 학생이 한 단계씩 직접 호출한다.
    """
    from application.data.analyze_representativeness import (
        AnalyzeRepresentativenessCommand,
    )
    from application.data.design_training_data import DesignTrainingDataCommand
    from application.data.inspect_signal_plausibility import (
        InspectSignalPlausibilityCommand,
    )
    from application.data.inspect_time_axis import InspectTimeAxisCommand
    from application.data.partition_dataset import PartitionDatasetCommand
    from domain.data.partition import PartitionPlan, SplitRatio, SplitStrategy
    from domain.data.signal import SignalPlausibilityPolicy
    from domain.data.time_axis import SamplingInterval, TimeAxisPolicy
    from domain.data.training_spec import (
        NormalizationMethod,
        NormalizationSpec,
        TrainingDataSpec,
        WindowSpec,
    )

    register(container, dataset_id, path)
    profile(container, dataset_id)
    declare_schema(container, dataset_id)

    container.inspect_signal_plausibility().execute(
        InspectSignalPlausibilityCommand(
            dataset_id=dataset_id, policy=SignalPlausibilityPolicy()
        )
    )
    container.inspect_time_axis().execute(
        InspectTimeAxisCommand(
            dataset_id=dataset_id,
            policy=TimeAxisPolicy(
                expected_interval=SamplingInterval(SAMPLE_INTERVAL_SECONDS)
            ),
        )
    )
    define_labels(container, dataset_id)

    container.partition_dataset().execute(
        PartitionDatasetCommand(
            dataset_id=dataset_id,
            plan=PartitionPlan(
                strategy=SplitStrategy.TIME_ORDERED,
                ratio=SplitRatio.of(0.7, 0.15, 0.15),
                time_field=TIME_FIELD,
            ),
        )
    )

    # 학습 설계는 확정된 스키마 객체를 필요로 한다.
    from application.data.support import load_dataset

    aggregate = load_dataset(container.repository, dataset_id)
    container.design_training_data().execute(
        DesignTrainingDataCommand(
            dataset_id=dataset_id,
            spec=TrainingDataSpec(
                schema=aggregate.schema,
                feature_fields=FEATURE_FIELDS,
                label_field=LABEL_FIELD,
                window=WindowSpec(
                    length=30,
                    stride=30,
                    interval=SamplingInterval(SAMPLE_INTERVAL_SECONDS),
                ),
                normalization=NormalizationSpec(method=NormalizationMethod.ZSCORE),
            ),
            fit_normalization=True,
        )
    )

    container.analyze_representativeness().execute(
        AnalyzeRepresentativenessCommand(
            dataset_id=dataset_id, observed=power_source(observed_path, "최근 현장 표본")
        )
    )
