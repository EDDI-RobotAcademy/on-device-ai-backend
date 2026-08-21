"""모듈 2 실습 시나리오 빌더.

라벨 일관성 규칙은 실습 2-4 에서 학생이 현장 담당자에게 받아 적게 되는 것과 같다.
테스트에서는 매번 반복하지 않기 위해 한 곳에 모아 둔다.
"""

from __future__ import annotations

from pathlib import Path

from application.data_quality.evaluate_quality_gate import EvaluateQualityGateCommand
from application.data_quality.measure_balance import MeasureBalanceCommand
from application.data_quality.measure_completeness import MeasureCompletenessCommand
from application.data_quality.measure_label_quality import MeasureLabelQualityCommand
from application.data_quality.measure_noise import MeasureNoiseCommand
from application.data_quality.measure_uniqueness import MeasureUniquenessCommand
from application.data_quality.measure_validity import MeasureValidityCommand
from application.data_quality.score_quality import ScoreQualityCommand
from application.data_quality.start_quality_assessment import (
    StartQualityAssessmentCommand,
)
from domain.data.partition import PartitionPlan, SplitRatio, SplitStrategy
from domain.data.time_axis import SamplingInterval
from domain.data.training_spec import (
    NormalizationMethod,
    NormalizationSpec,
    TrainingDataSpec,
    WindowSpec,
)
from domain.data_quality.gate import QualityGatePolicy
from domain.data_quality.label_quality import LabelConsistencyRule
from infrastructure.config.container import DataContainer, DataQualityContainer
from tests.support.scenario import (
    FEATURE_FIELDS,
    LABEL_FIELD,
    SAMPLE_INTERVAL_SECONDS,
    TIME_FIELD,
    declare_schema,
    define_labels,
    profile,
    register,
)

NORMAL_LABEL = "NORMAL"
OVERLOAD_THRESHOLD_KW = 192.0
NORMAL_MAX_KW = 240.0
FAULT_MAX_KW = 30.0


def label_rules() -> tuple[LabelConsistencyRule, ...]:
    """실습 2-4 의 핵심 재료.

    이 세 줄은 데이터에서 나오지 않는다. 설비 담당자에게서 나온다.
    이것이 없으면 라벨 오류를 찾을 방법 자체가 없다.
    """
    return (
        LabelConsistencyRule(
            label="FAULT",
            field_name="active_power_kw",
            expected_max=FAULT_MAX_KW,
            description="보호 계전기가 동작하면 부하가 실제로 차단된다",
        ),
        LabelConsistencyRule(
            label="NORMAL",
            field_name="active_power_kw",
            expected_max=NORMAL_MAX_KW,
            description="정격의 110% 를 넘는 구간을 NORMAL 이라 부를 수는 없다",
        ),
        LabelConsistencyRule(
            label="OVERLOAD",
            field_name="active_power_kw",
            expected_min=OVERLOAD_THRESHOLD_KW,
            description="과부하 판정 기준 (SOP-PWR-03)",
        ),
    )


def structural_readiness_policy():  # noqa: ANN201
    """모듈 1의 **구조** 검증만 요구하는 기준.

    대표성(REPRESENTATIVENESS)은 '현실 대비' 축이라 여기서는 빼 둔다.
    실습 2-1 에서 그 이유를 직접 확인한다 — 오염된 데이터는 대표성 검사까지 오염시킨다.
    """
    from domain.data.inspection import InspectionKind
    from domain.data.readiness import ReadinessPolicy

    return ReadinessPolicy(
        required_kinds=frozenset(
            {
                InspectionKind.SCHEMA,
                InspectionKind.TIME_AXIS,
                InspectionKind.SIGNAL_PLAUSIBILITY,
                InspectionKind.LABEL_SPACE,
                InspectionKind.TRAINING_SPEC,
                InspectionKind.PARTITION,
            }
        ),
        allow_warnings=True,
        max_warning_count=10,
    )


def prepare_dataset(container: DataContainer, dataset_id: str, path: Path) -> None:
    """모듈 1의 구조 검증을 전부 통과시킨다 — 실습 2-1 의 출발점."""
    from application.data.design_training_data import DesignTrainingDataCommand
    from application.data.inspect_signal_plausibility import (
        InspectSignalPlausibilityCommand,
    )
    from application.data.inspect_time_axis import InspectTimeAxisCommand
    from application.data.partition_dataset import PartitionDatasetCommand
    from application.data.support import load_dataset
    from domain.data.signal import SignalPlausibilityPolicy
    from domain.data.time_axis import TimeAxisPolicy

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


def start(
    quality_container: DataQualityContainer, assessment_id: str, dataset_id: str
):  # noqa: ANN201
    return quality_container.start_quality_assessment().execute(
        StartQualityAssessmentCommand(
            assessment_id=assessment_id, dataset_id=dataset_id
        )
    )


def measure_all(
    quality_container: DataQualityContainer, assessment_id: str
) -> dict[str, object]:
    """여섯 축을 전부 측정한다. (실습 2-2 ~ 2-7)"""
    views: dict[str, object] = {}
    views["COMPLETENESS"] = quality_container.measure_completeness().execute(
        MeasureCompletenessCommand(assessment_id=assessment_id)
    )
    views["VALIDITY"] = quality_container.measure_validity().execute(
        MeasureValidityCommand(assessment_id=assessment_id)
    )
    views["LABEL_QUALITY"] = quality_container.measure_label_quality().execute(
        MeasureLabelQualityCommand(assessment_id=assessment_id, rules=label_rules())
    )
    views["BALANCE"] = quality_container.measure_balance().execute(
        MeasureBalanceCommand(assessment_id=assessment_id)
    )
    views["NOISE"] = quality_container.measure_noise().execute(
        MeasureNoiseCommand(assessment_id=assessment_id)
    )
    views["UNIQUENESS"] = quality_container.measure_uniqueness().execute(
        MeasureUniquenessCommand(assessment_id=assessment_id)
    )
    return views


def score(quality_container: DataQualityContainer, assessment_id: str):  # noqa: ANN201
    return quality_container.score_quality().execute(
        ScoreQualityCommand(assessment_id=assessment_id, label_rules=label_rules())
    )


def run_gate(
    quality_container: DataQualityContainer,
    assessment_id: str,
    policy: QualityGatePolicy | None = None,
):  # noqa: ANN201
    return quality_container.evaluate_quality_gate().execute(
        EvaluateQualityGateCommand(
            assessment_id=assessment_id, policy=policy or QualityGatePolicy()
        )
    )


def full_assessment(
    container: DataContainer,
    quality_container: DataQualityContainer,
    *,
    dataset_id: str,
    assessment_id: str,
    path: Path,
):  # noqa: ANN201
    """실습 2-1 ~ 2-8 을 순서대로 한 번에 수행한다. (실습 2-9, 2-10 에서 사용)"""
    prepare_dataset(container, dataset_id, path)
    start(quality_container, assessment_id, dataset_id)
    measure_all(quality_container, assessment_id)
    return score(quality_container, assessment_id)
