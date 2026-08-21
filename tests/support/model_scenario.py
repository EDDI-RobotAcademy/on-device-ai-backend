"""모듈 3 실습 시나리오 빌더.

모듈 3 은 앞의 두 모듈 위에 선다. 그래서 준비 과정이 길다.

    모듈 1 파이프라인 → READY
    모듈 2 파이프라인 → Data Quality Gate PASSED
    그 다음에야 학습을 준비할 수 있다.

이 순서를 매번 손으로 쓰지 않기 위해 여기에 모아 둔다.
실제 실습에서는 학생이 한 단계씩 직접 호출한다.
"""

from __future__ import annotations

from pathlib import Path

from application.data_quality.evaluate_quality_gate import EvaluateQualityGateCommand
from application.model.accept_model import AcceptModelCommand
from application.model.evaluate_model import EvaluateModelCommand
from application.model.execute_training_run import ExecuteTrainingRunCommand
from application.model.prepare_training_run import PrepareTrainingRunCommand
from domain.model.architecture import ArchitectureKind, ModelArchitecture
from domain.model.tensor_spec import TensorLayout, TensorSpec
from domain.model.training_config import TrainingConfig
from domain.model.windowing import WindowingPlan, WindowLabelPolicy
from infrastructure.config.container import (
    DataContainer,
    DataQualityContainer,
    ModelContainer,
)
from tests.support import quality_scenario as qs
from tests.support.scenario import FEATURE_FIELDS

WINDOW_LENGTH = 30
STRIDE = 10
FEATURE_COUNT = len(FEATURE_FIELDS)


def window_label_policy() -> WindowLabelPolicy:
    """창 하나의 라벨을 정하는 규칙. (실습 3-4)

    "30 표본(5분) 중 9 표본이라도 트립이면 그 창은 트립이다."
    이 30% 라는 숫자는 코드가 정하는 것이 아니라 현장이 정한다.
    """
    return WindowLabelPolicy(
        priority=(("FAULT", 0.3), ("OVERLOAD", 0.5)),
        default_label="NORMAL",
    )


def windowing_plan(stride: int = STRIDE) -> WindowingPlan:
    return WindowingPlan(
        window_length=WINDOW_LENGTH,
        stride=stride,
        label_policy=window_label_policy(),
    )


def input_spec() -> TensorSpec:
    return TensorSpec(
        shape=(WINDOW_LENGTH, FEATURE_COUNT), layout=TensorLayout.TIME_FIRST
    )


def cnn_architecture(hidden: tuple[int, ...] = (16, 32)) -> ModelArchitecture:
    return ModelArchitecture(
        kind=ArchitectureKind.CNN1D,
        input_spec=input_spec(),
        class_count=3,
        hidden_channels=hidden,
        kernel_size=5,
        dropout=0.1,
    )


def mlp_architecture(hidden: tuple[int, ...] = (64, 32)) -> ModelArchitecture:
    """시간 축을 펼쳐 버리는 모델. 순서를 못 본다. (실습 3-2 의 비교 대상)"""
    return ModelArchitecture(
        kind=ArchitectureKind.MLP,
        input_spec=input_spec(),
        class_count=3,
        hidden_channels=hidden,
    )


def training_config(**overrides) -> TrainingConfig:  # noqa: ANN003
    base: dict[str, object] = dict(
        epochs=10, batch_size=32, learning_rate=3e-3, seed=42
    )
    base.update(overrides)
    return TrainingConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 앞 모듈 통과시키기
# ---------------------------------------------------------------------------
def pass_both_gates(
    container: DataContainer,
    quality_container: DataQualityContainer,
    *,
    dataset_id: str,
    assessment_id: str,
    path: Path,
) -> None:
    """모듈 1의 학습 착수 판정과 모듈 2의 Data Quality Gate 를 통과시킨다."""
    from application.data.certify_dataset_readiness import (
        CertifyDatasetReadinessCommand,
    )

    qs.prepare_dataset(container, dataset_id, path)
    container.certify_dataset_readiness().execute(
        CertifyDatasetReadinessCommand(
            dataset_id=dataset_id, policy=qs.structural_readiness_policy()
        )
    )
    qs.start(quality_container, assessment_id, dataset_id)
    qs.measure_all(quality_container, assessment_id)
    quality_container.evaluate_quality_gate().execute(
        EvaluateQualityGateCommand(assessment_id=assessment_id)
    )


# ---------------------------------------------------------------------------
# 모듈 3
# ---------------------------------------------------------------------------
def prepare(
    model_container: ModelContainer,
    *,
    run_id: str,
    dataset_id: str,
    assessment_id: str | None,
    architecture: ModelArchitecture | None = None,
    config: TrainingConfig | None = None,
    plan: WindowingPlan | None = None,
    require_gates: bool = True,
):  # noqa: ANN201
    return model_container.prepare_training_run().execute(
        PrepareTrainingRunCommand(
            run_id=run_id,
            dataset_id=dataset_id,
            assessment_id=assessment_id,
            architecture=architecture or cnn_architecture(),
            config=config or training_config(),
            windowing=plan or windowing_plan(),
            require_gates=require_gates,
        )
    )


def execute(model_container: ModelContainer, run_id: str):  # noqa: ANN201
    return model_container.execute_training_run().execute(
        ExecuteTrainingRunCommand(run_id=run_id)
    )


def evaluate(model_container: ModelContainer, run_id: str, split: str = "test", **kwargs):  # noqa: ANN003, ANN201
    return model_container.evaluate_model().execute(
        EvaluateModelCommand(run_id=run_id, split=split, **kwargs)
    )


def accept(model_container: ModelContainer, run_id: str, **kwargs):  # noqa: ANN003, ANN201
    return model_container.accept_model().execute(
        AcceptModelCommand(run_id=run_id, **kwargs)
    )


def evaluate_on_field(
    model_container: ModelContainer, run_id: str, field_uri: str, **kwargs
):  # noqa: ANN003, ANN201
    """현장 홀드아웃으로 평가한다. (실습 3-10)"""
    from application.model.evaluate_model import EvaluateOnFieldCommand

    return model_container.evaluate_model_on_field().execute(
        EvaluateOnFieldCommand(run_id=run_id, field_uri=str(field_uri), **kwargs)
    )


def build_pipeline(
    *,
    dataset_id: str = "power-model",
    assessment_id: str = "qa-power-model",
    run_id: str = "run-cnn",
    train_path: Path,
    stride: int = STRIDE,
    architecture: ModelArchitecture | None = None,
    config: TrainingConfig | None = None,
):  # noqa: ANN201
    """두 게이트 통과 → 준비 → 학습 → 평가까지 한 번에.

    컨테이너를 직접 만든다 (fixture 에 의존하지 않는다).
    세션 단위로 한 번만 돌리기 위한 것이다 — 학습은 매번 하기에는 느리다.
    """
    from types import SimpleNamespace

    from infrastructure.monitoring.event_log import RecordingEventPublisher
    from infrastructure.persistence.in_memory_assessment_repository import (
        InMemoryAssessmentRepository,
    )
    from infrastructure.persistence.in_memory_dataset_repository import (
        InMemoryDatasetRepository,
    )
    from infrastructure.persistence.in_memory_training_run_repository import (
        InMemoryTrainingRunRepository,
    )

    events = RecordingEventPublisher()
    data = DataContainer(repository=InMemoryDatasetRepository(), publisher=events)
    quality = DataQualityContainer(
        datasets=data.repository,
        assessments=InMemoryAssessmentRepository(),
        publisher=events,
    )
    model = ModelContainer(
        datasets=data.repository,
        assessments=quality.assessments,
        runs=InMemoryTrainingRunRepository(),
        publisher=events,
    )

    pass_both_gates(
        data,
        quality,
        dataset_id=dataset_id,
        assessment_id=assessment_id,
        path=train_path,
    )
    preparation = prepare(
        model,
        run_id=run_id,
        dataset_id=dataset_id,
        assessment_id=assessment_id,
        architecture=architecture,
        config=config,
        plan=windowing_plan(stride=stride),
    )
    curve = execute(model, run_id)
    evaluation = evaluate(model, run_id, "test")

    return SimpleNamespace(
        data=data,
        quality=quality,
        model=model,
        events=events,
        run_id=run_id,
        dataset_id=dataset_id,
        assessment_id=assessment_id,
        preparation=preparation,
        curve=curve,
        evaluation=evaluation,
    )


# ---------------------------------------------------------------------------
# 실험 비교 (실습 3-12, 3-14, 3-15)
# ---------------------------------------------------------------------------
def _lab_containers():  # noqa: ANN202
    """게이트를 통과시킨 컨테이너 한 벌. 여러 학습이 이것을 공유한다."""
    from infrastructure.monitoring.event_log import RecordingEventPublisher
    from infrastructure.persistence.in_memory_assessment_repository import (
        InMemoryAssessmentRepository,
    )
    from infrastructure.persistence.in_memory_dataset_repository import (
        InMemoryDatasetRepository,
    )
    from infrastructure.persistence.in_memory_training_run_repository import (
        InMemoryTrainingRunRepository,
    )

    events = RecordingEventPublisher()
    data = DataContainer(repository=InMemoryDatasetRepository(), publisher=events)
    quality = DataQualityContainer(
        datasets=data.repository,
        assessments=InMemoryAssessmentRepository(),
        publisher=events,
    )
    model = ModelContainer(
        datasets=data.repository,
        assessments=quality.assessments,
        runs=InMemoryTrainingRunRepository(),
        publisher=events,
    )
    return data, quality, model, events


def train_variant(
    model_container: ModelContainer,
    *,
    run_id: str,
    dataset_id: str,
    assessment_id: str | None,
    window_length: int,
    stride: int,
    hidden: tuple[int, ...] = (16, 32),
    config: TrainingConfig | None = None,
    require_gates: bool = True,
):  # noqa: ANN201
    """한 가지 설정으로 학습하고 test 까지 평가한다."""
    spec = TensorSpec(
        shape=(window_length, FEATURE_COUNT), layout=TensorLayout.TIME_FIRST
    )
    architecture = ModelArchitecture(
        kind=ArchitectureKind.CNN1D,
        input_spec=spec,
        class_count=3,
        hidden_channels=hidden,
        kernel_size=5,
        dropout=0.1,
    )
    plan = WindowingPlan(
        window_length=window_length,
        stride=stride,
        label_policy=window_label_policy(),
    )
    prepare(
        model_container,
        run_id=run_id,
        dataset_id=dataset_id,
        assessment_id=assessment_id,
        architecture=architecture,
        config=config or training_config(),
        plan=plan,
        require_gates=require_gates,
    )
    execute(model_container, run_id)
    evaluate(model_container, run_id, "test")
    return run_id


def build_structure_lab(train_path: Path):  # noqa: ANN201
    """창 길이만 바꿔 가며 학습한다. (실습 3-12)"""
    from types import SimpleNamespace

    data, quality, model, events = _lab_containers()
    pass_both_gates(
        data,
        quality,
        dataset_id="power-structure",
        assessment_id="qa-power-structure",
        path=train_path,
    )
    windows = (15, 30, 60, 120)
    for length in windows:
        train_variant(
            model,
            run_id=f"run-win{length}",
            dataset_id="power-structure",
            assessment_id="qa-power-structure",
            window_length=length,
            stride=length,  # 겹치지 않게 자른다 (실습 3-8)
        )
    return SimpleNamespace(model=model, events=events, windows=windows)


def build_hyperparameter_lab(train_path: Path):  # noqa: ANN201
    """학습 설정만 바꿔 가며 학습한다. (실습 3-14)"""
    from types import SimpleNamespace

    data, quality, model, events = _lab_containers()
    pass_both_gates(
        data,
        quality,
        dataset_id="power-hp",
        assessment_id="qa-power-hp",
        path=train_path,
    )

    variants = (
        ("base", dict(learning_rate=3e-3), (16, 32)),
        ("lr-낮춤", dict(learning_rate=3e-4), (16, 32)),
        ("채널-키움", dict(learning_rate=3e-3), (32, 64)),
        ("둘-한꺼번에", dict(learning_rate=3e-4), (32, 64)),
    )
    for label, overrides, hidden in variants:
        train_variant(
            model,
            run_id=f"run-hp-{label}",
            dataset_id="power-hp",
            assessment_id="qa-power-hp",
            window_length=WINDOW_LENGTH,
            stride=WINDOW_LENGTH,
            hidden=hidden,
            config=training_config(**overrides),
        )
    return SimpleNamespace(model=model, events=events, variants=variants)


def build_quality_impact_lab(
    *, dirty_path: Path, clean_path: Path, holdout_path: Path
):  # noqa: ANN201
    """오염된 데이터와 정제된 데이터로 각각 학습한다. (실습 3-15)

    **두 모델을 같은 시험지로 채점한다.** 시험지는 어느 쪽 학습 파일과도 겹치지 않는
    '다른 날' 데이터다. 그러지 않으면 정제된 쪽이 자기 답안지를 본 셈이 된다.

    dirty 는 Data Quality Gate 를 통과하지 못한다 — 그게 모듈 2 의 결론이었다.
    여기서는 **일부러 우회해서** 통과시키지 않은 데이터로 학습하면 어떻게 되는지 본다.
    현장에서 이 우회를 하면 안 된다.
    """
    from types import SimpleNamespace

    from application.data.certify_dataset_readiness import (
        CertifyDatasetReadinessCommand,
    )

    data, quality, model, events = _lab_containers()
    results = {}

    for label, path, bypass in (
        ("오염된 데이터", dirty_path, True),
        ("정제된 데이터", clean_path, False),
    ):
        dataset_id = f"power-quality-{'dirty' if bypass else 'clean'}"
        qs.prepare_dataset(data, dataset_id, path)
        data.certify_dataset_readiness().execute(
            CertifyDatasetReadinessCommand(
                dataset_id=dataset_id, policy=qs.structural_readiness_policy()
            )
        )
        assessment_id = f"qa-{dataset_id}"
        qs.start(quality, assessment_id, dataset_id)
        qs.measure_all(quality, assessment_id)
        gate = quality.evaluate_quality_gate().execute(
            EvaluateQualityGateCommand(assessment_id=assessment_id)
        )

        run_id = f"run-quality-{'dirty' if bypass else 'clean'}"
        train_variant(
            model,
            run_id=run_id,
            dataset_id=dataset_id,
            assessment_id=assessment_id,
            window_length=WINDOW_LENGTH,
            stride=WINDOW_LENGTH,
            require_gates=False,  # **일부러 우회한다.** 현장에서는 하지 마라.
        )
        evaluate_on_field(model, run_id, str(holdout_path), split_name="holdout")
        results[label] = SimpleNamespace(run_id=run_id, gate=gate)

    return SimpleNamespace(model=model, events=events, results=results)
