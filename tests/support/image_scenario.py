"""이미지 학습 실습 시나리오 빌더. (실습 3-11)

모듈 3 의 시계열 시나리오(`model_scenario.py`)와 같은 자리에 있다.
다른 점은 앞 모듈을 통과시키는 절차가 없다는 것이다 —
이미지에는 이미지의 게이트가 있고, 그건 준비 Use Case 안에서 돈다.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from application.model.evaluate_model import EvaluateModelCommand
from application.model.execute_image_training_run import (
    ExecuteImageTrainingRunCommand,
)
from application.model.prepare_image_training_run import (
    PrepareImageTrainingRunCommand,
)
from domain.model.architecture import ArchitectureKind, GlobalPooling, ModelArchitecture
from domain.model.image_data_ref import ImageReadinessPolicy
from domain.model.tensor_spec import ImageTensorSpec
from domain.model.training_config import TrainingConfig
from infrastructure.config.container import ModelContainer
from infrastructure.monitoring.event_log import RecordingEventPublisher
from infrastructure.persistence.in_memory_training_run_repository import (
    InMemoryTrainingRunRepository,
)

IMAGE_SIZE = 48
"""학습 입력 한 변. 저장은 96×96 이다 — 현장 카메라가 더 크게 찍는다."""

EPOCHS = 25


def image_spec(size: int = IMAGE_SIZE) -> ImageTensorSpec:
    return ImageTensorSpec(width=size, height=size, channels=3)


def image_architecture(
    *,
    class_count: int,
    pooling: GlobalPooling = GlobalPooling.AVERAGE,
    hidden: tuple[int, ...] = (16, 32),
    size: int = IMAGE_SIZE,
) -> ModelArchitecture:
    return ModelArchitecture(
        kind=ArchitectureKind.CNN2D,
        input_spec=image_spec(size).to_tensor_spec(),
        class_count=class_count,
        hidden_channels=hidden,
        kernel_size=3,
        dropout=0.1,
        pooling=pooling,
    )


def training_config(**overrides) -> TrainingConfig:  # noqa: ANN003
    base: dict[str, object] = dict(
        epochs=EPOCHS, batch_size=32, learning_rate=3e-3, seed=42
    )
    base.update(overrides)
    return TrainingConfig(**base)  # type: ignore[arg-type]


def new_container() -> ModelContainer:
    """이미지 학습만 하는 조립품. Dataset/Assessment 저장소는 쓰지 않는다."""
    return ModelContainer(
        runs=InMemoryTrainingRunRepository(),
        publisher=RecordingEventPublisher(),
    )


def prepare(
    container: ModelContainer,
    *,
    run_id: str,
    dataset_ref: str,
    root: Path,
    architecture: ModelArchitecture,
    config: TrainingConfig | None = None,
    policy: ImageReadinessPolicy | None = None,
    size: int = IMAGE_SIZE,
    require_gates: bool = True,
):  # noqa: ANN201
    return container.prepare_image_training_run().execute(
        PrepareImageTrainingRunCommand(
            run_id=run_id,
            dataset_ref=dataset_ref,
            root_uri=str(root),
            spec=image_spec(size),
            architecture=architecture,
            config=config or training_config(),
            readiness_policy=policy or ImageReadinessPolicy(),
            require_gates=require_gates,
        )
    )


def build_pipeline(
    root: Path,
    *,
    class_count: int,
    run_id: str,
    dataset_ref: str,
    pooling: GlobalPooling = GlobalPooling.AVERAGE,
    hidden: tuple[int, ...] = (16, 32),
    config: TrainingConfig | None = None,
):  # noqa: ANN201
    """준비 → 학습 → 평가까지 한 번에. 세션당 한 번만 돌리기 위한 것이다."""
    container = new_container()
    architecture = image_architecture(
        class_count=class_count, pooling=pooling, hidden=hidden
    )
    preparation = prepare(
        container,
        run_id=run_id,
        dataset_ref=dataset_ref,
        root=root,
        architecture=architecture,
        config=config,
    )
    curve = container.execute_image_training_run().execute(
        ExecuteImageTrainingRunCommand(run_id=run_id)
    )
    evaluation = container.evaluate_model().execute(
        EvaluateModelCommand(run_id=run_id, split="test")
    )
    return SimpleNamespace(
        container=container,
        run_id=run_id,
        architecture=architecture,
        preparation=preparation,
        curve=curve,
        evaluation=evaluation,
    )
