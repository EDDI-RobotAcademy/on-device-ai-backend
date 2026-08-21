"""PrepareImageTrainingRun — 이미지로 정상과 불량을 가르는 모델을 준비하라. (실습 3-11)

`PrepareTrainingRun` 과 나란히 서는 Use Case 다. 순서는 같다.

    1. 폴더를 재고 판정한다 (이미지 게이트)
    2. TrainingRun 을 준비한다 — 통과 못 했으면 여기서 막힌다
    3. 구조를 조립해 층별 계산량을 센다 (실습 3-2 와 같은 프로파일러)

창을 자르는 단계가 없다. 이미지에는 시간 축이 없기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.data.dto import FindingView
from application.model.dto import PreparationView, TensorSummaryView
from application.model.image_data_mapper import image_data_from
from application.model.support import commit
from application.shared.errors import ConflictingRequest
from application.shared.ports import EventPublisher
from domain.model.architecture import ModelArchitecture
from domain.model.identifiers import TrainingRunId
from domain.model.image_data_ref import ImageReadinessPolicy
from domain.model.ports import (
    ArchitectureProfiler,
    ImageFolderInspector,
    ImageTensorMaterializer,
    TrainingRunRepository,
)
from domain.model.tensor_spec import BatchSpec, ImageTensorSpec
from domain.model.training_config import TrainingConfig
from domain.model.training_run import TrainingRun


@dataclass(frozen=True, slots=True)
class PrepareImageTrainingRunCommand:
    run_id: str
    dataset_ref: str
    root_uri: str
    spec: ImageTensorSpec
    architecture: ModelArchitecture
    config: TrainingConfig
    readiness_policy: ImageReadinessPolicy = ImageReadinessPolicy()
    split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15)
    require_gates: bool = True


class PrepareImageTrainingRun:
    def __init__(
        self,
        runs: TrainingRunRepository,
        inspector: ImageFolderInspector,
        materializer: ImageTensorMaterializer,
        profiler: ArchitectureProfiler,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._inspector = inspector
        self._materializer = materializer
        self._profiler = profiler
        self._publisher = publisher

    def execute(self, command: PrepareImageTrainingRunCommand) -> PreparationView:
        run_id = TrainingRunId.of(command.run_id)
        if self._runs.exists(run_id):
            raise ConflictingRequest(
                f"학습 '{run_id}' 은 이미 존재한다.", subject=str(run_id)
            )

        data, report = image_data_from(
            self._inspector,
            dataset_ref=command.dataset_ref,
            root_uri=command.root_uri,
            spec=command.spec,
            policy=command.readiness_policy,
            split_ratio=command.split_ratio,
        )

        run = TrainingRun.prepare_images(
            run_id,
            data,
            command.architecture,
            command.config,
            require_gates=command.require_gates,
        )

        summaries = self._materializer.materialize(data.root_uri, command.spec)
        run.attach_tensor_summary(summaries["all"])
        run.attach_architecture_profile(self._profiler.profile(command.architecture))
        commit(self._runs, run, self._publisher)

        batch = BatchSpec(
            sample=command.architecture.input_spec, batch_size=command.config.batch_size
        )
        return PreparationView(
            run_id=str(run.id),
            dataset_ref=data.dataset_ref,
            architecture=command.architecture.describe(),
            windowing="창 없음 — 이미지에는 자를 시간 축이 없다",
            input_shape=command.architecture.input_spec.shape,
            batch_shape=batch.shape,
            bytes_per_batch=batch.bytes_per_batch,
            summaries=(TensorSummaryView.of(summaries["all"]),),
            windowing_report=report.describe(),
            findings=tuple(FindingView.of(f) for f in data.readiness_findings),
        )
