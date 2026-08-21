"""PrepareTrainingRun — 데이터를 처음으로 AI에게 먹여보자. (실습 3-1, 3-2, 3-4)

이 Use Case 는 학습을 시작하지 않는다. **먹일 수 있는지 확인만 한다.**

    1. Dataset 을 TrainingDataRef 로 번역한다 (게이트 통과 여부 포함)
    2. TrainingRun 을 준비한다 — 게이트를 안 통과했으면 여기서 막힌다
    3. 데이터를 실제로 텐서로 바꿔 본다 (실습 3-1, 3-4)
    4. 구조를 조립해 층별 계산량을 센다 (실습 3-2)

여기까지 통과해야 학습을 시작할 수 있다.
"첫 배치에서 shape 이 안 맞아 터지는" 일을 시작 전에 잡는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.data.dto import FindingView
from application.data.support import load_dataset
from application.data_quality.support import load_assessment
from application.model.dto import PreparationView, TensorSummaryView
from application.model.support import commit
from application.model.training_data_mapper import training_data_from
from application.shared.errors import ConflictingRequest
from application.shared.ports import EventPublisher
from domain.data.ports import DatasetRepository
from domain.data_quality.ports import QualityAssessmentRepository
from domain.model.architecture import ModelArchitecture
from domain.model.identifiers import TrainingRunId
from domain.model.ports import (
    ArchitectureProfiler,
    TensorMaterializer,
    TrainingRunRepository,
)
from domain.model.tensor_spec import BatchSpec
from domain.model.training_config import TrainingConfig
from domain.model.training_run import TrainingRun
from domain.model.windowing import WindowingPlan, WindowingPolicy


@dataclass(frozen=True, slots=True)
class PrepareTrainingRunCommand:
    run_id: str
    dataset_id: str
    architecture: ModelArchitecture
    config: TrainingConfig
    windowing: WindowingPlan
    assessment_id: str | None = None
    windowing_policy: WindowingPolicy = WindowingPolicy()
    require_gates: bool = True


class PrepareTrainingRun:
    def __init__(
        self,
        runs: TrainingRunRepository,
        datasets: DatasetRepository,
        assessments: QualityAssessmentRepository,
        materializer: TensorMaterializer,
        profiler: ArchitectureProfiler,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._datasets = datasets
        self._assessments = assessments
        self._materializer = materializer
        self._profiler = profiler
        self._publisher = publisher

    def execute(self, command: PrepareTrainingRunCommand) -> PreparationView:
        run_id = TrainingRunId.of(command.run_id)
        if self._runs.exists(run_id):
            raise ConflictingRequest(
                f"학습 '{run_id}' 은 이미 존재한다.", subject=str(run_id)
            )

        dataset = load_dataset(self._datasets, command.dataset_id)
        assessment = (
            load_assessment(self._assessments, command.assessment_id)
            if command.assessment_id
            else None
        )
        data = training_data_from(dataset, assessment)

        # 게이트를 통과하지 않았다면 Domain 이 여기서 막는다.
        run = TrainingRun.prepare(
            run_id,
            data,
            command.architecture,
            command.config,
            command.windowing,
            require_gates=command.require_gates,
        )

        windowing_summary, summaries = self._materializer.materialize(
            data,
            command.windowing.window_length,
            command.windowing.stride,
            command.windowing.label_policy,
        )
        run.attach_windowing_summary(windowing_summary)
        for summary in summaries.values():
            run.attach_tensor_summary(summary)

        run.attach_architecture_profile(self._profiler.profile(command.architecture))
        commit(self._runs, run, self._publisher)

        batch = BatchSpec(
            sample=command.architecture.input_spec, batch_size=command.config.batch_size
        )
        findings = command.windowing_policy.inspect(windowing_summary)

        return PreparationView(
            run_id=str(run.id),
            dataset_ref=data.dataset_ref,
            architecture=command.architecture.describe(),
            windowing=command.windowing.describe(),
            input_shape=command.architecture.input_spec.shape,
            batch_shape=batch.shape,
            bytes_per_batch=batch.bytes_per_batch,
            summaries=tuple(
                TensorSummaryView.of(summaries[split])
                for split in ("train", "validation", "test")
                if split in summaries
            ),
            windowing_report=windowing_summary.describe(),
            findings=tuple(FindingView.of(f) for f in findings),
        )
