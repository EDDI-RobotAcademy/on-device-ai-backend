"""Model Use Case — 세 Context 를 잇는 번역과 Job 흐름.

여기서도 torch 를 쓰지 않는다. 가짜 학습기를 끼운다.
Use Case 가 정말로 Port 만 보고 있다면 이렇게 갈아끼워도 동작해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from application.model.evaluate_model import EvaluateModelCommand
from application.model.execute_training_run import ExecuteTrainingRunCommand
from application.model.get_training_run import GetTrainingRunQuery
from application.model.prepare_training_run import PrepareTrainingRunCommand
from application.model.training_data_mapper import training_data_from
from application.shared.errors import ConflictingRequest, UnsupportedOperation
from domain.model.architecture import ArchitectureProfile, LayerProfile
from domain.model.curve import EpochRecord
from domain.model.errors import TrainingRunNotFound
from domain.model.evaluation import ConfusionMatrix, EvaluationResult
from domain.model.protocol import SplitUsage
from domain.model.tensor_spec import DatasetTensorSummary
from domain.model.windowing import WindowingSummary
from domain.shared.errors import IllegalStateTransition
from infrastructure.config.container import ModelContainer
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
from tests.application.data_quality.test_use_cases import make_dataset
from tests.support import model_scenario as ms


class StubMaterializer:
    def __init__(self) -> None:
        self.received: object | None = None

    def materialize(self, data, window_length, stride, label_policy):  # noqa: ANN001, ANN201
        self.received = data
        summary = WindowingSummary(
            source_row_count=1000,
            window_length=window_length,
            stride=stride,
            window_count=98,
            label_counts={"NORMAL": 80, "OVERLOAD": 15, "FAULT": 3},
        )
        splits = {
            "train": 68,
            "validation": 15,
            "test": 15,
        }
        return summary, {
            split: DatasetTensorSummary(
                split=split,
                sample_count=count,
                sample_shape=(window_length, len(data.feature_fields)),
                class_counts={"NORMAL": count},
            )
            for split, count in splits.items()
        }


class StubProfiler:
    def profile(self, architecture):  # noqa: ANN001, ANN201
        return ArchitectureProfile(
            layers=(LayerProfile("fc", "Linear", (3,), 100, 100),),
            input_shape=architecture.input_spec.shape,
            output_shape=(architecture.class_count,),
        )


@dataclass(slots=True)
class StubOutcome:
    epochs: tuple[EpochRecord, ...]
    usage: SplitUsage
    artifact_uri: str


class StubTrainer:
    def __init__(self, epochs: int = 3) -> None:
        self.calls = 0
        self._epochs = epochs

    def train(self, data, architecture, config, batch, window_length, stride, label_policy):  # noqa: ANN001, ANN201
        self.calls += 1
        records = tuple(
            EpochRecord(
                epoch=i + 1,
                train_loss=1.0 / (i + 1),
                validation_loss=1.0 / (i + 1),
                train_accuracy=0.5 + i * 0.1,
                validation_accuracy=0.5 + i * 0.1,
            )
            for i in range(self._epochs)
        )
        return StubOutcome(
            epochs=records,
            usage=SplitUsage(
                train_sample_count=68,
                validation_sample_count=60,
                test_sample_count=60,
                validation_evaluations=self._epochs,
            ),
            artifact_uri="memory://models/mv-stub",
        )


class StubEvaluator:
    def evaluate(self, model_version_id, split):  # noqa: ANN001, ANN201
        return EvaluationResult(
            split=split,
            matrix=ConfusionMatrix.from_pairs(
                ("FAULT", "OVERLOAD", "NORMAL"),
                [("FAULT", "FAULT")] * 8
                + [("OVERLOAD", "OVERLOAD")] * 40
                + [("NORMAL", "NORMAL")] * 100,
            ),
            latency_ms_p95=1.5,
        )


@pytest.fixture
def wiring():  # noqa: ANN201
    datasets = InMemoryDatasetRepository()
    assessments = InMemoryAssessmentRepository()
    publisher = RecordingEventPublisher()
    container = ModelContainer(
        datasets=datasets,
        assessments=assessments,
        runs=InMemoryTrainingRunRepository(),
        publisher=publisher,
        materializer=StubMaterializer(),
        profiler=StubProfiler(),
        trainer=StubTrainer(),
        evaluator=StubEvaluator(),
    )
    return container, datasets, publisher


def prepare_command(**overrides) -> PrepareTrainingRunCommand:  # noqa: ANN003
    base: dict[str, object] = dict(
        run_id="run-1",
        dataset_id="ds-1",
        assessment_id=None,
        architecture=ms.cnn_architecture(),
        config=ms.training_config(epochs=3),
        windowing=ms.windowing_plan(),
        require_gates=False,
    )
    base.update(overrides)
    return PrepareTrainingRunCommand(**base)  # type: ignore[arg-type]


class TestTrainingDataMapper:
    def test_학습_설계가_없으면_번역할_수_없다(self) -> None:
        with pytest.raises(UnsupportedOperation, match="학습 설계"):
            training_data_from(make_dataset())

    def test_게이트_통과_여부를_함께_담는다(self) -> None:
        ref = training_data_from(make_dataset(with_spec=True))
        assert ref.dataset_ref == "ds-1"
        assert ref.label_field == "condition"
        assert ref.gates_passed is False  # 판정도 품질 평가도 없었다
        assert len(ref.missing_gates) == 2

    def test_학습_설계의_입력_필드를_쓴다(self) -> None:
        ref = training_data_from(make_dataset(with_spec=True))
        assert ref.feature_fields == ("active_power_kw",)


class TestPrepare:
    def test_준비하면_요약이_함께_돌아온다(self, wiring) -> None:
        container, datasets, publisher = wiring
        datasets.save(make_dataset(with_spec=True))

        view = container.prepare_training_run().execute(
            prepare_command(architecture=_single_feature_architecture())
        )
        assert view.run_id == "run-1"
        assert len(view.summaries) == 3
        assert "TrainingRunPrepared" in publisher.names()

    def test_같은_학습을_두_번_준비할_수_없다(self, wiring) -> None:
        container, datasets, _ = wiring
        datasets.save(make_dataset(with_spec=True))
        command = prepare_command(architecture=_single_feature_architecture())
        container.prepare_training_run().execute(command)
        with pytest.raises(ConflictingRequest):
            container.prepare_training_run().execute(command)

    def test_없는_학습을_조회하면_도메인_예외다(self, wiring) -> None:
        container, _, _ = wiring
        with pytest.raises(TrainingRunNotFound):
            container.get_training_run().execute(GetTrainingRunQuery(run_id="없음"))


class TestExecute:
    def _prepared(self, wiring):  # noqa: ANN001, ANN202
        container, datasets, publisher = wiring
        datasets.save(make_dataset(with_spec=True))
        container.prepare_training_run().execute(
            prepare_command(architecture=_single_feature_architecture())
        )
        return container, publisher

    def test_학습기는_Use_Case_를_통해서만_불린다(self, wiring) -> None:
        container, _ = self._prepared(wiring)
        assert container.trainer.calls == 0

        container.execute_training_run().execute(
            ExecuteTrainingRunCommand(run_id="run-1")
        )
        assert container.trainer.calls == 1

    def test_epoch_이_Aggregate_에_기록된다(self, wiring) -> None:
        container, publisher = self._prepared(wiring)
        view = container.execute_training_run().execute(
            ExecuteTrainingRunCommand(run_id="run-1")
        )
        assert view.epoch_count == 3
        assert view.status == "COMPLETED"
        assert publisher.names().count("EpochCompleted") == 3

    def test_평가는_학습이_끝난_뒤에만(self, wiring) -> None:
        container, _ = self._prepared(wiring)
        from domain.model.errors import ModelNotTrained

        with pytest.raises(ModelNotTrained):
            container.evaluate_model().execute(EvaluateModelCommand(run_id="run-1"))

    def test_평가_결과가_Aggregate_에_남는다(self, wiring) -> None:
        container, publisher = self._prepared(wiring)
        container.execute_training_run().execute(
            ExecuteTrainingRunCommand(run_id="run-1")
        )
        view = container.evaluate_model().execute(EvaluateModelCommand(run_id="run-1"))

        assert view.split == "test"
        assert view.accuracy > 0.9
        assert "ModelEvaluated" in publisher.names()

    def test_두_번_실행할_수_없다(self, wiring) -> None:
        container, _ = self._prepared(wiring)
        container.execute_training_run().execute(
            ExecuteTrainingRunCommand(run_id="run-1")
        )
        with pytest.raises(IllegalStateTransition):
            container.execute_training_run().execute(
                ExecuteTrainingRunCommand(run_id="run-1")
            )


def _single_feature_architecture():  # noqa: ANN202
    """make_dataset(with_spec=True) 의 입력 필드가 하나뿐이라 맞춰 준다."""
    from domain.model.architecture import ArchitectureKind, ModelArchitecture
    from domain.model.tensor_spec import TensorLayout, TensorSpec

    return ModelArchitecture(
        kind=ArchitectureKind.CNN1D,
        input_spec=TensorSpec(shape=(30, 1), layout=TensorLayout.TIME_FIRST),
        class_count=3,
        hidden_channels=(8,),
    )
