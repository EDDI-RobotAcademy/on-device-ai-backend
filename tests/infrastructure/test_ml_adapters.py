"""PyTorch 어댑터 — Domain 이 요구한 숫자를 실제로 내는가."""

from __future__ import annotations

import numpy as np
import pytest

from domain.model.architecture import ArchitectureKind, ModelArchitecture
from domain.model.tensor_spec import ImageTensorSpec, TensorLayout, TensorSpec
from domain.model.training_config import TrainingConfig
from domain.model.training_data_ref import TrainingDataRef
from infrastructure.errors import SourceUnreadable
from infrastructure.ml.torch_architecture import (
    TorchArchitectureProfiler,
    build_module,
)
from infrastructure.ml.torch_materializer import (
    PillowImageTensorMaterializer,
    TorchTensorMaterializer,
)
from infrastructure.ml.torch_trainer import (
    PyTorchModelTrainer,
    TorchModelEvaluator,
)
from infrastructure.ml.windowing import build_windows
from tests.support import model_scenario as ms
from tests.support.scenario import FEATURE_FIELDS

SPEC = TensorSpec(shape=(30, 6), layout=TensorLayout.TIME_FIRST)


def data_ref(path, **overrides):  # noqa: ANN001, ANN003
    base: dict[str, object] = dict(
        dataset_ref="model-train",
        uri=str(path),
        feature_fields=FEATURE_FIELDS,
        label_field="condition",
        time_field="timestamp",
        readiness_certified=True,
        quality_gate_passed=True,
    )
    base.update(overrides)
    return TrainingDataRef(**base)  # type: ignore[arg-type]


class TestWindowing:
    def test_창_수와_모양을_낸다(self, model_data) -> None:
        dataset = build_windows(
            data_ref(model_data.train),
            window_length=30,
            stride=10,
            label_policy=ms.window_label_policy(),
        )
        assert dataset.features["train"].shape[1:] == (30, 6)
        assert dataset.summary.window_count == 1294
        assert dataset.labels == ("FAULT", "OVERLOAD", "NORMAL")

    def test_분할_경계_겹침을_숫자로_돌려준다(self, model_data) -> None:
        overlapping = build_windows(
            data_ref(model_data.train),
            window_length=30,
            stride=10,
            label_policy=ms.window_label_policy(),
        )
        disjoint = build_windows(
            data_ref(model_data.train),
            window_length=30,
            stride=30,
            label_policy=ms.window_label_policy(),
        )
        assert overlapping.boundary_overlap_samples == 40
        assert disjoint.boundary_overlap_samples == 0

    def test_모든_분할에_사건이_들어간다(self, model_data) -> None:
        """M02 — 트립을 36시간에 고르게 배치한 이유."""
        dataset = build_windows(
            data_ref(model_data.train),
            window_length=30,
            stride=10,
            label_policy=ms.window_label_policy(),
        )
        for split in ("train", "validation", "test"):
            targets = dataset.targets[split]
            assert len(set(targets.tolist())) == 3

    def test_정규화_통계를_주면_적용한다(self, model_data) -> None:
        import pandas as pd

        frame = pd.read_csv(model_data.train)
        train = frame.head(int(len(frame) * 0.7))
        stats = {
            f: (float(train[f].mean()), float(train[f].std(ddof=0)))
            for f in FEATURE_FIELDS
        }
        dataset = build_windows(
            data_ref(model_data.train, normalization=stats),
            window_length=30,
            stride=30,
            label_policy=ms.window_label_policy(),
        )
        assert abs(float(dataset.features["train"].mean())) < 0.05
        assert abs(float(dataset.features["train"].std()) - 1.0) < 0.05

    def test_없는_열을_지목하면_거부한다(self, model_data) -> None:
        with pytest.raises(SourceUnreadable, match="입력 열"):
            build_windows(
                data_ref(model_data.train, feature_fields=("없는열",)),
                window_length=30,
                stride=30,
                label_policy=ms.window_label_policy(),
            )

    def test_창이_데이터보다_길면_거부한다(self, model_data) -> None:
        with pytest.raises(SourceUnreadable, match="보다 길다"):
            build_windows(
                data_ref(model_data.train),
                window_length=99_999,
                stride=1,
                label_policy=ms.window_label_policy(),
            )


class TestArchitectureProfiler:
    def test_층별_모양과_파라미터를_센다(self) -> None:
        profile = TorchArchitectureProfiler().profile(ms.cnn_architecture())

        assert profile.input_shape == (30, 6)
        assert profile.output_shape == (3,)
        assert profile.parameter_count > 0
        assert profile.mac_count > 0
        assert any(layer.kind == "Conv1d" for layer in profile.layers)

    def test_MLP_는_파라미터가_더_많다(self) -> None:
        profiler = TorchArchitectureProfiler()
        cnn = profiler.profile(ms.cnn_architecture())
        mlp = profiler.profile(ms.mlp_architecture())
        assert mlp.parameter_count > cnn.parameter_count

    def test_명세대로_모듈이_조립된다(self) -> None:
        import torch

        module = build_module(ms.cnn_architecture())
        output = module(torch.zeros(4, 30, 6))
        assert tuple(output.shape) == (4, 3)

    def test_이미지_구조도_같은_방식으로_센다(self) -> None:
        profile = TorchArchitectureProfiler().profile(
            ModelArchitecture(
                kind=ArchitectureKind.CNN2D,
                input_spec=ImageTensorSpec(width=64, height=64).to_tensor_spec(),
                class_count=2,
                hidden_channels=(8, 16),
                kernel_size=3,
            )
        )
        assert profile.output_shape == (2,)
        assert profile.mac_count > 0


class TestImageMaterializer:
    def test_이미지를_고정된_모양으로_만든다(self, castings) -> None:
        summaries = PillowImageTensorMaterializer().materialize(
            str(castings.root), ImageTensorSpec(width=48, height=48)
        )
        assert summaries["all"].sample_shape == (3, 48, 48)
        assert summaries["unreadable"].sample_count == 3

    def test_흑백_한_채널도_지원한다(self, castings) -> None:
        summaries = PillowImageTensorMaterializer().materialize(
            str(castings.root),
            ImageTensorSpec(width=32, height=32, channels=1, mean=(0.5,), std=(0.5,)),
        )
        assert summaries["all"].sample_shape == (1, 32, 32)

    def test_없는_디렉터리는_거부한다(self, tmp_path) -> None:
        with pytest.raises(SourceUnreadable):
            PillowImageTensorMaterializer().materialize(
                str(tmp_path / "없음"), ImageTensorSpec(width=32, height=32)
            )


@pytest.fixture(scope="module")
def outcome(model_data):  # noqa: ANN201
    """학습은 느리다. 이 모듈에서 한 번만 돌린다."""
    import pandas as pd

    from domain.model.tensor_spec import BatchSpec

    frame = pd.read_csv(model_data.train)
    train = frame.head(int(len(frame) * 0.7))
    stats = {
        f: (float(train[f].mean()), float(train[f].std(ddof=0)))
        for f in FEATURE_FIELDS
    }
    trainer = PyTorchModelTrainer()
    result = trainer.train(
        data_ref(model_data.train, normalization=stats),
        ms.cnn_architecture(),
        TrainingConfig(epochs=4, batch_size=32, learning_rate=3e-3, seed=42),
        BatchSpec(sample=SPEC, batch_size=32),
        30,
        30,
        ms.window_label_policy(),
    )
    return trainer, result


class TestTrainer:
    def test_epoch_마다_기록을_남긴다(self, outcome) -> None:
        _, result = outcome
        assert len(result.epochs) == 4
        assert [r.epoch for r in result.epochs] == [1, 2, 3, 4]
        assert result.epochs[-1].train_loss < result.epochs[0].train_loss

    def test_분할_사용_기록을_함께_돌려준다(self, outcome) -> None:
        _, result = outcome
        assert result.usage.validation_evaluations == 4
        assert result.usage.test_evaluations == 0
        assert result.usage.overlapping_samples == 0

    def test_seed_가_같으면_결과도_같다(self, model_data) -> None:
        from domain.model.tensor_spec import BatchSpec

        def run_once() -> float:
            trainer = PyTorchModelTrainer()
            result = trainer.train(
                data_ref(model_data.train),
                ms.cnn_architecture(),
                TrainingConfig(epochs=2, batch_size=32, learning_rate=3e-3, seed=7),
                BatchSpec(sample=SPEC, batch_size=32),
                30,
                30,
                ms.window_label_policy(),
            )
            return result.epochs[-1].train_loss

        assert run_once() == pytest.approx(run_once(), rel=1e-9)

    def test_평가는_혼동_행렬과_지연시간을_함께_낸다(self, outcome) -> None:
        trainer, _ = outcome
        result = TorchModelEvaluator(trainer.registry).evaluate(
            trainer.last_version_id, "test"
        )
        assert result.matrix.labels == ("FAULT", "OVERLOAD", "NORMAL")
        assert result.matrix.total > 0
        assert result.latency_ms_p95 >= result.latency_ms_p50 > 0

    def test_현장_홀드아웃도_같은_방식으로_평가한다(self, outcome, model_data) -> None:
        trainer, _ = outcome
        result = TorchModelEvaluator(trainer.registry).evaluate_external(
            trainer.last_version_id,
            data_ref(model_data.field, dataset_ref="field"),
            window_length=30,
            stride=30,
            label_policy=ms.window_label_policy(),
            split_name="field",
        )
        assert result.split == "field"
        assert result.matrix.total > 100

    def test_학습되지_않은_모델은_평가할_수_없다(self, outcome) -> None:
        from domain.model.identifiers import ModelVersionId

        trainer, _ = outcome
        with pytest.raises(KeyError):
            TorchModelEvaluator(trainer.registry).evaluate(
                ModelVersionId.of("없는모델"), "test"
            )


class TestModelSampleData:
    def test_같은_seed_는_같은_파일을_만든다(self, tmp_path) -> None:
        from infrastructure.sample_data import write_model_samples

        first = write_model_samples(tmp_path / "a", seed=11)
        second = write_model_samples(tmp_path / "b", seed=11)
        assert first.train.read_bytes() == second.train.read_bytes()

    def test_사건이_시간에_고르게_분포한다(self, model_data) -> None:
        """M02 — 무작위로 뿌리면 test 분할에 사건이 하나도 없게 된다."""
        import pandas as pd

        frame = pd.read_csv(model_data.train)
        positions = np.flatnonzero((frame["condition"] == "FAULT").to_numpy())
        thirds = np.array_split(np.arange(len(frame)), 3)
        for part in thirds:
            assert np.intersect1d(positions, part).size > 0
