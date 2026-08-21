"""Model Context 의 Value Object 불변식.

torch 도 pandas 도 부르지 않는다. 모양과 숫자만 있다.
"""

from __future__ import annotations

import pytest

from domain.model.architecture import (
    ArchitectureKind,
    ArchitectureProfile,
    LayerProfile,
    ModelArchitecture,
)
from domain.model.curve import EpochRecord, TrainingCurve
from domain.model.evaluation import ConfusionMatrix
from domain.model.tensor_spec import (
    BatchSpec,
    DatasetTensorSummary,
    ImageTensorSpec,
    TensorLayout,
    TensorSpec,
)
from domain.model.training_config import Optimizer, TrainingConfig
from domain.model.training_data_ref import TrainingDataRef
from domain.model.windowing import WindowingPlan, WindowingSummary, WindowLabelPolicy
from domain.shared.errors import InvariantViolation


class TestTensorSpec:
    def test_모양이_비어_있으면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="모양이 비어"):
            TensorSpec(shape=(), layout=TensorLayout.TIME_FIRST)

    def test_0_이하_축은_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="1 이상"):
            TensorSpec(shape=(30, 0), layout=TensorLayout.TIME_FIRST)

    def test_원소_수와_바이트를_계산한다(self) -> None:
        spec = TensorSpec(shape=(30, 6), layout=TensorLayout.TIME_FIRST)
        assert spec.element_count == 180
        assert spec.bytes_per_sample == 720
        assert TensorSpec(
            shape=(30, 6), layout=TensorLayout.TIME_FIRST, dtype="int8"
        ).bytes_per_sample == 180

    def test_배치_축을_앞에_붙인다(self) -> None:
        spec = TensorSpec(shape=(3, 224, 224), layout=TensorLayout.CHANNEL_FIRST)
        assert spec.with_batch(8) == (8, 3, 224, 224)
        with pytest.raises(InvariantViolation):
            spec.with_batch(0)


class TestBatchSpec:
    def test_배치_수를_센다(self) -> None:
        spec = TensorSpec(shape=(30, 6), layout=TensorLayout.TIME_FIRST)
        batch = BatchSpec(sample=spec, batch_size=32)
        assert batch.batch_count(1000) == 32
        assert batch.batch_count(96) == 3
        assert BatchSpec(sample=spec, batch_size=32, drop_last=True).batch_count(
            1000
        ) == 31


class TestImageTensorSpec:
    def test_축_순서에_따라_모양이_달라진다(self) -> None:
        first = ImageTensorSpec(width=224, height=112, layout=TensorLayout.CHANNEL_FIRST)
        last = ImageTensorSpec(width=224, height=112, layout=TensorLayout.CHANNEL_LAST)
        assert first.to_tensor_spec().shape == (3, 112, 224)
        assert last.to_tensor_spec().shape == (112, 224, 3)

    def test_지원하지_않는_채널은_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="채널"):
            ImageTensorSpec(width=64, height=64, channels=4)


class TestDatasetTensorSummary:
    def test_모델_입력과_모양을_맞춰_본다(self) -> None:
        summary = DatasetTensorSummary(
            split="train", sample_count=100, sample_shape=(30, 6)
        )
        assert summary.matches(TensorSpec(shape=(30, 6), layout=TensorLayout.TIME_FIRST))
        assert not summary.matches(
            TensorSpec(shape=(60, 6), layout=TensorLayout.TIME_FIRST)
        )

    def test_NaN_이_있으면_유한하지_않다(self) -> None:
        assert DatasetTensorSummary("train", 10, (30, 6), nan_count=1).is_finite is False


class TestModelArchitecture:
    def test_입력_차원이_구조와_맞아야_한다(self) -> None:
        with pytest.raises(InvariantViolation, match="차원 입력"):
            ModelArchitecture(
                kind=ArchitectureKind.CNN1D,
                input_spec=TensorSpec(
                    shape=(3, 64, 64), layout=TensorLayout.CHANNEL_FIRST
                ),
                class_count=3,
            )

    def test_클래스가_둘_미만이면_분류가_아니다(self) -> None:
        with pytest.raises(InvariantViolation, match="분류 문제"):
            ModelArchitecture(
                kind=ArchitectureKind.CNN1D,
                input_spec=TensorSpec(shape=(30, 6), layout=TensorLayout.TIME_FIRST),
                class_count=1,
            )

    def test_커널은_홀수여야_한다(self) -> None:
        with pytest.raises(InvariantViolation, match="홀수"):
            ModelArchitecture(
                kind=ArchitectureKind.CNN1D,
                input_spec=TensorSpec(shape=(30, 6), layout=TensorLayout.TIME_FIRST),
                class_count=3,
                kernel_size=4,
            )


class TestArchitectureProfile:
    def test_무거운_층과_바쁜_층은_다를_수_있다(self) -> None:
        profile = ArchitectureProfile(
            layers=(
                LayerProfile("conv", "Conv1d", (32, 30), parameter_count=500, mac_count=90_000),
                LayerProfile("fc", "Linear", (3,), parameter_count=9_000, mac_count=9_000),
            )
        )
        assert profile.parameter_count == 9_500
        assert profile.mac_count == 99_000
        assert profile.heaviest_layer.name == "fc"
        assert profile.busiest_layer.name == "conv"
        assert profile.parameter_bytes == 38_000


class TestTrainingConfig:
    def test_잘못된_설정은_만들어지지_않는다(self) -> None:
        with pytest.raises(InvariantViolation):
            TrainingConfig(batch_size=0)
        with pytest.raises(InvariantViolation):
            TrainingConfig(weight_decay=-1.0)

    def test_설명에_재현에_필요한_값이_들어간다(self) -> None:
        text = TrainingConfig(seed=7, optimizer=Optimizer.SGD).describe()
        assert "seed=7" in text and "SGD" in text


class TestWindowing:
    def test_우선순위_규칙으로_창의_라벨을_정한다(self) -> None:
        policy = WindowLabelPolicy(
            priority=(("FAULT", 0.3), ("OVERLOAD", 0.5)), default_label="NORMAL"
        )
        assert policy.label_for({"NORMAL": 21, "FAULT": 9}) == "FAULT"
        assert policy.label_for({"NORMAL": 14, "OVERLOAD": 16}) == "OVERLOAD"
        assert policy.label_for({"NORMAL": 30}) == "NORMAL"
        assert policy.label_for({}) == "NORMAL"

    def test_stride_가_창보다_크면_거부한다(self) -> None:
        policy = WindowLabelPolicy(priority=(("F", 0.3),), default_label="N")
        with pytest.raises(InvariantViolation, match="통째로 버려진다"):
            WindowingPlan(window_length=30, stride=31, label_policy=policy)

    def test_겹침과_독립_표본_수를_계산한다(self) -> None:
        summary = WindowingSummary(
            source_row_count=1000, window_length=30, stride=10, window_count=98
        )
        assert summary.overlap_ratio == pytest.approx(2 / 3)
        assert summary.effective_sample_count == 32
        assert summary.coverage_ratio == pytest.approx(1.0)


class TestTrainingCurve:
    def _curve(self) -> TrainingCurve:
        return TrainingCurve(
            records=(
                EpochRecord(1, 1.0, 0.9, 0.4, 0.45),
                EpochRecord(2, 0.4, 0.30, 0.85, 0.90),
                EpochRecord(3, 0.1, 0.55, 0.99, 0.84),
            )
        )

    def test_최저점과_과적합_시작점을_찾는다(self) -> None:
        curve = self._curve()
        assert curve.best_epoch.epoch == 2
        assert curve.overfitting_epoch == 3
        assert curve.wasted_epochs == 1
        assert curve.train_loss_drop == pytest.approx(0.9)
        assert curve.final_gap == pytest.approx(0.15)

    def test_순서가_틀린_기록은_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="순서대로"):
            TrainingCurve(
                records=(EpochRecord(2, 1.0, 1.0, 0.5, 0.5), EpochRecord(1, 1.0, 1.0, 0.5, 0.5))
            )

    def test_정확도는_0에서_1_사이여야_한다(self) -> None:
        with pytest.raises(InvariantViolation):
            EpochRecord(1, 1.0, 1.0, 1.5, 0.5)


class TestConfusionMatrix:
    def test_모르는_라벨은_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="모르는 라벨"):
            ConfusionMatrix.from_pairs(("A", "B"), [("A", "C")])

    def test_행렬_크기가_맞아야_한다(self) -> None:
        with pytest.raises(InvariantViolation, match="2×2"):
            ConfusionMatrix(labels=("A", "B"), counts=((1, 2, 3), (4, 5, 6)))

    def test_한_번도_예측하지_않은_클래스를_찾는다(self) -> None:
        matrix = ConfusionMatrix.from_pairs(
            ("A", "B"), [("A", "B")] * 5 + [("B", "B")] * 95
        )
        assert matrix.never_predicted == ("A",)
        assert matrix.recall_of("A") == 0.0
        assert matrix.macro_recall == 0.5
        assert matrix.baseline_accuracy == 0.95


class TestTrainingDataRef:
    def test_게이트_통과_여부를_함께_들고_다닌다(self) -> None:
        ref = TrainingDataRef(
            dataset_ref="ds",
            uri="x.csv",
            feature_fields=("a",),
            label_field="y",
        )
        assert ref.gates_passed is False
        assert len(ref.missing_gates) == 2

        passed = TrainingDataRef(
            dataset_ref="ds",
            uri="x.csv",
            feature_fields=("a",),
            label_field="y",
            readiness_certified=True,
            quality_gate_passed=True,
        )
        assert passed.gates_passed is True
        assert passed.missing_gates == ()

    def test_라벨이_없으면_지도학습을_할_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="라벨 필드가 없다"):
            TrainingDataRef(dataset_ref="ds", uri="x.csv", feature_fields=("a",))

    def test_분할_비율의_합은_1이어야_한다(self) -> None:
        with pytest.raises(InvariantViolation, match="합이"):
            TrainingDataRef(
                dataset_ref="ds",
                uri="x.csv",
                feature_fields=("a",),
                label_field="y",
                split_ratio=(0.8, 0.15, 0.15),
            )
