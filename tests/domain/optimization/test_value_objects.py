"""Optimization Context 의 Value Object 와 Policy.

전부 순수 계산이다. 파일도, 모델도, 프레임워크도 필요 없다.
"""

from __future__ import annotations

import pytest

from domain.optimization.baseline_ref import BaselineModelRef
from domain.optimization.benchmark import BenchmarkResult, MeasurementProtocol
from domain.optimization.identifiers import ArtifactId
from domain.optimization.roofline import (
    BottleneckKind,
    DeviceCapability,
    LayerCost,
    RooflineProfile,
)
from domain.optimization.runtime import ModelArtifact, Precision, RuntimeTarget
from domain.optimization.selection import DeviceBudget
from domain.optimization.tradeoff import ArtifactAccuracy
from domain.shared.errors import InvariantViolation


class TestModelArtifact:
    def artifact(self, **overrides) -> ModelArtifact:  # noqa: ANN003
        base: dict[str, object] = dict(
            artifact_id=ArtifactId.of("a"),
            runtime=RuntimeTarget.TFLITE,
            precision=Precision.INT8,
            size_bytes=9_512,
            uri="mem://a",
            parameter_count=3_187,
        )
        base.update(overrides)
        return ModelArtifact(**base)  # type: ignore[arg-type]

    def test_이론_크기는_정밀도가_정한다(self) -> None:
        assert self.artifact().theoretical_weight_bytes == 3_187
        assert (
            self.artifact(precision=Precision.FP32).theoretical_weight_bytes == 12_748
        )

    def test_오버헤드는_실제에서_이론을_뺀_것이다(self) -> None:
        assert self.artifact().overhead_bytes == 9_512 - 3_187

    def test_오버헤드가_가중치보다_클_수_있다(self) -> None:
        """작은 모델에서는 흔하다. (실습 4-5)"""
        small = self.artifact()
        assert small.overhead_bytes > small.theoretical_weight_bytes

    def test_크기가_음수면_결과물이_아니다(self) -> None:
        with pytest.raises(InvariantViolation):
            self.artifact(size_bytes=-1)

    def test_위치가_없으면_결과물이_아니다(self) -> None:
        with pytest.raises(InvariantViolation):
            self.artifact(uri="  ")

    def test_라벨은_경로와_정밀도의_조합이다(self) -> None:
        assert self.artifact().label == "TFLITE/INT8"


class TestBenchmarkResult:
    def result(self, **overrides) -> BenchmarkResult:  # noqa: ANN003
        base: dict[str, object] = dict(
            protocol=MeasurementProtocol(),
            p50_ms=1.0,
            p95_ms=1.5,
            p99_ms=1.8,
            min_ms=0.9,
            max_ms=2.0,
        )
        base.update(overrides)
        return BenchmarkResult(**base)  # type: ignore[arg-type]

    def test_분위수가_순서대로가_아니면_측정이_잘못된_것이다(self) -> None:
        with pytest.raises(InvariantViolation):
            self.result(p95_ms=0.5)

    def test_지터는_p95와_p50의_비다(self) -> None:
        assert self.result().jitter_ratio == pytest.approx(1.5)

    def test_배속은_p95로_잰다(self) -> None:
        """최악을 기준으로 비교한다."""
        fast = self.result(p50_ms=0.4, p95_ms=0.5, p99_ms=0.6, min_ms=0.3, max_ms=0.7)
        assert fast.speedup_over(self.result()) == pytest.approx(3.0)

    def test_평균은_없다(self) -> None:
        """일부러 뺐다. 현장에서 중요한 것은 최악이다."""
        assert not hasattr(self.result(), "mean_ms")


class TestMeasurementProtocol:
    def test_측정_횟수는_1_이상이어야_한다(self) -> None:
        with pytest.raises(InvariantViolation):
            MeasurementProtocol(measured_runs=0)

    def test_스레드는_1_이상이어야_한다(self) -> None:
        with pytest.raises(InvariantViolation):
            MeasurementProtocol(threads=0)

    def test_어떻게_쟀는지를_말할_수_있어야_한다(self) -> None:
        assert "warmup=30" in MeasurementProtocol().describe()


class TestArtifactAccuracy:
    def test_범위를_벗어난_정확도는_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            ArtifactAccuracy(accuracy=1.2, macro_recall=0.9)

    def test_가장_크게_떨어진_클래스를_찾는다(self) -> None:
        baseline = ArtifactAccuracy(
            accuracy=0.97,
            macro_recall=0.95,
            per_class_recall={"FAULT": 0.90, "NORMAL": 0.99},
        )
        quantized = ArtifactAccuracy(
            accuracy=0.96,
            macro_recall=0.80,
            per_class_recall={"FAULT": 0.55, "NORMAL": 0.99},
        )
        assert quantized.worst_class_drop_from(baseline) == ("FAULT", pytest.approx(0.35))

    def test_사라진_클래스는_0으로_센다(self) -> None:
        baseline = ArtifactAccuracy(
            accuracy=0.97, macro_recall=0.95, per_class_recall={"FAULT": 0.90}
        )
        broken = ArtifactAccuracy(accuracy=0.9, macro_recall=0.5, per_class_recall={})
        assert broken.worst_class_drop_from(baseline) == ("FAULT", pytest.approx(0.90))


class TestDeviceBudget:
    def test_지연시간_예산은_0보다_커야_한다(self) -> None:
        with pytest.raises(InvariantViolation):
            DeviceBudget(name="설비", latency_p95_ms=0)

    def test_비율_항목은_0에서_1_사이다(self) -> None:
        with pytest.raises(InvariantViolation):
            DeviceBudget(name="설비", latency_p95_ms=30, max_accuracy_drop=1.5)


class TestRoofline:
    def device(self) -> DeviceCapability:
        return DeviceCapability(
            name="edge",
            peak_gmac_per_second=2.0,
            memory_bandwidth_gb_per_second=1.6,
        )

    def test_균형점은_연산성능을_대역폭으로_나눈_값이다(self) -> None:
        assert self.device().machine_balance == pytest.approx(1.25)

    def test_대역폭이_0이면_기계가_아니다(self) -> None:
        with pytest.raises(InvariantViolation):
            DeviceCapability(
                name="x", peak_gmac_per_second=1, memory_bandwidth_gb_per_second=0
            )

    def test_계산도_이동도_없는_층은_병목이_아니다(self) -> None:
        idle = LayerCost(
            name="reshape", kind="Reshape", mac_count=0, weight_bytes=0, activation_bytes=0
        )
        assert idle.bottleneck(1.25) is BottleneckKind.NEGLIGIBLE

    def test_균형점_근처는_어느_쪽도_아니다(self) -> None:
        balanced = LayerCost(
            name="conv", kind="Conv1d", mac_count=125, weight_bytes=50, activation_bytes=50
        )
        assert balanced.arithmetic_intensity == pytest.approx(1.25)
        assert balanced.bottleneck(1.25) is BottleneckKind.BALANCED

    def test_층이_없으면_병목도_없다(self) -> None:
        assert RooflineProfile().dominant_bottleneck is BottleneckKind.NEGLIGIBLE


class TestBaselineModelRef:
    def test_승인받지_않았으면_어떤_게이트가_남았는지_말한다(self) -> None:
        ref = BaselineModelRef(
            model_version_id="mv-1",
            run_ref="run-1",
            input_shape=(30, 6),
            class_labels=("A", "B"),
            parameter_count=100,
            mac_count=1_000,
            accuracy=0.9,
            macro_recall=0.9,
            accepted=False,
        )
        assert ref.missing_gates
        assert "모듈 3" in ref.missing_gates[0]
