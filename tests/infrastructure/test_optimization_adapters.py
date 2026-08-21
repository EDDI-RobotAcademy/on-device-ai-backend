"""변환·측정 어댑터가 실제로 무엇을 하는지.

여기서는 진짜 파일이 만들어지고 진짜 런타임이 돈다.
Domain 테스트가 0.02초에 끝나는 것과 대비된다 — **그래서 층을 나눈다.**
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.optimization.benchmark import MeasurementProtocol
from domain.optimization.identifiers import ArtifactId, OptimizationRunId
from domain.optimization.roofline import BottleneckKind, DeviceCapability
from domain.optimization.runtime import Precision, RuntimeTarget
from infrastructure.optimization.runtime_registry import LoadedRuntime, RuntimeRegistry

FAST = MeasurementProtocol(warmup_runs=3, measured_runs=20)


@pytest.fixture
def wired(optimized):  # noqa: ANN001, ANN201
    """세션 파이프라인이 이미 만들어 둔 어댑터와 결과물을 그대로 쓴다."""
    optimization = optimized.optimization
    run = optimization.runs.find_by_id(OptimizationRunId.of(optimized.run_id))
    return optimization, run.baseline


class TestRuntimeRegistry:
    def test_없는_결과물은_없다고_말한다(self) -> None:
        registry = RuntimeRegistry()
        with pytest.raises(KeyError):
            registry.require(ArtifactId.of("없음"))

    def test_넣은_것을_꺼낼_수_있다(self) -> None:
        registry = RuntimeRegistry()
        loaded = LoadedRuntime(
            predict=lambda x: x, input_shape=(30, 6), activation_bytes=100
        )
        registry.put(ArtifactId.of("a"), loaded)
        assert registry.require(ArtifactId.of("a")) is loaded
        registry.clear()
        assert registry.get(ArtifactId.of("a")) is None


class TestExporters:
    def test_지원하는_조합만_맡는다(self, wired) -> None:
        optimization, _ = wired
        exporter = optimization.exporter

        assert exporter.supports(RuntimeTarget.ONNX, Precision.FP32)
        assert exporter.supports(RuntimeTarget.TFLITE, Precision.INT8)
        assert not exporter.supports(RuntimeTarget.ONNX, Precision.INT8)

    def test_크기는_파일에서_잰다(self, wired) -> None:
        from pathlib import Path

        optimization, baseline = wired
        artifact = optimization.exporter.export(
            baseline, RuntimeTarget.ONNX, Precision.FP32
        )
        assert Path(artifact.uri).exists()
        assert artifact.size_bytes == Path(artifact.uri).stat().st_size

    def test_내보낸_것은_바로_실행할_수_있다(self, wired) -> None:
        optimization, baseline = wired
        artifact = optimization.exporter.export(
            baseline, RuntimeTarget.TORCHSCRIPT, Precision.FP32
        )
        loaded = optimization.runtimes.require(artifact.artifact_id)

        output = loaded.predict(
            np.zeros((1, *loaded.input_shape), dtype="float32")
        )
        assert output.shape == (1, len(baseline.class_labels))

    def test_활성값은_배치와_함께_커진다(self, wired) -> None:
        optimization, baseline = wired
        artifact = optimization.exporter.export(
            baseline, RuntimeTarget.ONNX, Precision.FP32
        )
        one = optimization.benchmarker.benchmark(artifact, FAST)
        four = optimization.benchmarker.benchmark(
            artifact, MeasurementProtocol(warmup_runs=3, measured_runs=20, batch_size=4)
        )
        assert four.activation_bytes == one.activation_bytes * 4


class TestBenchmarker:
    def test_분위수가_순서대로_나온다(self, wired) -> None:
        optimization, baseline = wired
        artifact = optimization.exporter.export(
            baseline, RuntimeTarget.ONNX, Precision.FP32
        )
        result = optimization.benchmarker.benchmark(artifact, FAST)

        assert result.min_ms <= result.p50_ms <= result.p95_ms <= result.p99_ms
        assert result.p99_ms <= result.max_ms
        assert result.protocol is FAST


class TestEquivalenceChecker:
    def test_0이_아닌_실제_데이터로_비교한다(self, wired) -> None:
        """0으로 채운 텐서는 아무것도 검증하지 못한다."""
        optimization, baseline = wired
        artifact = optimization.exporter.export(
            baseline, RuntimeTarget.ONNX, Precision.FP32
        )
        equivalence = optimization.checker.compare(baseline, artifact, 40)

        assert equivalence.sample_count == 40
        assert equivalence.agreement_ratio == 1.0

    def test_학습된_모델이_없으면_비교할_수_없다(self, wired) -> None:
        from dataclasses import replace

        optimization, baseline = wired
        artifact = optimization.exporter.export(
            baseline, RuntimeTarget.ONNX, Precision.FP32
        )
        with pytest.raises(KeyError):
            optimization.checker.compare(
                replace(baseline, model_version_id="없는-모델"), artifact, 8
            )


class TestAccuracyMeter:
    def test_같은_분할로_다시_잰다(self, wired) -> None:
        optimization, baseline = wired
        artifact = optimization.exporter.export(
            baseline, RuntimeTarget.ONNX, Precision.FP32
        )
        measured = optimization.accuracy.evaluate(baseline, artifact, "test")

        assert set(measured.per_class_recall) == set(baseline.class_labels)
        assert 0.0 <= measured.macro_recall <= 1.0


class TestRooflineProfiler:
    def test_층별_MAC과_바이트를_센다(self, wired) -> None:
        optimization, baseline = wired
        profile = optimization.roofline.profile(
            baseline,
            DeviceCapability(
                name="edge",
                peak_gmac_per_second=2.0,
                memory_bandwidth_gb_per_second=1.6,
            ),
        )
        assert profile.total_macs == baseline.mac_count
        assert profile.total_bytes_moved > 0
        assert profile.dominant_bottleneck in tuple(BottleneckKind)

    def test_가중치_바이트는_FP32_기준이다(self, wired) -> None:
        optimization, baseline = wired
        profile = optimization.roofline.profile(
            baseline,
            DeviceCapability(
                name="edge",
                peak_gmac_per_second=2.0,
                memory_bandwidth_gb_per_second=1.6,
            ),
        )
        total_weight = sum(layer.weight_bytes for layer in profile.layers)
        assert total_weight == baseline.parameter_count * 4


class TestKerasBridge:
    """PyTorch → Keras 가중치 이식. **축 순서를 틀리면 조용히 다른 모델이 된다.**"""

    def transplanted(self, architecture):  # noqa: ANN001, ANN201
        """같은 명세로 두 프레임워크의 모델을 만들고 같은 입력을 넣는다."""
        import torch

        from infrastructure.ml.torch_architecture import build_module
        from infrastructure.optimization.keras_bridge import build_keras_model

        torch.manual_seed(7)
        module = build_module(architecture)
        module.eval()

        keras_model = build_keras_model(architecture, module.state_dict())

        sample = np.random.default_rng(7).normal(
            size=(4, *architecture.input_spec.shape)
        ).astype("float32")
        with torch.no_grad():
            expected = module(torch.from_numpy(sample)).numpy()
        produced = np.asarray(keras_model(sample, training=False))
        return expected, produced

    def architecture(self, kind):  # noqa: ANN001, ANN201
        from domain.model.architecture import ArchitectureKind, ModelArchitecture
        from domain.model.tensor_spec import TensorLayout, TensorSpec

        return ModelArchitecture(
            kind=kind,
            input_spec=TensorSpec(shape=(30, 6), layout=TensorLayout.TIME_FIRST),
            class_count=3,
            hidden_channels=(16, 32),
            kernel_size=5,
            dropout=0.1,
        )

    def test_CNN1D_는_같은_답을_낸다(self) -> None:
        from domain.model.architecture import ArchitectureKind

        expected, produced = self.transplanted(
            self.architecture(ArchitectureKind.CNN1D)
        )
        assert np.allclose(expected, produced, atol=1e-4)

    def test_MLP_도_같은_답을_낸다(self) -> None:
        """Flatten → Linear 경로. state_dict 의 층 번호 계산이 다르다."""
        from domain.model.architecture import ArchitectureKind

        expected, produced = self.transplanted(self.architecture(ArchitectureKind.MLP))
        assert np.allclose(expected, produced, atol=1e-4)

    def test_은닉층_수가_달라도_옮겨진다(self) -> None:
        """층 번호를 하드코딩하지 않았는지 확인한다."""
        from dataclasses import replace

        from domain.model.architecture import ArchitectureKind

        for hidden in ((8,), (8, 16, 32)):
            for kind in (ArchitectureKind.CNN1D, ArchitectureKind.MLP):
                architecture = replace(
                    self.architecture(kind), hidden_channels=hidden
                )
                expected, produced = self.transplanted(architecture)
                assert np.allclose(expected, produced, atol=1e-4), (
                    f"{kind.value} hidden={hidden}"
                )
