"""Optimization Context 가 바깥 세계에 요구하는 것(Port).

이 파일에 torch 도 tensorflow 도 onnx 도 없다.

    Domain          ModelArtifact / BenchmarkResult / NumericalEquivalence
        ↓ Port
    Infrastructure  TorchScriptExporter / OnnxExporter / TFLiteExporter …

TFLite 지원이 사라지고 다른 것이 와도, 바뀌는 것은 어댑터 하나다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from domain.optimization.baseline_ref import BaselineModelRef
from domain.optimization.benchmark import BenchmarkResult, MeasurementProtocol
from domain.optimization.conversion import NumericalEquivalence
from domain.optimization.identifiers import OptimizationRunId
from domain.optimization.optimization_run import OptimizationRun
from domain.optimization.quantization import QuantizationComparison
from domain.optimization.resource import BatchScaling, ResourceUsage
from domain.optimization.roofline import DeviceCapability, RooflineProfile
from domain.optimization.runtime import ModelArtifact, Precision, RuntimeTarget
from domain.optimization.structural import StructuralOutcome, StructuralReduction
from domain.optimization.tradeoff import ArtifactAccuracy


@runtime_checkable
class OptimizationRunRepository(Protocol):
    def save(self, run: OptimizationRun) -> None: ...

    def find_by_id(self, run_id: OptimizationRunId) -> OptimizationRun | None: ...

    def exists(self, run_id: OptimizationRunId) -> bool: ...

    def list_all(self) -> Sequence[OptimizationRun]: ...


@runtime_checkable
class ModelExporter(Protocol):
    """모델을 다른 실행 경로/정밀도로 내보낸다. (실습 4-2 ~ 4-7)

    지원하지 않는 조합이면 예외를 던진다. 조용히 다른 것을 만들지 않는다.
    """

    def supports(self, runtime: RuntimeTarget, precision: Precision) -> bool: ...

    def export(
        self,
        baseline: BaselineModelRef,
        runtime: RuntimeTarget,
        precision: Precision,
    ) -> ModelArtifact: ...


@runtime_checkable
class EquivalenceChecker(Protocol):
    """변환 전후가 같은 답을 내는지 확인한다. (실습 4-2 ~ 4-4)"""

    def compare(
        self, baseline: BaselineModelRef, artifact: ModelArtifact, sample_count: int
    ) -> NumericalEquivalence: ...


@runtime_checkable
class RuntimeBenchmarker(Protocol):
    """지연시간을 잰다. **어떻게 쟀는지(protocol)를 함께 돌려준다.** (실습 4-1)"""

    def benchmark(
        self, artifact: ModelArtifact, protocol: MeasurementProtocol
    ) -> BenchmarkResult: ...


@runtime_checkable
class ArtifactAccuracyEvaluator(Protocol):
    """변환된 결과물을 같은 평가 집합으로 다시 잰다. (실습 4-7, 4-8)"""

    def evaluate(
        self, baseline: BaselineModelRef, artifact: ModelArtifact, split: str
    ) -> ArtifactAccuracy: ...


@runtime_checkable
class RooflineProfiler(Protocol):
    """층별 연산량과 데이터 이동량을 센다. (실습 4-9)"""

    def profile(
        self, baseline: BaselineModelRef, device: DeviceCapability
    ) -> RooflineProfile: ...


@runtime_checkable
class StructuralReducer(Protocol):
    """구조 자체를 줄인다. (실습 4-11)

    양자화(ModelExporter)와 다른 축이다.
    **여기서는 곱셈 횟수가 실제로 줄어들 수 있다.**
    """

    def reduce(
        self,
        baseline: BaselineModelRef,
        reduction: StructuralReduction,
        *,
        label: str,
        split: str = "test",
        fine_tune_epochs: int = 6,
    ) -> StructuralOutcome: ...


@runtime_checkable
class QuantizationLab(Protocol):
    """PTQ 와 QAT 를 같은 자리에서 잰다. (실습 4-12)"""

    def compare(
        self,
        baseline: BaselineModelRef,
        *,
        bits: int,
        split: str = "test",
        epochs: int = 12,
        per_channel: bool = True,
    ) -> QuantizationComparison: ...


@runtime_checkable
class ResourceMeter(Protocol):
    """추론 중 CPU·메모리를 **실제로** 잰다. (실습 4-13)"""

    def measure(
        self, artifact: ModelArtifact, protocol: MeasurementProtocol
    ) -> ResourceUsage: ...


@runtime_checkable
class BatchScalingMeter(Protocol):
    """배치 크기를 바꿔 가며 잰다. (실습 4-14)"""

    def scale(
        self,
        artifact: ModelArtifact,
        batch_sizes: tuple[int, ...],
        protocol: MeasurementProtocol,
    ) -> BatchScaling: ...
