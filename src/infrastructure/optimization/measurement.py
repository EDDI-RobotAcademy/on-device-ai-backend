"""측정 어댑터 — 지연시간, 동등성, 정확도, 병목. (실습 4-1, 4-3, 4-7, 4-9)

전부 `RuntimeRegistry` 에 등록된 `predict` 함수 하나만 쓴다.
그래서 이 파일에는 런타임별 분기가 없다.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from domain.model.identifiers import ModelVersionId
from domain.optimization.baseline_ref import BaselineModelRef
from domain.optimization.benchmark import BenchmarkResult, MeasurementProtocol
from domain.optimization.conversion import NumericalEquivalence
from domain.optimization.roofline import (
    DeviceCapability,
    LayerCost,
    RooflineProfile,
)
from domain.optimization.runtime import ModelArtifact
from domain.optimization.tradeoff import ArtifactAccuracy
from infrastructure.ml.torch_trainer import TorchModelRegistry
from infrastructure.optimization.runtime_registry import RuntimeRegistry


class RuntimeLatencyBenchmarker:
    """domain.optimization.ports.RuntimeBenchmarker 구현. (실습 4-1)

    측정 순서가 결과를 바꾼다.
        1. 스레드 수를 프로토콜대로 고정한다
        2. 워밍업을 돌린다 — 첫 호출은 항상 느리다
        3. 그 다음부터 잰다
    """

    def __init__(self, runtimes: RuntimeRegistry) -> None:
        self._runtimes = runtimes

    def benchmark(
        self, artifact: ModelArtifact, protocol: MeasurementProtocol
    ) -> BenchmarkResult:
        loaded = self._runtimes.require(artifact.artifact_id)
        torch.set_num_threads(protocol.threads)

        shape = (protocol.batch_size, *loaded.input_shape)
        sample = np.zeros(shape, dtype="float32")

        for _ in range(protocol.warmup_runs):
            loaded.predict(sample)

        timings = np.empty(protocol.measured_runs, dtype="float64")
        for index in range(protocol.measured_runs):
            started = time.perf_counter()
            loaded.predict(sample)
            timings[index] = (time.perf_counter() - started) * 1000.0

        return BenchmarkResult(
            protocol=protocol,
            p50_ms=float(np.percentile(timings, 50)),
            p95_ms=float(np.percentile(timings, 95)),
            p99_ms=float(np.percentile(timings, 99)),
            min_ms=float(timings.min()),
            max_ms=float(timings.max()),
            activation_bytes=loaded.activation_bytes * protocol.batch_size,
        )


class OutputEquivalenceChecker:
    """domain.optimization.ports.EquivalenceChecker 구현. (실습 4-2 ~ 4-4)

    같은 입력을 기준 모델과 변환 결과에 넣고 출력을 맞대어 본다.
    입력은 **실제 평가 데이터**를 쓴다 — 0 으로 채운 텐서는 아무것도 검증하지 못한다.
    """

    def __init__(
        self, models: TorchModelRegistry, runtimes: RuntimeRegistry
    ) -> None:
        self._models = models
        self._runtimes = runtimes

    def compare(
        self, baseline: BaselineModelRef, artifact: ModelArtifact, sample_count: int
    ) -> NumericalEquivalence:
        trained = self._models.get(ModelVersionId.of(baseline.model_version_id))
        if trained is None:
            raise KeyError(f"학습된 모델이 없다: {baseline.model_version_id}")

        features = trained.dataset.features["test"][:sample_count]
        if len(features) == 0:
            raise ValueError("대조할 표본이 없다.")

        module = trained.module
        module.load_state_dict(trained.best_state)
        module.eval()
        with torch.no_grad():
            reference = module(torch.from_numpy(features)).numpy()

        produced = self._runtimes.require(artifact.artifact_id).predict(features)

        difference = np.abs(reference - produced)
        agreement = float(
            (reference.argmax(axis=1) == produced.argmax(axis=1)).mean()
        )
        return NumericalEquivalence(
            sample_count=int(len(features)),
            max_abs_diff=float(difference.max()),
            mean_abs_diff=float(difference.mean()),
            agreement_ratio=agreement,
        )


class ArtifactAccuracyMeter:
    """domain.optimization.ports.ArtifactAccuracyEvaluator 구현. (실습 4-7, 4-8)

    변환된 결과물을 **같은 평가 집합**으로 다시 잰다.
    모듈 3 에서 쓴 그 분할이다. 다른 데이터로 재면 비교가 성립하지 않는다.
    """

    def __init__(
        self, models: TorchModelRegistry, runtimes: RuntimeRegistry
    ) -> None:
        self._models = models
        self._runtimes = runtimes

    def evaluate(
        self, baseline: BaselineModelRef, artifact: ModelArtifact, split: str
    ) -> ArtifactAccuracy:
        trained = self._models.get(ModelVersionId.of(baseline.model_version_id))
        if trained is None:
            raise KeyError(f"학습된 모델이 없다: {baseline.model_version_id}")

        features = trained.dataset.features[split]
        targets = trained.dataset.targets[split]
        predictions = self._runtimes.require(artifact.artifact_id).predict(
            features
        ).argmax(axis=1)

        labels = trained.dataset.labels
        per_class: dict[str, float] = {}
        recalls: list[float] = []
        f1s: list[float] = []
        for index, label in enumerate(labels):
            support = int((targets == index).sum())
            if support == 0:
                per_class[label] = 0.0
                continue
            hits = int(((targets == index) & (predictions == index)).sum())
            recall = hits / support
            predicted = int((predictions == index).sum())
            precision = hits / predicted if predicted else 0.0
            per_class[label] = float(recall)
            recalls.append(float(recall))
            # 재현율만 보면 '정상을 이상이라 부르는' 실패가 안 보인다 (실습 4-8).
            f1s.append(
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )

        total = len(predictions)
        return ArtifactAccuracy(
            accuracy=float((predictions == targets).mean()),
            macro_recall=float(np.mean(recalls)) if recalls else 0.0,
            macro_f1=float(np.mean(f1s)) if f1s else 0.0,
            per_class_recall=per_class,
            # 모듈 5 가 현장 분포를 여기에 견준다 (실습 5-6).
            predicted_mix={
                label: float((predictions == index).sum() / total) if total else 0.0
                for index, label in enumerate(labels)
            },
        )


class TorchRooflineProfiler:
    """domain.optimization.ports.RooflineProfiler 구현. (실습 4-9)

    층마다 두 가지를 센다.
        MAC 수        — 계산량 (모듈 3 의 ArchitectureProfiler 가 이미 센 값)
        옮긴 바이트   — 가중치 + 입출력 활성값

    둘의 비가 산술 강도다. 그 값으로 병목이 어디인지 갈린다.
    """

    def __init__(self, models: TorchModelRegistry) -> None:
        self._models = models

    def profile(
        self, baseline: BaselineModelRef, device: DeviceCapability
    ) -> RooflineProfile:
        from infrastructure.ml.torch_architecture import TorchArchitectureProfiler

        trained = self._models.get(ModelVersionId.of(baseline.model_version_id))
        if trained is None:
            raise KeyError(f"학습된 모델이 없다: {baseline.model_version_id}")

        profile = TorchArchitectureProfiler().profile(trained.architecture)
        costs: list[LayerCost] = []
        previous = _elements(trained.architecture.input_spec.shape)

        for layer in profile.layers:
            current = _elements(layer.output_shape)
            costs.append(
                LayerCost(
                    name=layer.name,
                    kind=layer.kind,
                    mac_count=layer.mac_count,
                    weight_bytes=layer.parameter_count * 4,
                    activation_bytes=(previous + current) * 4,
                )
            )
            previous = current

        return RooflineProfile(layers=tuple(costs), device=device)


def _elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
