"""BenchmarkBaseline — PC에서 돌아간다고 끝난 게 아니다. (실습 4-1)

기준 모델을 **프로토콜을 명시해서** 잰다.
같은 모델을 다르게 재면 다른 숫자가 나온다. 그래서 프로토콜이 결과에 붙어 다닌다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.optimization.dto import BenchmarkView
from application.optimization.support import commit, load_run
from application.shared.ports import EventPublisher
from domain.optimization.benchmark import BenchmarkPolicy, MeasurementProtocol
from domain.optimization.ports import (
    ArtifactAccuracyEvaluator,
    ModelExporter,
    OptimizationRunRepository,
    RuntimeBenchmarker,
)
from domain.optimization.runtime import Precision, RuntimeTarget


@dataclass(frozen=True, slots=True)
class BenchmarkBaselineCommand:
    run_id: str
    protocol: MeasurementProtocol = field(default_factory=MeasurementProtocol)
    policy: BenchmarkPolicy = field(default_factory=BenchmarkPolicy)
    split: str = "test"


class BenchmarkBaseline:
    """기준 모델을 내보내고, 재고, 정확도까지 같은 방식으로 다시 잰다.

    정확도를 여기서 **다시** 재는 이유:
    모듈 3 의 숫자와 이 숫자가 다르면 그 자체가 정보다.
    같은 척도로 재야 후보와 비교할 수 있다.
    """

    def __init__(
        self,
        runs: OptimizationRunRepository,
        exporter: ModelExporter,
        benchmarker: RuntimeBenchmarker,
        accuracy: ArtifactAccuracyEvaluator,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._exporter = exporter
        self._benchmarker = benchmarker
        self._accuracy = accuracy
        self._publisher = publisher

    def execute(self, command: BenchmarkBaselineCommand) -> BenchmarkView:
        run = load_run(self._runs, command.run_id)

        artifact = self._exporter.export(
            run.baseline, RuntimeTarget.PYTORCH, Precision.FP32
        )
        result = self._benchmarker.benchmark(artifact, command.protocol)
        accuracy = self._accuracy.evaluate(run.baseline, artifact, command.split)

        run.record_baseline(artifact, result, accuracy)
        commit(self._runs, run, self._publisher)

        findings = command.policy.inspect(result)
        return BenchmarkView.of(
            str(run.id),
            artifact.label,
            artifact.size_bytes,
            result,
            tuple(FindingView.of(f) for f in findings),
        )
