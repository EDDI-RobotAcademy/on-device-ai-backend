"""ConvertModel — 한 번의 변환을 끝까지 밀어붙인다. (실습 4-2 ~ 4-7)

한 후보를 만드는 데 필요한 것은 파일 하나가 아니다.

    1. 내보낸다        → ModelArtifact (크기는 파일에서 잰다)
    2. 대조한다        → NumericalEquivalence (같은 답을 내는가)
    3. 잰다            → BenchmarkResult (얼마나 빠른가)
    4. 다시 평가한다   → ArtifactAccuracy (무엇을 잃었는가)

넷 중 하나라도 빠지면 그 후보는 비교표에 올릴 수 없다.
**'변환 성공'만으로 후보가 되지 않는다.**
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.optimization.dto import CandidateView
from application.optimization.support import commit, load_run
from application.shared.errors import UnsupportedOperation
from application.shared.ports import EventPublisher
from domain.optimization.benchmark import BenchmarkPolicy, MeasurementProtocol
from domain.optimization.conversion import ConversionRecord, EquivalencePolicy
from domain.optimization.errors import ConversionFailed
from domain.optimization.ports import (
    ArtifactAccuracyEvaluator,
    EquivalenceChecker,
    ModelExporter,
    OptimizationRunRepository,
    RuntimeBenchmarker,
)
from domain.optimization.runtime import Precision, RuntimeTarget


@dataclass(frozen=True, slots=True)
class ConvertModelCommand:
    run_id: str
    runtime: RuntimeTarget
    precision: Precision = Precision.FP32
    equivalence_samples: int = 128
    split: str = "test"
    protocol: MeasurementProtocol = field(default_factory=MeasurementProtocol)
    equivalence_policy: EquivalencePolicy = field(default_factory=EquivalencePolicy)
    benchmark_policy: BenchmarkPolicy = field(default_factory=BenchmarkPolicy)


class ConvertModel:
    def __init__(
        self,
        runs: OptimizationRunRepository,
        exporter: ModelExporter,
        checker: EquivalenceChecker,
        benchmarker: RuntimeBenchmarker,
        accuracy: ArtifactAccuracyEvaluator,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._exporter = exporter
        self._checker = checker
        self._benchmarker = benchmarker
        self._accuracy = accuracy
        self._publisher = publisher

    def execute(self, command: ConvertModelCommand) -> CandidateView:
        run = load_run(self._runs, command.run_id)
        label = f"{command.runtime.value}/{command.precision.value}"

        if not self._exporter.supports(command.runtime, command.precision):
            raise UnsupportedOperation(
                f"{label} 조합을 만들 수 있는 어댑터가 없다.", subject=label
            )

        try:
            artifact = self._exporter.export(
                run.baseline, command.runtime, command.precision
            )
        except ConversionFailed as exc:
            # 실패도 결과다. 남기지 않으면 다음 사람이 똑같이 하루를 쓴다.
            run.record_rejection(label, str(exc))
            commit(self._runs, run, self._publisher)
            raise

        equivalence = self._checker.compare(
            run.baseline, artifact, command.equivalence_samples
        )
        conversion = ConversionRecord(
            source_runtime=RuntimeTarget.PYTORCH,
            target_runtime=command.runtime,
            precision=command.precision,
            equivalence=equivalence,
        )
        benchmark = self._benchmarker.benchmark(artifact, command.protocol)
        measured = self._accuracy.evaluate(run.baseline, artifact, command.split)

        from domain.optimization.tradeoff import OptimizationCandidate

        candidate = OptimizationCandidate(
            artifact=artifact,
            conversion=conversion,
            benchmark=benchmark,
            accuracy=measured,
        )
        run.add_candidate(candidate)
        commit(self._runs, run, self._publisher)

        findings = (
            *command.equivalence_policy.inspect(conversion),
            *command.benchmark_policy.inspect(benchmark),
        )
        return CandidateView.of(
            str(run.id), candidate, tuple(FindingView.of(f) for f in findings)
        )
