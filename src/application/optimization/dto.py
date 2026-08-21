"""Optimization Use Case 결과 DTO."""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from domain.optimization.benchmark import BenchmarkResult
from domain.optimization.optimization_run import OptimizationRun
from domain.optimization.roofline import RooflineProfile
from domain.optimization.selection import OptimizationCertificate
from domain.optimization.tradeoff import OptimizationCandidate, TradeoffTable
from domain.optimization.structural import ReductionComparison
from domain.optimization.quantization import QuantizationComparison
from domain.optimization.resource import BatchScaling, ResourceUsage


@dataclass(frozen=True, slots=True)
class BenchmarkView:
    run_id: str
    label: str
    protocol: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    jitter_ratio: float
    size_bytes: int
    activation_bytes: int
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        run_id: str,
        label: str,
        size_bytes: int,
        result: BenchmarkResult,
        findings: tuple[FindingView, ...] = (),
    ) -> BenchmarkView:
        return cls(
            run_id=run_id,
            label=label,
            protocol=result.protocol.describe(),
            p50_ms=result.p50_ms,
            p95_ms=result.p95_ms,
            p99_ms=result.p99_ms,
            jitter_ratio=result.jitter_ratio,
            size_bytes=size_bytes,
            activation_bytes=result.activation_bytes,
            findings=findings,
        )

    def render(self) -> str:
        lines = [
            f"측정 ({self.label})",
            f"  프로토콜 : {self.protocol}",
            f"  p50 {self.p50_ms:.4f}ms / p95 {self.p95_ms:.4f}ms "
            f"/ p99 {self.p99_ms:.4f}ms / 지터 {self.jitter_ratio:.2f}배",
            f"  파일 {self.size_bytes:,}B / 활성값 추정 {self.activation_bytes:,}B",
        ]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class CandidateView:
    run_id: str
    label: str
    artifact_id: str
    runtime: str
    precision: str
    size_bytes: int
    theoretical_weight_bytes: int
    overhead_bytes: int
    p50_ms: float
    p95_ms: float
    accuracy: float
    macro_recall: float
    macro_f1: float
    per_class_recall: dict[str, float]
    equivalence: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        run_id: str,
        candidate: OptimizationCandidate,
        findings: tuple[FindingView, ...] = (),
    ) -> CandidateView:
        artifact = candidate.artifact
        return cls(
            run_id=run_id,
            label=candidate.label,
            artifact_id=str(artifact.artifact_id),
            runtime=artifact.runtime.value,
            precision=artifact.precision.value,
            size_bytes=artifact.size_bytes,
            theoretical_weight_bytes=artifact.theoretical_weight_bytes,
            overhead_bytes=artifact.overhead_bytes,
            p50_ms=candidate.benchmark.p50_ms,
            p95_ms=candidate.benchmark.p95_ms,
            accuracy=candidate.accuracy.accuracy,
            macro_recall=candidate.accuracy.macro_recall,
            macro_f1=candidate.accuracy.macro_f1,
            per_class_recall=dict(candidate.accuracy.per_class_recall),
            equivalence=candidate.conversion.describe(),
            findings=findings,
        )

    def render(self) -> str:
        lines = [
            f"후보 {self.label}",
            f"  변환   : {self.equivalence}",
            f"  크기   : {self.size_bytes:,}B "
            f"(가중치 {self.theoretical_weight_bytes:,} + 오버헤드 {self.overhead_bytes:,})",
            f"  속도   : p50 {self.p50_ms:.4f}ms / p95 {self.p95_ms:.4f}ms",
            f"  정확도 : {self.accuracy:.4f} / macro recall {self.macro_recall:.4f}"
            f" / macro F1 {self.macro_f1:.4f}",
        ]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RooflineView:
    run_id: str
    device: str
    total_macs: int
    total_bytes_moved: int
    overall_intensity: float
    machine_balance: float
    dominant_bottleneck: str
    busiest_layer: str | None
    heaviest_traffic_layer: str | None
    table: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        run_id: str,
        profile: RooflineProfile,
        findings: tuple[FindingView, ...] = (),
    ) -> RooflineView:
        busiest = profile.busiest_layer
        traffic = profile.heaviest_traffic_layer
        return cls(
            run_id=run_id,
            device=profile.device.describe() if profile.device else "",
            total_macs=profile.total_macs,
            total_bytes_moved=profile.total_bytes_moved,
            overall_intensity=profile.overall_intensity,
            machine_balance=profile.machine_balance,
            dominant_bottleneck=profile.dominant_bottleneck.value,
            busiest_layer=busiest.name if busiest else None,
            heaviest_traffic_layer=traffic.name if traffic else None,
            table=profile.render(),
            findings=findings,
        )

    def render(self) -> str:
        lines = [self.table]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TradeoffView:
    run_id: str
    table: str
    fastest: str
    smallest: str
    most_accurate: str
    slower_than_baseline: tuple[str, ...]
    pareto_front: tuple[str, ...]

    @classmethod
    def of(cls, run_id: str, table: TradeoffTable) -> TradeoffView:
        return cls(
            run_id=run_id,
            table=table.render(),
            fastest=table.fastest.label,
            smallest=table.smallest.label,
            most_accurate=table.most_accurate.label,
            slower_than_baseline=tuple(
                c.label for c in table.slower_than_baseline
            ),
            pareto_front=tuple(c.label for c in table.pareto_front),
        )

    def render(self) -> str:
        lines = [
            self.table,
            "",
            f"  가장 빠름   : {self.fastest}",
            f"  가장 작음   : {self.smallest}",
            f"  가장 정확함 : {self.most_accurate}",
            f"  파레토 전선 : {', '.join(self.pareto_front)}",
        ]
        if self.slower_than_baseline:
            lines.append(
                f"  기준보다 느려진 후보: {', '.join(self.slower_than_baseline)}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SelectionView:
    run_id: str
    verdict: str
    has_selection: bool
    selected_label: str | None
    selected_p95_ms: float | None
    selected_accuracy: float | None
    budget: str
    objective: str
    rejected: tuple[tuple[str, tuple[str, ...]], ...]
    _rendered: str = ""

    @classmethod
    def of(cls, run_id: str, certificate: OptimizationCertificate) -> SelectionView:
        return cls(
            run_id=run_id,
            verdict=certificate.verdict.value,
            has_selection=certificate.has_selection,
            selected_label=certificate.selected_label,
            selected_p95_ms=certificate.selected_p95_ms,
            selected_accuracy=certificate.selected_accuracy,
            budget=certificate.budget.describe(),
            objective=certificate.objective.value,
            rejected=tuple(
                (v.label, v.reasons) for v in certificate.verdicts if not v.accepted
            ),
            _rendered=certificate.render(),
        )

    def render(self) -> str:
        return self._rendered


@dataclass(frozen=True, slots=True)
class OptimizationRunView:
    run_id: str
    model_version_id: str
    status: str
    baseline_label: str | None
    candidate_labels: tuple[str, ...]
    rejections: tuple[tuple[str, str], ...]
    selected_label: str | None
    verdict: str | None

    @classmethod
    def of(cls, run: OptimizationRun) -> OptimizationRunView:
        certificate = run.certificate
        baseline = run.baseline_candidate
        return cls(
            run_id=str(run.id),
            model_version_id=run.baseline.model_version_id,
            status=run.status.value,
            baseline_label=baseline.label if baseline else None,
            candidate_labels=tuple(c.label for c in run.candidates),
            rejections=run.rejections,
            selected_label=certificate.selected_label if certificate else None,
            verdict=certificate.verdict.value if certificate else None,
        )


@dataclass(frozen=True, slots=True)
class ReductionOutcomeView:
    label: str
    reduction: str
    parameter_count_before: int
    parameter_count_after: int
    nonzero_parameter_count: int
    sparsity: float
    mac_count_before: int
    mac_count_after: int
    mac_reduction: float
    size_bytes_before: int
    size_bytes_after: int
    size_reduction: float
    accuracy_before: float
    accuracy_after: float
    accuracy_drop: float
    verdict: str


@dataclass(frozen=True, slots=True)
class ReductionComparisonView:
    """구조 축소 비교표. (실습 4-11)"""

    run_id: str
    outcomes: tuple[ReductionOutcomeView, ...]
    usable: tuple[str, ...]
    table: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        run_id: str,
        comparison: ReductionComparison,
        findings: tuple[FindingView, ...] = (),
    ) -> ReductionComparisonView:
        return cls(
            run_id=run_id,
            outcomes=tuple(
                ReductionOutcomeView(
                    label=outcome.label,
                    reduction=outcome.reduction.describe(),
                    parameter_count_before=outcome.parameter_count_before,
                    parameter_count_after=outcome.parameter_count_after,
                    nonzero_parameter_count=outcome.nonzero_parameter_count,
                    sparsity=outcome.sparsity,
                    mac_count_before=outcome.mac_count_before,
                    mac_count_after=outcome.mac_count_after,
                    mac_reduction=outcome.mac_reduction,
                    size_bytes_before=outcome.size_bytes_before,
                    size_bytes_after=outcome.size_bytes_after,
                    size_reduction=outcome.size_reduction,
                    accuracy_before=outcome.accuracy_before,
                    accuracy_after=outcome.accuracy_after,
                    accuracy_drop=outcome.accuracy_drop,
                    verdict=(
                        "BLOCKED"
                        if any(f.is_blocking for f in row_findings)
                        else ("WARNED" if row_findings else "OK")
                    ),
                )
                for outcome, row_findings in comparison.rows
            ),
            usable=comparison.usable,
            table=comparison.render(),
            findings=findings,
        )

    def outcome_of(self, label: str) -> ReductionOutcomeView | None:
        return next((o for o in self.outcomes if o.label == label), None)

    def render(self) -> str:
        return f"구조 축소 비교 ({self.run_id})\n{self.table}"


@dataclass(frozen=True, slots=True)
class QuantizationOutcomeView:
    label: str
    approach: str
    bits: int
    baseline_accuracy: float
    quantized_accuracy: float
    accuracy_drop: float
    quantized_macro_recall: float
    training_seconds: float
    weight_bytes: int


@dataclass(frozen=True, slots=True)
class QuantizationComparisonView:
    """PTQ 대 QAT. (실습 4-12)"""

    run_id: str
    bits: int
    post_training: QuantizationOutcomeView
    quantization_aware: QuantizationOutcomeView
    recovered: float
    extra_training_seconds: float
    table: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        run_id: str,
        comparison: QuantizationComparison,
        findings: tuple[FindingView, ...] = (),
    ) -> QuantizationComparisonView:
        def view(outcome) -> QuantizationOutcomeView:  # noqa: ANN001
            return QuantizationOutcomeView(
                label=outcome.label,
                approach=outcome.spec.approach.value,
                bits=outcome.spec.bits,
                baseline_accuracy=outcome.baseline_accuracy,
                quantized_accuracy=outcome.quantized_accuracy,
                accuracy_drop=outcome.accuracy_drop,
                quantized_macro_recall=outcome.quantized_macro_recall,
                training_seconds=outcome.training_seconds,
                weight_bytes=outcome.weight_bytes,
            )

        return cls(
            run_id=run_id,
            bits=comparison.bits,
            post_training=view(comparison.post_training),
            quantization_aware=view(comparison.quantization_aware),
            recovered=comparison.recovered,
            extra_training_seconds=comparison.extra_training_seconds,
            table=comparison.render(),
            findings=findings,
        )

    def render(self) -> str:
        lines = [self.table]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ResourceUsageView:
    """추론 중 자원 사용. (실습 4-13)"""

    label: str
    baseline_rss_bytes: int
    peak_rss_bytes: int
    model_rss_bytes: int
    cpu_time_ms: float
    wall_time_ms: float
    cpu_utilization: float
    artifact_bytes: int
    rss_to_artifact_ratio: float
    verdict: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls, usage: ResourceUsage, findings: tuple[FindingView, ...] = ()
    ) -> ResourceUsageView:
        blocking = any(f.severity == "CRITICAL" for f in findings)
        return cls(
            label=usage.label,
            baseline_rss_bytes=usage.baseline_rss_bytes,
            peak_rss_bytes=usage.peak_rss_bytes,
            model_rss_bytes=usage.model_rss_bytes,
            cpu_time_ms=usage.cpu_time_ms,
            wall_time_ms=usage.wall_time_ms,
            cpu_utilization=usage.cpu_utilization,
            artifact_bytes=usage.artifact_bytes,
            rss_to_artifact_ratio=usage.rss_to_artifact_ratio,
            verdict=(
                "BLOCKED" if blocking else ("WARNED" if findings else "OK")
            ),
            findings=findings,
        )

    def render(self) -> str:
        lines = [
            f"{self.label}",
            f"  실행 중 메모리  {self.peak_rss_bytes / 1024 / 1024:>8.1f} MiB "
            f"(모델 몫 {self.model_rss_bytes / 1024 / 1024:.1f} MiB)",
            f"  모델 파일       {self.artifact_bytes / 1024:>8.1f} KiB "
            f"({self.rss_to_artifact_ratio:,.0f}배)",
            f"  CPU             {self.cpu_utilization:>8.2f} 코어",
        ]
        lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class BatchScalingView:
    """배치 크기별 지연시간. (실습 4-14)"""

    label: str
    cycle_time_ms: float
    batch_sizes: tuple[int, ...]
    per_sample_ms: tuple[float, ...]
    first_answer_ms: tuple[float, ...]
    throughput_gain: float
    latency_cost: float
    table: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        label: str,
        scaling: BatchScaling,
        *,
        cycle_time_ms: float,
        findings: tuple[FindingView, ...] = (),
    ) -> BatchScalingView:
        return cls(
            label=label,
            cycle_time_ms=cycle_time_ms,
            batch_sizes=tuple(p.batch_size for p in scaling.points),
            per_sample_ms=tuple(p.per_sample_ms for p in scaling.points),
            first_answer_ms=tuple(p.first_answer_ms for p in scaling.points),
            throughput_gain=scaling.throughput_gain,
            latency_cost=scaling.latency_cost,
            table=scaling.render(),
            findings=findings,
        )

    def at(self, batch_size: int) -> tuple[float, float] | None:
        for size, per_sample, first in zip(
            self.batch_sizes, self.per_sample_ms, self.first_answer_ms, strict=True
        ):
            if size == batch_size:
                return per_sample, first
        return None

    def render(self) -> str:
        lines = [f"[{self.label}] 사이클 타임 {self.cycle_time_ms:g}ms", self.table]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)
