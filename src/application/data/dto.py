"""Use Case 결과 DTO.

Domain 객체를 그대로 밖으로 내보내지 않는다. (CLAUDE.md §10)
Interface Layer 는 이 DTO 만 본다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.data.dataset import Dataset
from domain.data.inspection import Finding, InspectionKind, InspectionReport, Verdict
from domain.data.profile import DatasetProfile
from domain.data.readiness import ReadinessCertificate
from domain.data.sampling_design import SamplingTradeoff


@dataclass(frozen=True, slots=True)
class FindingView:
    code: str
    message: str
    severity: str
    subject: str | None
    measured: float | None
    threshold: float | None

    @classmethod
    def of(cls, finding: Finding) -> FindingView:
        return cls(
            code=finding.code,
            message=finding.message,
            severity=finding.severity.value,
            subject=finding.subject,
            measured=finding.measured,
            threshold=finding.threshold,
        )

    def describe(self) -> str:
        parts = [f"{self.severity} {self.code}"]
        if self.subject:
            parts.append(f"({self.subject})")
        parts.append(self.message)
        if self.measured is not None and self.threshold is not None:
            parts.append(f"[측정 {self.measured:.4g} / 기준 {self.threshold:.4g}]")
        elif self.measured is not None:
            parts.append(f"[측정 {self.measured:.4g}]")
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class InspectionView:
    dataset_id: str
    kind: str
    verdict: str
    findings: tuple[FindingView, ...]

    @classmethod
    def of(cls, dataset_id: str, report: InspectionReport) -> InspectionView:
        return cls(
            dataset_id=dataset_id,
            kind=report.kind.value,
            verdict=report.verdict.value,
            findings=tuple(FindingView.of(f) for f in report.findings),
        )

    @property
    def passed(self) -> bool:
        return self.verdict != Verdict.FAILED.value

    @property
    def blocking(self) -> tuple[FindingView, ...]:
        return tuple(f for f in self.findings if f.severity == "CRITICAL")

    def render(self) -> str:
        """테스트에서 눈으로 확인하기 위한 출력."""
        lines = [f"[{self.kind}] {self.verdict}  ({self.dataset_id})"]
        lines += [f"  - {f.describe()}" for f in self.findings]
        if not self.findings:
            lines.append("  - (지적 사항 없음)")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ColumnProfileView:
    name: str
    inferred_type: str
    total_count: int
    missing_count: int
    missing_ratio: float
    distinct_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    stddev: float | None
    sample_values: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DatasetProfileView:
    dataset_id: str
    row_count: int
    byte_size: int
    columns: tuple[ColumnProfileView, ...]

    @classmethod
    def of(cls, dataset_id: str, profile: DatasetProfile) -> DatasetProfileView:
        return cls(
            dataset_id=dataset_id,
            row_count=profile.row_count,
            byte_size=profile.byte_size,
            columns=tuple(
                ColumnProfileView(
                    name=c.name,
                    inferred_type=c.inferred_type.value,
                    total_count=c.total_count,
                    missing_count=c.missing_count,
                    missing_ratio=c.missing_ratio,
                    distinct_count=c.distinct_count,
                    minimum=c.minimum,
                    maximum=c.maximum,
                    mean=c.mean,
                    stddev=c.stddev,
                    sample_values=c.sample_values,
                )
                for c in profile.columns
            ),
        )

    def render(self) -> str:
        header = (
            f"{'column':<20}{'type':<11}{'missing':>9}{'distinct':>10}"
            f"{'min':>14}{'max':>14}"
        )
        lines = [
            f"rows={self.row_count:,}  columns={len(self.columns)}  bytes={self.byte_size:,}",
            header,
            "-" * len(header),
        ]
        for c in self.columns:
            minimum = f"{c.minimum:.4g}" if c.minimum is not None else "-"
            maximum = f"{c.maximum:.4g}" if c.maximum is not None else "-"
            lines.append(
                f"{c.name:<20}{c.inferred_type:<11}"
                f"{c.missing_ratio * 100:>8.2f}%{c.distinct_count:>10,}"
                f"{minimum:>14}{maximum:>14}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DatasetView:
    dataset_id: str
    name: str
    status: str
    modality: str
    collected_from: str
    uri: str
    row_count: int | None
    field_count: int | None
    label_class_count: int | None
    input_shape: tuple[int, ...] | None
    inspections: tuple[InspectionView, ...]
    verdict: str | None

    @classmethod
    def of(cls, dataset: Dataset) -> DatasetView:
        dataset_id = str(dataset.id)
        return cls(
            dataset_id=dataset_id,
            name=dataset.name,
            status=dataset.status.value,
            modality=dataset.source.modality.value,
            collected_from=dataset.source.collected_from,
            uri=dataset.source.uri,
            row_count=dataset.profile.row_count if dataset.profile else None,
            field_count=len(dataset.schema.fields) if dataset.schema else None,
            label_class_count=(
                len(dataset.label_space.definitions) if dataset.label_space else None
            ),
            input_shape=(
                dataset.training_spec.input_shape if dataset.training_spec else None
            ),
            inspections=tuple(
                InspectionView.of(dataset_id, report)
                for _, report in sorted(
                    dataset.reports.items(), key=lambda item: item[0].value
                )
            ),
            verdict=dataset.latest_verdict.value if dataset.latest_verdict else None,
        )

    def inspection_of(self, kind: InspectionKind | str) -> InspectionView | None:
        wanted = kind.value if isinstance(kind, InspectionKind) else kind
        for view in self.inspections:
            if view.kind == wanted:
                return view
        return None


@dataclass(frozen=True, slots=True)
class ReadinessView:
    dataset_id: str
    verdict: str
    is_ready: bool
    evaluated_kinds: tuple[str, ...]
    missing_kinds: tuple[str, ...]
    blocking: tuple[FindingView, ...]
    warnings: tuple[FindingView, ...]

    @classmethod
    def of(cls, certificate: ReadinessCertificate) -> ReadinessView:
        return cls(
            dataset_id=str(certificate.dataset_id),
            verdict=certificate.verdict.value,
            is_ready=certificate.is_ready,
            evaluated_kinds=tuple(k.value for k in certificate.evaluated_kinds),
            missing_kinds=tuple(k.value for k in certificate.missing_kinds),
            blocking=tuple(FindingView.of(f) for f in certificate.blocking_findings),
            warnings=tuple(FindingView.of(f) for f in certificate.warning_findings),
        )

    def render(self) -> str:
        lines = [
            f"학습 착수 판정: {self.verdict}  ({self.dataset_id})",
            f"수행한 검사: {', '.join(self.evaluated_kinds) or '없음'}",
        ]
        if self.missing_kinds:
            lines.append(f"누락된 검사: {', '.join(self.missing_kinds)}")
        if self.blocking:
            lines.append("차단 사유:")
            lines += [f"  ✗ {f.describe()}" for f in self.blocking]
        if self.warnings:
            lines.append("경고(진행은 가능):")
            lines += [f"  ! {f.describe()}" for f in self.warnings]
        if not self.blocking and not self.warnings:
            lines.append("  ✓ 막는 것도, 걸리는 것도 없다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SamplingPlanView:
    interval_seconds: float
    value_resolution: float
    retention_days: int
    row_count: int
    event_run_count: int
    lost_event_runs: int
    event_ratio: float
    bytes_per_day: float
    bytes_retained: float
    verdict: str


@dataclass(frozen=True, slots=True)
class SamplingTradeoffView:
    """수집 주기 비교표. (실습 1-11)"""

    dataset_id: str
    plans: tuple[SamplingPlanView, ...]
    cheapest_acceptable: str | None
    acceptable_count: int
    table: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        dataset_id: str,
        tradeoff: SamplingTradeoff,
        findings: tuple[FindingView, ...] = (),
    ) -> SamplingTradeoffView:
        best = tradeoff.cheapest_acceptable()
        return cls(
            dataset_id=dataset_id,
            plans=tuple(
                SamplingPlanView(
                    interval_seconds=plan.interval_seconds,
                    value_resolution=plan.value_resolution,
                    retention_days=plan.retention_days,
                    row_count=observation.row_count,
                    event_run_count=observation.event_run_count,
                    lost_event_runs=observation.lost_event_runs,
                    event_ratio=observation.event_ratio,
                    bytes_per_day=plan.bytes_per_day,
                    bytes_retained=plan.bytes_retained,
                    verdict=(
                        "BLOCKED"
                        if any(f.is_blocking for f in plan_findings)
                        else ("WARNED" if plan_findings else "OK")
                    ),
                )
                for plan, observation, plan_findings in tradeoff.rows
            ),
            cheapest_acceptable=best.describe() if best else None,
            acceptable_count=len(tradeoff.acceptable),
            table=tradeoff.render(),
            findings=findings,
        )

    def plan_at(self, interval_seconds: float) -> SamplingPlanView | None:
        return next(
            (p for p in self.plans if p.interval_seconds == interval_seconds), None
        )

    def render(self) -> str:
        lines = [f"수집 설계 비교 ({self.dataset_id})", self.table]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)
