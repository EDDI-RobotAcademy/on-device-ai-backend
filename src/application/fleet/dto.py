"""Fleet Use Case 결과 DTO."""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from domain.fleet.dataset_build import DatasetBuildCheck
from domain.fleet.device import FleetSummary
from domain.fleet.fleet import Fleet
from domain.fleet.lineage import LoopClosure
from domain.fleet.object_key import ObjectStats
from domain.fleet.release import ReleaseCheck
from domain.fleet.rollout import Rollout, WaveResult
from domain.fleet.training_job import RemoteTrainingJob
from domain.fleet.endpoint import EndpointSpec, EndpointState, OnlineInferenceProfile
from domain.fleet.experiment_record import ExperimentLedger
from domain.fleet.governance import BucketGovernance


@dataclass(frozen=True, slots=True)
class UplinkView:
    fleet_id: str
    device_id: str
    accepted: bool
    uri: str
    record_count: int
    payload_kib: float
    sent_today_kib: float
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    def render(self) -> str:
        lines = [
            f"업링크 {self.device_id}  "
            f"{self.record_count:,}건 {self.payload_kib:,.1f}KiB  "
            f"(오늘 누적 {self.sent_today_kib:,.1f}KiB)",
            f"  → {self.uri or '(거절됨)'}",
        ]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class LakeLayoutView:
    fleet_id: str
    prefix: str
    object_count: int
    total_mib: float
    mean_kib: float
    distinct_prefixes: int
    can_narrow: bool
    narrowed_prefix: str
    narrowed_count: int
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        fleet_id: str,
        prefix: str,
        stats: ObjectStats,
        *,
        can_narrow: bool,
        narrowed_prefix: str,
        narrowed_count: int,
        findings: tuple[FindingView, ...] = (),
    ) -> LakeLayoutView:
        return cls(
            fleet_id=fleet_id,
            prefix=prefix,
            object_count=stats.object_count,
            total_mib=stats.total_mib,
            mean_kib=stats.mean_bytes / 1024,
            distinct_prefixes=stats.distinct_prefixes,
            can_narrow=can_narrow,
            narrowed_prefix=narrowed_prefix,
            narrowed_count=narrowed_count,
            findings=findings,
        )

    def render(self) -> str:
        lines = [
            f"객체 저장소 — {self.prefix}",
            f"  객체 {self.object_count:,}개  합계 {self.total_mib:,.1f}MiB  "
            f"평균 {self.mean_kib:,.1f}KiB  접두어 {self.distinct_prefixes:,}종",
            "",
            f"  좁힌 접두어 : {self.narrowed_prefix}",
            f"  훑는 객체   : {self.narrowed_count:,}개 "
            f"(전체 {self.object_count:,}개 중)",
        ]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FleetView:
    fleet_id: str
    name: str
    size: int
    channels: str
    reachable: int
    stale_ratio: float
    version_count: int
    dominant_version: str
    dominant_share: float
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    _summary: str = ""

    @classmethod
    def of(
        cls, fleet: Fleet, summary: FleetSummary, findings: tuple[FindingView, ...] = ()
    ) -> FleetView:
        return cls(
            fleet_id=str(fleet.id),
            name=fleet.name,
            size=fleet.size,
            channels=fleet.channels.describe(),
            reachable=summary.reachable,
            stale_ratio=summary.stale_ratio,
            version_count=summary.version_count,
            dominant_version=summary.dominant_version,
            dominant_share=summary.dominant_share,
            findings=findings,
            _summary=summary.render(),
        )

    def render(self) -> str:
        lines = [
            f"플릿 {self.fleet_id} — {self.name}",
            f"  채널: {self.channels}",
            "",
            self._summary,
        ]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DatasetBuildView:
    fleet_id: str
    build_id: str
    verdict: str
    can_build: bool
    dataset_uri: str
    total_records: int
    total_labeled: int
    device_count: int
    excluded: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    _rendered: str = ""

    @classmethod
    def of(
        cls, fleet_id: str, check: DatasetBuildCheck, dataset_uri: str = ""
    ) -> DatasetBuildView:
        spec = check.spec
        return cls(
            fleet_id=fleet_id,
            build_id=spec.build_id,
            verdict=check.verdict.value,
            can_build=check.can_build,
            dataset_uri=dataset_uri,
            total_records=spec.total_records,
            total_labeled=spec.total_labeled,
            device_count=len(spec.device_ids),
            excluded=spec.excluded_devices,
            findings=tuple(FindingView.of(f) for f in check.findings),
            _rendered=check.render(),
        )

    def render(self) -> str:
        lines = [self._rendered]
        if self.dataset_uri:
            lines.append(f"\n  → {self.dataset_uri}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TrainingJobView:
    job_id: str
    status: str
    is_terminal: bool
    succeeded: bool
    dataset_uri: str
    artifact_uri: str
    failure_reason: str
    instance: str
    worst_case_cost_usd: float
    metrics: dict[str, float] = field(default_factory=dict)
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls, job: RemoteTrainingJob, findings: tuple[FindingView, ...] = ()
    ) -> TrainingJobView:
        return cls(
            job_id=job.job_id,
            status=job.status.value,
            is_terminal=job.is_terminal,
            succeeded=job.succeeded,
            dataset_uri=job.dataset_uri,
            artifact_uri=job.artifact_uri,
            failure_reason=job.failure_reason,
            instance=job.compute.describe(),
            worst_case_cost_usd=job.compute.worst_case_cost_usd,
            metrics=dict(job.metrics),
            findings=findings,
        )

    def render(self) -> str:
        lines = [
            f"학습 {self.job_id}  {self.status}",
            f"  기계   : {self.instance}",
            f"  입력   : {self.dataset_uri}",
        ]
        if self.artifact_uri:
            lines.append(f"  결과물 : {self.artifact_uri}")
        if self.failure_reason:
            lines.append(f"  실패   : {self.failure_reason}")
        if self.metrics:
            lines.append(
                "  지표   : "
                + "  ".join(f"{k}={v:.4f}" for k, v in sorted(self.metrics.items()))
            )
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ReleaseView:
    fleet_id: str
    version: str
    channel: str
    verdict: str
    can_publish: bool
    artifact_bytes: int
    fleet_transfer_mib: float
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    _rendered: str = ""

    @classmethod
    def of(
        cls, fleet_id: str, check: ReleaseCheck, *, device_count: int
    ) -> ReleaseView:
        bundle = check.bundle
        return cls(
            fleet_id=fleet_id,
            version=bundle.version,
            channel=bundle.channel.value,
            verdict=check.verdict.value,
            can_publish=check.can_publish,
            artifact_bytes=bundle.artifact_bytes,
            fleet_transfer_mib=bundle.fleet_transfer_kib(device_count) / 1024,
            findings=tuple(FindingView.of(f) for f in check.findings),
            _rendered=check.render(),
        )

    def render(self) -> str:
        return self._rendered


@dataclass(frozen=True, slots=True)
class RolloutView:
    rollout_id: str
    version: str
    previous_version: str
    status: str
    device_count: int
    succeeded: int
    failed: int
    unreachable: int
    coverage: float
    halt_reason: str
    current_wave: str
    history: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    _rendered: str = ""

    @classmethod
    def of(
        cls, rollout: Rollout, findings: tuple[FindingView, ...] = ()
    ) -> RolloutView:
        wave = rollout.current_wave
        return cls(
            rollout_id=str(rollout.id),
            version=rollout.version,
            previous_version=rollout.previous_version,
            status=rollout.status.value,
            device_count=rollout.device_count,
            succeeded=rollout.succeeded_count,
            failed=rollout.failed_count,
            unreachable=rollout.unreachable_count,
            coverage=rollout.coverage,
            halt_reason=rollout.halt_reason,
            current_wave=wave.name if wave else "",
            history=rollout.history,
            findings=findings,
            _rendered=rollout.render(),
        )

    def render(self) -> str:
        lines = [self._rendered]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class WaveView:
    rollout_id: str
    wave: str
    size: int
    succeeded: int
    failed: int
    unreachable: int
    pending: int
    failure_ratio: float
    halted: bool
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    summary: str = ""

    @classmethod
    def of(
        cls,
        rollout_id: str,
        result: WaveResult,
        *,
        halted: bool,
        findings: tuple[FindingView, ...] = (),
    ) -> WaveView:
        from domain.fleet.rollout import DeviceOutcome

        return cls(
            rollout_id=rollout_id,
            wave=result.wave.name,
            size=result.wave.size,
            succeeded=result.count(DeviceOutcome.SUCCEEDED),
            failed=result.count(DeviceOutcome.FAILED),
            unreachable=result.count(DeviceOutcome.UNREACHABLE),
            pending=result.count(DeviceOutcome.PENDING),
            failure_ratio=result.failure_ratio,
            halted=halted,
            findings=findings,
            summary=result.describe(),
        )

    def render(self) -> str:
        lines = [self.summary]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        if self.halted:
            lines.append("  ⚠ 자동 중단됐다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class LineageView:
    fleet_id: str
    device_id: str
    closed: bool
    verdict: str
    broken_stages: tuple[str, ...] = field(default_factory=tuple)
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    _rendered: str = ""

    @classmethod
    def of(cls, fleet_id: str, closure: LoopClosure) -> LineageView:
        return cls(
            fleet_id=fleet_id,
            device_id=closure.trace.device_id,
            closed=closure.closed,
            verdict=closure.verdict.value,
            broken_stages=tuple(link.stage for link in closure.trace.broken),
            findings=tuple(FindingView.of(f) for f in closure.findings),
            _rendered=closure.render(),
        )

    def render(self) -> str:
        return self._rendered


# ---------------------------------------------------------------------------
# 실습 6-12, 6-13, 6-14
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BucketGovernanceView:
    """버킷 거버넌스 현황. (실습 6-13)"""

    bucket: str
    versioning_enabled: bool
    encryption_algorithm: str | None
    public_access_blocked: bool
    lifecycle_expiration_days: int | None
    statement_count: int
    overwritten_keys: tuple[str, ...]
    verdict: str
    table: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls, governance: BucketGovernance, findings: tuple[FindingView, ...] = ()
    ) -> BucketGovernanceView:
        blocking = any(f.severity == "CRITICAL" for f in findings)
        return cls(
            bucket=governance.bucket,
            versioning_enabled=governance.versioning_enabled,
            encryption_algorithm=governance.encryption_algorithm,
            public_access_blocked=governance.public_access_blocked,
            lifecycle_expiration_days=governance.lifecycle_expiration_days,
            statement_count=len(governance.statements),
            overwritten_keys=governance.overwritten_keys,
            verdict="BLOCKED" if blocking else ("WARNED" if findings else "OK"),
            table=governance.describe(),
            findings=findings,
        )

    def render(self) -> str:
        lines = [self.table]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ExperimentLedgerView:
    """실험 기록 대장. (실습 6-12)"""

    experiment_id: str
    metric: str
    trial_count: int
    reproducible_count: int
    best_trial_id: str | None
    missing_artifacts: tuple[str, ...]
    table: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        ledger: ExperimentLedger,
        *,
        metric: str = "macro_f1",
        findings: tuple[FindingView, ...] = (),
    ) -> ExperimentLedgerView:
        best = ledger.best_by(metric)
        return cls(
            experiment_id=ledger.experiment_id,
            metric=metric,
            trial_count=len(ledger.records),
            reproducible_count=ledger.reproducible_count,
            best_trial_id=best.trial_id if best else None,
            missing_artifacts=ledger.missing_artifacts,
            table=ledger.render(metric),
            findings=findings,
        )

    def render(self) -> str:
        lines = [self.table]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class EndpointView:
    """실시간 추론 엔드포인트. (실습 6-14)"""

    name: str
    status: str
    variants: tuple[tuple[str, float], ...]
    instance_count: int
    monthly_cost_usd: float
    total_latency_ms: float
    cycle_time_ms: float
    verdict: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        spec: EndpointSpec,
        state: EndpointState,
        profile: OnlineInferenceProfile,
        findings: tuple[FindingView, ...] = (),
    ) -> EndpointView:
        blocking = any(f.severity == "CRITICAL" for f in findings)
        return cls(
            name=spec.name,
            status=state.status,
            variants=state.variants,
            instance_count=spec.instance_count,
            monthly_cost_usd=spec.monthly_cost_usd,
            total_latency_ms=profile.total_latency_ms,
            cycle_time_ms=profile.cycle_time_ms,
            verdict="BLOCKED" if blocking else ("WARNED" if findings else "OK"),
            findings=findings,
        )

    def render(self) -> str:
        lines = [
            f"[{self.name}] {self.status}",
            f"  갈래       {', '.join(f'{n} {w:g}' for n, w in self.variants)}",
            f"  인스턴스   {self.instance_count}대  월 ${self.monthly_cost_usd:,.0f}",
            f"  응답       왕복 포함 {self.total_latency_ms:g}ms / "
            f"사이클 {self.cycle_time_ms:g}ms",
        ]
        lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)
