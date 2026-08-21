"""한 창에 대한 종합 판정과, 창들을 이은 시간선. (실습 5-4)

**"언제부터 이상해졌죠?"**

이 질문에 답할 수 있는 유일한 방법은 창을 하나씩 남겨 두는 것이다.
지금 이상하다는 것은 지금 봐도 안다. 언제부터인지는 기록이 없으면 영영 모른다.

그리고 답할 때 두 가지를 구분해야 한다.

    알람이 울린 시각    사람이 알아챈 시각. **늘 늦다.**
    실제로 시작된 시각  지표가 처음 기준을 넘긴 시각

그리고 또 하나.

    한 번 튄 것(spike)  다음 창에서 돌아온다. 대개 아무 일도 아니다.
    무너진 것(sustained) 연속으로 넘는다. 이쪽이 사건이다.

둘을 구분하지 않으면 알람이 하루에 열 번 울리고, 사람은 곧 알람을 끈다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.operations.drift import DriftReport
from domain.operations.latency import LatencyProfile
from domain.operations.prediction_mix import PredictionMix
from domain.operations.window import ObservationWindow
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity, Verdict, derive_verdict


class HealthMetric(Enum):
    """시간선에서 추적할 수 있는 지표."""

    LATENCY_P95 = "LATENCY_P95"
    PREDICTION_SHIFT = "PREDICTION_SHIFT"
    INPUT_PSI = "INPUT_PSI"
    CONFIDENCE = "CONFIDENCE"


@dataclass(frozen=True, slots=True)
class HealthReport:
    """한 창을 본 결과 전부."""

    window: ObservationWindow
    deployment_version: int
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    latency: LatencyProfile | None = None
    mix: PredictionMix | None = None
    drift: DriftReport | None = None
    baseline_mix: dict[str, float] = field(default_factory=dict)

    @property
    def verdict(self) -> Verdict:
        """저장하지 않고 Finding 에서 유도한다 (Shared Kernel 규칙)."""
        return derive_verdict(self.findings)

    @property
    def is_healthy(self) -> bool:
        return self.verdict is Verdict.PASSED

    @property
    def blocking(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.CRITICAL)

    def value_of(self, metric: HealthMetric) -> float | None:
        """시간선에 찍을 숫자 하나를 꺼낸다."""
        if metric is HealthMetric.LATENCY_P95:
            return self.latency.p95_ms if self.latency else None
        if metric is HealthMetric.PREDICTION_SHIFT:
            if self.mix is None or not self.baseline_mix:
                return None
            return self.mix.shift_from(self.baseline_mix)
        if metric is HealthMetric.INPUT_PSI:
            return self.drift.max_psi if self.drift else None
        if metric is HealthMetric.CONFIDENCE:
            return self.mix.overall_confidence if self.mix else None
        return None

    def render(self) -> str:
        lines = [
            f"관측 {self.window.describe()}  v{self.deployment_version}  "
            f"→ {self.verdict.value}"
        ]
        if self.latency:
            lines.append(f"  지연시간 : {self.latency.describe()}")
        if self.mix:
            lines.append(
                f"  예측분포 : "
                + "  ".join(
                    f"{label} {self.mix.share_of(label):.1%}"
                    for label in sorted(self.mix.counts, key=lambda x: -self.mix.counts[x])
                )
                + f"  (확신 {self.mix.overall_confidence:.3f})"
            )
        if self.drift:
            lines.append(f"  입력분포 : 최대 PSI {self.drift.max_psi:.4f}")
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TimelinePoint:
    """시간선 위의 한 점."""

    window_label: str
    started_at: str
    value: float
    exceeded: bool


@dataclass(frozen=True, slots=True)
class OnsetFinding:
    """언제부터인가에 대한 답."""

    metric: HealthMetric
    threshold: float
    first_exceeded: TimelinePoint | None
    sustained_from: TimelinePoint | None
    consecutive_required: int
    points: tuple[TimelinePoint, ...] = field(default_factory=tuple)

    @property
    def is_sustained(self) -> bool:
        return self.sustained_from is not None

    @property
    def spike_only(self) -> bool:
        """넘긴 적은 있는데 이어지지 않았다. **사건이 아니다.**"""
        return self.first_exceeded is not None and self.sustained_from is None

    def render(self) -> str:
        lines = [
            f"{self.metric.value} 기준 {self.threshold:g} — "
            f"{self.consecutive_required}창 연속을 사건으로 본다",
            "-" * 62,
        ]
        for point in self.points:
            mark = "  ✗" if point.exceeded else "  ·"
            lines.append(
                f"{mark} {point.window_label:<24}{point.value:>10.4f}"
            )
        lines.append("-" * 62)
        if self.first_exceeded:
            lines.append(f"  처음 넘긴 창   : {self.first_exceeded.window_label}")
        else:
            lines.append("  한 번도 넘지 않았다")
        if self.sustained_from:
            lines.append(f"  **무너진 시점** : {self.sustained_from.window_label}")
        elif self.first_exceeded:
            lines.append("  이어지지 않았다 — 한 번 튄 것이다")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class HealthTimeline:
    """창들을 시간 순으로 이은 것."""

    reports: tuple[HealthReport, ...] = field(default_factory=tuple)

    @property
    def latest(self) -> HealthReport | None:
        return self.reports[-1] if self.reports else None

    def __len__(self) -> int:
        return len(self.reports)

    def onset_of(
        self,
        metric: HealthMetric,
        threshold: float,
        *,
        consecutive: int = 3,
    ) -> OnsetFinding:
        """기준을 처음 넘긴 창과, 연속으로 넘기 시작한 창을 찾는다.

        **찾는 것은 마지막이 아니라 처음이다.**
        사람이 알아챈 시각이 아니라 실제로 시작된 시각을 답해야 하기 때문이다.
        """
        if consecutive < 1:
            raise InvariantViolation(
                "연속 창 수는 1 이상이어야 한다.", subject="consecutive"
            )

        points: list[TimelinePoint] = []
        for report in self.reports:
            value = report.value_of(metric)
            if value is None:
                continue
            points.append(
                TimelinePoint(
                    window_label=report.window.label,
                    started_at=report.window.started_at,
                    value=value,
                    exceeded=value > threshold,
                )
            )

        first = next((p for p in points if p.exceeded), None)
        sustained = None
        run = 0
        for index, point in enumerate(points):
            run = run + 1 if point.exceeded else 0
            if run >= consecutive:
                sustained = points[index - consecutive + 1]
                break

        return OnsetFinding(
            metric=metric,
            threshold=threshold,
            first_exceeded=first,
            sustained_from=sustained,
            consecutive_required=consecutive,
            points=tuple(points),
        )

    def values_of(self, metric: HealthMetric) -> tuple[float, ...]:
        return tuple(
            value
            for value in (report.value_of(metric) for report in self.reports)
            if value is not None
        )

    def render(self) -> str:
        lines = [f"{'창':<24}{'판정':<22}{'p95(ms)':>10}{'분포이동':>10}{'PSI':>9}"]
        lines.append("-" * 76)
        for report in self.reports:
            latency = report.value_of(HealthMetric.LATENCY_P95)
            shift = report.value_of(HealthMetric.PREDICTION_SHIFT)
            psi = report.value_of(HealthMetric.INPUT_PSI)
            lines.append(
                f"{report.window.label:<24}{report.verdict.value:<22}"
                f"{latency if latency is not None else 0:>10.3f}"
                f"{shift if shift is not None else 0:>10.1%}"
                f"{psi if psi is not None else 0:>9.4f}"
            )
        return "\n".join(lines)
