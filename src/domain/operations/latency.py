"""Latency가 무너지면 무엇이 달라졌는가? (실습 5-5)

모델은 안 바뀌었다. 바이트 하나 안 바뀌었다.
그런데 현장 p95 가 최적화 때 잰 값의 몇 배가 됐다.

바뀐 것은 **환경**이다.

    디바이스가 뜨거워져 클럭이 내려갔다   (thermal throttling)
    다른 프로세스가 코어를 가져갔다
    입력이 밀려 큐가 쌓였다
    메모리가 모자라 스왑이 생겼다
    전원이 절전 모드로 들어갔다

그래서 이 검사는 **모델을 의심하지 않는다.** 환경을 의심한다.
그리고 그 구분을 하려면 모듈 4 에서 잰 기준선이 있어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.operations.window import ObservationWindow
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """한 창의 지연시간."""

    window: ObservationWindow
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    timeout_count: int = 0
    """아예 끝내지 못한 추론. 지연시간 분포에 안 잡힌다 — 따로 세야 한다."""

    def __post_init__(self) -> None:
        for name in ("p50_ms", "p95_ms", "p99_ms", "max_ms"):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)
        if not self.p50_ms <= self.p95_ms <= self.p99_ms <= self.max_ms:
            raise InvariantViolation(
                "분위수가 순서대로가 아니다.", subject="percentiles"
            )
        if self.timeout_count < 0:
            raise InvariantViolation("음수 개수는 없다.", subject="timeout_count")

    @property
    def jitter_ratio(self) -> float:
        return self.p95_ms / self.p50_ms if self.p50_ms else 1.0

    @property
    def timeout_ratio(self) -> float:
        total = self.window.sample_count + self.timeout_count
        return self.timeout_count / total if total else 0.0

    def regression_ratio_to(self, baseline_p95_ms: float) -> float:
        """기준 대비 몇 배가 됐는가. 1.0 이면 그대로다."""
        if baseline_p95_ms <= 0:
            return 0.0
        return self.p95_ms / baseline_p95_ms

    def describe(self) -> str:
        return (
            f"p50 {self.p50_ms:>8.3f}ms  p95 {self.p95_ms:>8.3f}ms  "
            f"p99 {self.p99_ms:>8.3f}ms  지터 {self.jitter_ratio:.2f}배"
            + (f"  타임아웃 {self.timeout_count}건" if self.timeout_count else "")
        )


@dataclass(frozen=True, slots=True)
class LatencyPolicy:
    """현장 지연시간을 어디까지 봐줄 것인가."""

    cycle_budget_ms: float
    """설비가 정한 사이클 타임. 모듈 4 의 DeviceBudget 과 같은 숫자여야 한다."""

    max_regression_ratio: float = 3.0
    """벤치마크의 몇 배까지. PC 와 디바이스는 원래 다르므로 1.0 을 요구하지 않는다."""

    max_jitter_ratio: float = 3.0
    max_timeout_ratio: float = 0.001

    def __post_init__(self) -> None:
        if self.cycle_budget_ms <= 0:
            raise InvariantViolation(
                "사이클 타임 예산은 0보다 커야 한다.", subject="cycle_budget_ms"
            )

    def inspect(
        self, profile: LatencyProfile, baseline_p95_ms: float
    ) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        if profile.p95_ms > self.cycle_budget_ms:
            findings.append(
                Finding(
                    code="OPS_OVER_CYCLE_BUDGET",
                    message=(
                        "현장에서 사이클 타임을 넘고 있다. "
                        "설비가 다음 부품을 기다리거나, 그냥 건너뛴다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=profile.window.label,
                    measured=profile.p95_ms,
                    threshold=self.cycle_budget_ms,
                )
            )

        ratio = profile.regression_ratio_to(baseline_p95_ms)
        if baseline_p95_ms > 0 and ratio > self.max_regression_ratio:
            findings.append(
                Finding(
                    code="OPS_LATENCY_REGRESSION",
                    message=(
                        f"벤치마크의 {ratio:.1f}배로 느려졌다. "
                        "**모델은 안 바뀌었다** — 발열, 다른 프로세스, 전원 정책, "
                        "밀린 큐 중 하나를 의심한다."
                    ),
                    severity=Severity.WARNING
                    if profile.p95_ms <= self.cycle_budget_ms
                    else Severity.CRITICAL,
                    subject=profile.window.label,
                    measured=ratio,
                    threshold=self.max_regression_ratio,
                )
            )

        if profile.jitter_ratio > self.max_jitter_ratio:
            findings.append(
                Finding(
                    code="OPS_LATENCY_JITTER",
                    message=(
                        f"p95 가 p50 의 {profile.jitter_ratio:.1f}배다. "
                        "평소엔 괜찮은데 가끔 크게 튄다 — 대개 다른 작업과 겹치는 순간이다."
                    ),
                    severity=Severity.WARNING,
                    subject=profile.window.label,
                    measured=profile.jitter_ratio,
                    threshold=self.max_jitter_ratio,
                )
            )

        if profile.timeout_ratio > self.max_timeout_ratio:
            findings.append(
                Finding(
                    code="OPS_TIMEOUT",
                    message=(
                        f"추론 {profile.timeout_count}건이 아예 끝나지 않았다. "
                        "이 건들은 지연시간 분포에 안 잡힌다 — p95 만 보면 안 보인다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=profile.window.label,
                    measured=profile.timeout_ratio,
                    threshold=self.max_timeout_ratio,
                )
            )

        return tuple(findings)
