"""얼마나 자주, 얼마나 곱게 잴 것인가. (실습 1-11)

수집 주기는 나중에 바꿀 수 없는 결정이다.

    빠르게 잰 것은 나중에 느리게 만들 수 있다.
    **느리게 잰 것은 나중에 빠르게 만들 수 없다.**

그래서 이 결정은 데이터 수집이 시작되기 전에 끝나야 한다.
한 달치를 30초 간격으로 모으고 나서 "10초가 필요했다"를 알게 되면
그 한 달은 없는 것이 된다.

기준은 두 개다.

    사건    잡으려는 것이 몇 초짜리인가. 그 사건이 표본 몇 개로 그려지는가.
    비용    주기를 반으로 줄이면 저장·전송·전력이 두 배가 된다.

해상도도 같은 이야기다. 유효 자릿수가 굵으면 작은 변화가 값에서 사라진다.
이미지라면 **가는 균열이 리사이즈에서 뭉개진다** (실습 3-11 에서 다시 만난다).
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    """수집 설계 하나."""

    interval_seconds: float
    """표본 사이의 간격."""

    value_resolution: float = 0.0
    """값을 얼마나 곱게 남기는가. 0 이면 원본 그대로.

    0.5 면 0.5 단위로 반올림해 저장한다는 뜻이다.
    저장은 줄지만 **그보다 작은 변화는 영원히 사라진다.**
    """

    retention_days: int = 30
    bytes_per_sample: int = 64

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise InvariantViolation(
                "수집 주기는 0보다 커야 한다.", subject="interval_seconds"
            )
        if self.value_resolution < 0:
            raise InvariantViolation(
                "해상도는 음수일 수 없다.", subject="value_resolution"
            )
        if self.retention_days < 1:
            raise InvariantViolation(
                "보관 기간은 1일 이상이어야 한다.", subject="retention_days"
            )

    @property
    def samples_per_day(self) -> float:
        return 86_400.0 / self.interval_seconds

    @property
    def bytes_per_day(self) -> float:
        return self.samples_per_day * self.bytes_per_sample

    @property
    def bytes_retained(self) -> float:
        return self.bytes_per_day * self.retention_days

    def samples_covering(self, duration_seconds: float) -> float:
        """이 길이의 사건이 표본 몇 개로 그려지는가."""
        return duration_seconds / self.interval_seconds

    def describe(self) -> str:
        resolution = (
            f"해상도 {self.value_resolution:g}" if self.value_resolution else "해상도 원본"
        )
        return (
            f"{self.interval_seconds:g}초 간격 / {resolution} / "
            f"{self.retention_days}일 보관 "
            f"({self.bytes_retained / 1024 / 1024:.1f} MiB)"
        )


@dataclass(frozen=True, slots=True)
class SamplingObservation:
    """이 주기로 실제로 다시 뽑아 본 결과. Infrastructure 가 채운다.

    설계가 옳은지는 이론이 아니라 **원본을 그 주기로 다시 뽑아 보면** 안다.
    """

    interval_seconds: float
    row_count: int
    event_row_count: int
    event_run_count: int
    """사건 구간(연속된 이상 구간)이 몇 덩어리로 남았는가."""

    shortest_event_seconds: float
    """원본에서 관측된 가장 짧은 사건 구간.

    **이것이 곧 설계 기준은 아니다.** 1표본짜리 깜빡임까지 잡겠다고 하면
    어떤 주기로도 통과하지 못한다. 무엇을 반드시 잡아야 하는지는
    현장이 정하고, 그 값은 SamplingDesignPolicy 에 들어간다.
    """

    typical_event_seconds: float = 0.0
    """사건 구간 길이의 중앙값. 설계 기준을 정할 때 이 숫자를 본다."""

    lost_event_runs: int = 0
    """원본에는 있었는데 이 주기에서는 통째로 사라진 사건 구간 수."""

    distinct_value_count: int = 0
    """이 해상도로 남는 서로 다른 값의 수."""

    def __post_init__(self) -> None:
        if self.row_count < 0 or self.event_row_count < 0:
            raise InvariantViolation("행 수는 음수일 수 없다.", subject="row_count")

    @property
    def event_ratio(self) -> float:
        return self.event_row_count / self.row_count if self.row_count else 0.0

    def describe(self) -> str:
        return (
            f"{self.interval_seconds:>6.0f}초  "
            f"{self.row_count:>8,}행  "
            f"사건 {self.event_run_count:>3}구간 / {self.event_row_count:>6,}행 "
            f"({self.event_ratio:.2%})  "
            f"사라진 구간 {self.lost_event_runs}"
        )


@dataclass(frozen=True, slots=True)
class SamplingDesignPolicy:
    """이 주기로 모아도 되는가. (실습 1-11)"""

    target_event_seconds: float = 60.0
    """**반드시 잡아야 하는 가장 짧은 사건**의 길이.

    이 숫자는 코드가 정하는 것이 아니다. 현장이 정한다.
    "30초짜리 트립은 놓쳐도 되지만 1분짜리는 안 된다" 같은 합의가 여기 들어온다.
    이 한 줄을 안 정하고 주기를 논하면 논의가 끝나지 않는다.
    """

    min_samples_per_event: float = 5.0
    """그 사건이 최소 이만큼의 표본으로 그려져야 한다.

    2개면 이론상 주파수는 복원되지만(나이퀴스트), 모양은 복원되지 않는다.
    모델이 보는 것은 주파수가 아니라 **모양**이다.
    """

    overkill_samples_per_event: float = 200.0
    """이보다 촘촘하면 사건을 위해서가 아니라 습관으로 모으고 있는 것이다."""

    max_bytes_retained: float = 512 * 1024 * 1024
    min_distinct_values: int = 50

    def inspect(
        self, plan: SamplingPlan, observation: SamplingObservation
    ) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        if observation.lost_event_runs:
            findings.append(
                Finding(
                    code="SAMPLING_EVENT_LOST",
                    message=(
                        f"이 주기에서는 사건 구간 {observation.lost_event_runs}개가 "
                        "**통째로 사라진다.** 표본 사이로 빠진 사건은 "
                        "라벨도 없고 학습에도 안 들어간다 — "
                        "그리고 나중에 '왜 못 잡느냐'는 질문으로 돌아온다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=f"{plan.interval_seconds:g}s",
                    measured=float(observation.lost_event_runs),
                    threshold=0.0,
                )
            )

        covered = plan.samples_covering(self.target_event_seconds)
        if covered < self.min_samples_per_event:
            findings.append(
                Finding(
                    code="SAMPLING_TOO_SLOW",
                    message=(
                        f"반드시 잡아야 하는 사건이 {self.target_event_seconds:g}초인데 "
                        f"표본 {covered:.1f}개로 그려진다. "
                        "**점 두세 개로는 모양이 되지 않는다** — "
                        "모델은 그 사건을 잡음과 구분할 근거가 없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=f"{plan.interval_seconds:g}s",
                    measured=covered,
                    threshold=self.min_samples_per_event,
                )
            )
        elif covered > self.overkill_samples_per_event:
            findings.append(
                Finding(
                    code="SAMPLING_OVERKILL",
                    message=(
                        f"목표 사건이 표본 {covered:.0f}개로 그려진다. "
                        "**필요해서가 아니라 습관으로 모으고 있는 것은 아닌가.** "
                        f"주기를 늘리면 하루 {plan.bytes_per_day / 1024 / 1024:.1f} MiB "
                        "가 그만큼 줄어든다."
                    ),
                    severity=Severity.INFO,
                    subject=f"{plan.interval_seconds:g}s",
                    measured=covered,
                    threshold=self.overkill_samples_per_event,
                )
            )

        if plan.bytes_retained > self.max_bytes_retained:
            findings.append(
                Finding(
                    code="SAMPLING_OVER_BUDGET",
                    message=(
                        f"{plan.retention_days}일치가 "
                        f"{plan.bytes_retained / 1024 / 1024:.0f} MiB 다. "
                        "**디바이스 한 대 기준이다** — "
                        "3,000대면 여기에 3,000을 곱한다 (모듈 6)."
                    ),
                    severity=Severity.WARNING,
                    subject=f"{plan.interval_seconds:g}s",
                    measured=plan.bytes_retained,
                    threshold=self.max_bytes_retained,
                )
            )

        if (
            plan.value_resolution > 0
            and 0 < observation.distinct_value_count < self.min_distinct_values
        ):
            findings.append(
                Finding(
                    code="RESOLUTION_TOO_COARSE",
                    message=(
                        f"해상도 {plan.value_resolution:g} 로 저장하면 "
                        f"서로 다른 값이 {observation.distinct_value_count}종류만 남는다. "
                        "**그보다 작은 변화는 값에서 영원히 사라진다** — "
                        "나중에 원본이 없으면 되살릴 수 없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=f"resolution={plan.value_resolution:g}",
                    measured=float(observation.distinct_value_count),
                    threshold=float(self.min_distinct_values),
                )
            )

        return tuple(findings)


@dataclass(frozen=True, slots=True)
class SamplingTradeoff:
    """여러 주기를 나란히 놓은 표. (실습 1-11)"""

    rows: tuple[tuple[SamplingPlan, SamplingObservation, tuple[Finding, ...]], ...]

    @property
    def acceptable(self) -> tuple[SamplingPlan, ...]:
        """막는 소견이 없는 설계들."""
        return tuple(
            plan
            for plan, _, findings in self.rows
            if not any(f.is_blocking for f in findings)
        )

    def cheapest_acceptable(self) -> SamplingPlan | None:
        """통과하는 것 중 가장 싼 것.

        **가장 빠른 것이 답이 아니다.** 사건을 담을 수 있는 것 중 가장 싼 것이 답이다.
        """
        candidates = self.acceptable
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.interval_seconds)

    def render(self) -> str:
        header = (
            f"{'주기':>8}{'행 수':>10}{'사건구간':>10}{'사라짐':>8}"
            f"{'하루':>12}{'판정':>8}"
        )
        lines = [header, "-" * len(header)]
        for plan, observation, findings in self.rows:
            verdict = (
                "막힘" if any(f.is_blocking for f in findings)
                else ("경고" if findings else "통과")
            )
            lines.append(
                f"{plan.interval_seconds:>7.0f}s{observation.row_count:>10,}"
                f"{observation.event_run_count:>10}{observation.lost_event_runs:>8}"
                f"{plan.bytes_per_day / 1024 / 1024:>11.1f}M{verdict:>8}"
            )
        lines.append("-" * len(header))
        best = self.cheapest_acceptable()
        if best is not None:
            lines.append(
                f"  통과하는 것 중 가장 싼 설계: {best.describe()}"
            )
        else:
            lines.append("  통과하는 설계가 없다. 사건을 담을 수 있는 주기가 아니다.")
        return "\n".join(lines)
