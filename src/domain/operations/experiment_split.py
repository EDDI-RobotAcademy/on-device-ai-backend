"""트래픽을 나눠서 두 모델을 동시에 재라. (실습 5-14)

실습 5-9 의 그림자(shadow)와 다르다. 그 차이가 이 실습의 전부다.

    그림자   새 모델도 같이 돌리지만 **답은 안 쓴다.**
             안전하다. 대신 "새 모델을 썼을 때 현장이 어떻게 달라지는가"는 모른다.

    A/B      트래픽의 일부에 **실제로 새 모델의 답을 쓴다.**
             현장 결과가 나온다. 대신 그 일부는 진짜 위험을 진다.

그래서 A/B 에는 그림자에 없는 규율이 필요하다.

    나누는 기준이 고정되어야 한다
        무작위로 매번 나누면 같은 설비가 어제는 A, 오늘은 B 가 된다.
        그러면 차이가 모델 때문인지 설비 때문인지 알 수 없다.

    양쪽에 최소 표본이 있어야 한다
        B 가 5%면 사고가 나기 전에는 차이가 안 보인다.

    멈출 기준을 먼저 적어야 한다
        "나빠 보이면 멈춘다"는 기준이 아니다.
        시작하기 전에 숫자로 적어 두지 않으면 아무도 못 멈춘다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class TrafficSplit:
    """트래픽을 어떻게 나눌 것인가."""

    control_version: str
    candidate_version: str
    candidate_ratio: float = 0.1
    assignment_key: str = "device_id"
    """무엇을 기준으로 나눌 것인가.

    **고정된 키여야 한다.** 요청마다 무작위로 나누면
    같은 설비가 두 모델을 오가면서 로그가 섞인다.
    """

    def __post_init__(self) -> None:
        if not self.control_version.strip() or not self.candidate_version.strip():
            raise InvariantViolation(
                "양쪽 버전이 모두 있어야 비교가 된다.", subject="version"
            )
        if self.control_version == self.candidate_version:
            raise InvariantViolation(
                "같은 버전을 A/B 로 나눌 수 없다.", subject="version"
            )
        if not 0.0 < self.candidate_ratio < 1.0:
            raise InvariantViolation(
                "후보 비율은 0 초과 1 미만이어야 한다.", subject="candidate_ratio"
            )
        if not self.assignment_key.strip():
            raise InvariantViolation(
                "나누는 기준이 없다. **매번 무작위로 나누면 비교가 성립하지 않는다.**",
                subject="assignment_key",
            )

    def describe(self) -> str:
        return (
            f"{self.control_version} {1 - self.candidate_ratio:.0%} vs "
            f"{self.candidate_version} {self.candidate_ratio:.0%} "
            f"({self.assignment_key} 기준)"
        )


@dataclass(frozen=True, slots=True)
class ArmResult:
    """한쪽에서 나온 현장 결과."""

    version: str
    sample_count: int
    device_count: int
    predicted_mix: Mapping[str, float] = field(default_factory=dict)
    latency_ms_p95: float = 0.0
    alert_count: int = 0
    confirmed_true: int = 0
    """사람이 확인해 준 진짜 이상. **이게 있어야 A/B 가 의미를 가진다.**"""

    confirmed_false: int = 0

    def __post_init__(self) -> None:
        for name in ("sample_count", "device_count", "alert_count"):
            if getattr(self, name) < 0:
                raise InvariantViolation("개수는 음수일 수 없다.", subject=name)

    @property
    def confirmed_total(self) -> int:
        return self.confirmed_true + self.confirmed_false

    @property
    def precision(self) -> float:
        """확인된 알람 중 진짜였던 비율. 현장이 체감하는 숫자다."""
        return (
            self.confirmed_true / self.confirmed_total if self.confirmed_total else 0.0
        )

    @property
    def alerts_per_1k(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.alert_count * 1000.0 / self.sample_count

    def describe(self) -> str:
        return (
            f"{self.version:<14}{self.sample_count:>8,}건 / "
            f"디바이스 {self.device_count:>3}대  "
            f"알람 {self.alerts_per_1k:>6.2f}/1k  "
            f"정밀도 {self.precision:.3f}  p95 {self.latency_ms_p95:>7.2f}ms"
        )


@dataclass(frozen=True, slots=True)
class SplitOutcome:
    """A/B 한 판의 결과. (실습 5-14)"""

    split: TrafficSplit
    control: ArmResult
    candidate: ArmResult

    @property
    def precision_gain(self) -> float:
        return self.candidate.precision - self.control.precision

    @property
    def alert_rate_change(self) -> float:
        return self.candidate.alerts_per_1k - self.control.alerts_per_1k

    @property
    def latency_change(self) -> float:
        return self.candidate.latency_ms_p95 - self.control.latency_ms_p95

    @property
    def actual_ratio(self) -> float:
        total = self.control.sample_count + self.candidate.sample_count
        return self.candidate.sample_count / total if total else 0.0

    def render(self) -> str:
        return "\n".join(
            [
                f"[A/B] {self.split.describe()}",
                f"  {self.control.describe()}",
                f"  {self.candidate.describe()}",
                "",
                f"  정밀도 {self.precision_gain:+.3f}  "
                f"알람율 {self.alert_rate_change:+.2f}/1k  "
                f"p95 {self.latency_change:+.2f}ms",
                f"  실제 배분 {self.actual_ratio:.1%} "
                f"(설계 {self.split.candidate_ratio:.0%})",
            ]
        )


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """이 A/B 를 믿어도 되는가. 그리고 멈춰야 하는가. (실습 5-14)"""

    min_samples_per_arm: int = 500
    min_devices_per_arm: int = 2
    min_confirmed_per_arm: int = 20
    max_ratio_drift: float = 0.05
    """설계한 비율과 실제 배분이 이만큼 이상 어긋나면 나누기가 고장난 것이다."""

    stop_on_precision_drop: float = 0.1
    """후보의 정밀도가 이만큼 떨어지면 **즉시 멈춘다.**"""

    stop_on_latency_ratio: float = 1.5

    def inspect(self, outcome: SplitOutcome) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        for name, arm in (("대조군", outcome.control), ("후보", outcome.candidate)):
            if arm.sample_count < self.min_samples_per_arm:
                findings.append(
                    Finding(
                        code="AB_TOO_FEW_SAMPLES",
                        message=(
                            f"{name}({arm.version})의 표본이 {arm.sample_count:,}건이다. "
                            "**이 수로 나온 차이는 다음 주에 뒤집힌다.**"
                        ),
                        severity=Severity.CRITICAL,
                        subject=arm.version,
                        measured=float(arm.sample_count),
                        threshold=float(self.min_samples_per_arm),
                    )
                )
            if arm.device_count < self.min_devices_per_arm:
                findings.append(
                    Finding(
                        code="AB_TOO_FEW_DEVICES",
                        message=(
                            f"{name}이 디바이스 {arm.device_count}대에서만 돌았다. "
                            "**한 대의 특성이 곧 결과가 된다** — "
                            "그 설비가 원래 문제가 많았을 수도 있다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=arm.version,
                        measured=float(arm.device_count),
                        threshold=float(self.min_devices_per_arm),
                    )
                )
            if arm.confirmed_total < self.min_confirmed_per_arm:
                findings.append(
                    Finding(
                        code="AB_NO_GROUND_TRUTH",
                        message=(
                            f"{name}에서 사람이 확인해 준 건이 "
                            f"{arm.confirmed_total}건뿐이다. "
                            "**정답 없이는 '더 낫다'를 말할 수 없다** — "
                            "예측 분포가 달라진 것만으로는 어느 쪽이 옳은지 모른다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=arm.version,
                        measured=float(arm.confirmed_total),
                        threshold=float(self.min_confirmed_per_arm),
                    )
                )

        drift = abs(outcome.actual_ratio - outcome.split.candidate_ratio)
        if drift > self.max_ratio_drift:
            findings.append(
                Finding(
                    code="AB_ASSIGNMENT_SKEWED",
                    message=(
                        f"설계는 {outcome.split.candidate_ratio:.0%} 인데 "
                        f"실제는 {outcome.actual_ratio:.1%} 다. "
                        "**나누기가 고장났거나 한쪽 디바이스가 더 바쁘다** — "
                        "어느 쪽이든 지금 비교는 공평하지 않다."
                    ),
                    severity=Severity.WARNING,
                    subject=outcome.split.assignment_key,
                    measured=drift,
                    threshold=self.max_ratio_drift,
                )
            )

        if outcome.precision_gain < -self.stop_on_precision_drop:
            findings.append(
                Finding(
                    code="AB_STOP_PRECISION",
                    message=(
                        f"후보의 정밀도가 {outcome.precision_gain:.3f} 낮다. "
                        "**멈춤 기준을 넘었다 — 지금 되돌린다.** "
                        "이 기준은 시작 전에 적어 두었기 때문에 지금 논쟁이 필요 없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=outcome.candidate.version,
                    measured=outcome.precision_gain,
                    threshold=-self.stop_on_precision_drop,
                )
            )

        if (
            outcome.control.latency_ms_p95 > 0
            and outcome.candidate.latency_ms_p95
            > outcome.control.latency_ms_p95 * self.stop_on_latency_ratio
        ):
            findings.append(
                Finding(
                    code="AB_STOP_LATENCY",
                    message=(
                        f"후보의 p95 가 대조군의 "
                        f"{outcome.candidate.latency_ms_p95 / outcome.control.latency_ms_p95:.2f}배다. "
                        "**정확도가 좋아도 사이클을 놓치면 못 쓴다.**"
                    ),
                    severity=Severity.CRITICAL,
                    subject=outcome.candidate.version,
                    measured=outcome.candidate.latency_ms_p95,
                    threshold=outcome.control.latency_ms_p95
                    * self.stop_on_latency_ratio,
                )
            )

        return tuple(findings)
