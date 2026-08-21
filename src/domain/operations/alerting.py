"""알람이 너무 많으면 아무도 안 본다. (실습 5-13)

모델이 이상을 잡아냈다고 알람이 되는 것이 아니다.
**알람은 사람이 반응할 수 있을 때만 알람이다.**

현장에서 실제로 벌어지는 일:

    1주차   알람이 뜬다. 사람이 달려간다.
    2주차   알람이 하루 200번 뜬다. 사람이 몇 개만 본다.
    3주차   알람 소리를 끈다.
    4주차   진짜 사고가 났을 때 아무도 못 봤다.

그래서 알람에는 규율이 필요하다.

    억제(debounce)  같은 상태가 이어지는 동안 한 번만 낸다
    확인(dwell)     N번 연속 같은 판단일 때만 낸다 — 한 번 튄 것으로 깨우지 않는다
    상한(budget)    시간당 최대 몇 건. 넘으면 **묶어서** 한 건으로 보낸다
    해제(clear)     정상으로 돌아왔다는 것도 알려야 한다

여기서 정하는 것은 "무엇을 이상이라 부를 것인가"가 아니다.
그건 모델이 한다. 여기서 정하는 것은 **"그것을 어떻게 알릴 것인가"**다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class AlertRule:
    """언제 사람을 부를 것인가."""

    alert_labels: tuple[str, ...]
    """이 라벨이 나오면 알람 후보다."""

    dwell: int = 3
    """이만큼 연속으로 같은 판단이어야 알람을 낸다.

    1이면 한 번 튄 것에도 사람을 부른다. **현장 신호는 원래 가끔 튄다.**
    """

    min_confidence: float = 0.6
    """이보다 확신이 낮으면 알람으로 올리지 않는다. 대신 보류로 센다."""

    cooldown_seconds: float = 300.0
    """한 번 낸 뒤 이 시간 동안은 같은 알람을 다시 내지 않는다."""

    hourly_budget: int = 12
    """시간당 상한. 넘으면 묶는다."""

    def __post_init__(self) -> None:
        if not self.alert_labels:
            raise InvariantViolation(
                "무엇을 알릴지 정하지 않았다.", subject="alert_labels"
            )
        if self.dwell < 1:
            raise InvariantViolation("확인 횟수는 1 이상이어야 한다.", subject="dwell")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise InvariantViolation(
                "확신 문턱은 0~1 이어야 한다.", subject="min_confidence"
            )
        if self.cooldown_seconds < 0:
            raise InvariantViolation(
                "억제 시간은 음수일 수 없다.", subject="cooldown_seconds"
            )
        if self.hourly_budget < 1:
            raise InvariantViolation(
                "시간당 상한은 1 이상이어야 한다.", subject="hourly_budget"
            )

    def describe(self) -> str:
        return (
            f"{'/'.join(self.alert_labels)} · {self.dwell}회 연속 · "
            f"확신≥{self.min_confidence:g} · 억제 {self.cooldown_seconds:g}s · "
            f"시간당 {self.hourly_budget}건"
        )


@dataclass(frozen=True, slots=True)
class Signal:
    """디바이스가 낸 판단 하나. 알람의 재료다."""

    at_seconds: float
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise InvariantViolation("확신은 0~1 이어야 한다.", subject="confidence")


@dataclass(frozen=True, slots=True)
class Alert:
    """실제로 나간 알람 하나."""

    at_seconds: float
    label: str
    confidence: float
    merged_count: int = 1
    """상한에 걸려 묶인 건수. 1이면 묶지 않은 것이다."""

    @property
    def is_merged(self) -> bool:
        return self.merged_count > 1


@dataclass(frozen=True, slots=True)
class AlertLedger:
    """규율을 적용한 결과. (실습 5-13)

    **원본 신호 수와 실제 알람 수를 함께 들고 다닌다.**
    둘 중 하나만 보면 "잘 잡는다"와 "너무 많다"를 구분할 수 없다.
    """

    rule: AlertRule
    raw_signal_count: int
    alerts: tuple[Alert, ...] = field(default_factory=tuple)
    withheld_low_confidence: int = 0
    suppressed_by_dwell: int = 0
    suppressed_by_cooldown: int = 0
    merged_by_budget: int = 0
    observed_seconds: float = 3600.0

    @property
    def alert_count(self) -> int:
        return len(self.alerts)

    @property
    def alerts_per_hour(self) -> float:
        if self.observed_seconds <= 0:
            return 0.0
        return self.alert_count * 3600.0 / self.observed_seconds

    @property
    def suppression_ratio(self) -> float:
        """원본 신호 중 알람이 되지 않은 비율."""
        if self.raw_signal_count == 0:
            return 0.0
        return 1.0 - self.alert_count / self.raw_signal_count

    def render(self) -> str:
        return "\n".join(
            [
                f"[알람 규율] {self.rule.describe()}",
                f"  원본 신호        {self.raw_signal_count:>7,}건",
                f"  확신 부족 보류   {self.withheld_low_confidence:>7,}건",
                f"  연속 확인 미달   {self.suppressed_by_dwell:>7,}건",
                f"  억제 시간 내     {self.suppressed_by_cooldown:>7,}건",
                f"  상한으로 묶음    {self.merged_by_budget:>7,}건",
                f"  **실제 알람      {self.alert_count:>7,}건**  "
                f"(시간당 {self.alerts_per_hour:.1f}건)",
            ]
        )


class AlertGate:
    """신호를 알람으로 바꾸는 Domain Service. (실습 5-13)

    상태를 들고 있지 않다. 신호 묶음을 통째로 받아 한 번에 판정한다.
    그래야 같은 입력에 항상 같은 결과가 나온다 — 현장 사고 재현에 필요하다.
    """

    def apply(self, rule: AlertRule, signals: Sequence[Signal]) -> AlertLedger:
        ordered = sorted(signals, key=lambda s: s.at_seconds)
        raw = [s for s in ordered if s.label in rule.alert_labels]

        alerts: list[Alert] = []
        withheld = dwell_blocked = cooled = 0
        streak = 0
        last_alert_at: float | None = None

        for signal in ordered:
            if signal.label not in rule.alert_labels:
                streak = 0
                continue
            if signal.confidence < rule.min_confidence:
                withheld += 1
                streak = 0  # 확신 없는 판단은 연속으로 세지 않는다
                continue

            streak += 1
            if streak < rule.dwell:
                dwell_blocked += 1
                continue
            if (
                last_alert_at is not None
                and signal.at_seconds - last_alert_at < rule.cooldown_seconds
            ):
                cooled += 1
                continue

            alerts.append(
                Alert(
                    at_seconds=signal.at_seconds,
                    label=signal.label,
                    confidence=signal.confidence,
                )
            )
            last_alert_at = signal.at_seconds

        observed = (
            ordered[-1].at_seconds - ordered[0].at_seconds if len(ordered) > 1 else 3600.0
        )
        observed = max(observed, 1.0)
        alerts, merged = _apply_budget(alerts, rule, observed)

        return AlertLedger(
            rule=rule,
            raw_signal_count=len(raw),
            alerts=tuple(alerts),
            withheld_low_confidence=withheld,
            suppressed_by_dwell=dwell_blocked,
            suppressed_by_cooldown=cooled,
            merged_by_budget=merged,
            observed_seconds=observed,
        )


def _apply_budget(
    alerts: list[Alert], rule: AlertRule, observed_seconds: float
) -> tuple[list[Alert], int]:
    """시간당 상한을 넘으면 **버리지 않고 묶는다.**

    버리면 그 사건이 없던 일이 된다. 묶으면 "이 시간에 N건"이 남는다.
    """
    if not alerts:
        return alerts, 0

    kept: list[Alert] = []
    merged = 0
    bucket_start = alerts[0].at_seconds
    in_bucket = 0
    overflow = 0

    for alert in alerts:
        if alert.at_seconds - bucket_start >= 3600.0:
            if overflow:
                kept.append(
                    Alert(
                        at_seconds=bucket_start + 3600.0,
                        label=alert.label,
                        confidence=alert.confidence,
                        merged_count=overflow,
                    )
                )
                merged += overflow
            bucket_start = alert.at_seconds
            in_bucket = overflow = 0

        if in_bucket < rule.hourly_budget:
            kept.append(alert)
            in_bucket += 1
        else:
            overflow += 1

    if overflow:
        kept.append(
            Alert(
                at_seconds=bucket_start + 3600.0,
                label=alerts[-1].label,
                confidence=alerts[-1].confidence,
                merged_count=overflow,
            )
        )
        merged += overflow

    return kept, merged


@dataclass(frozen=True, slots=True)
class AlertFatiguePolicy:
    """이 알람 설정으로 현장이 버틸 수 있는가. (실습 5-13)"""

    max_alerts_per_hour: float = 6.0
    """사람이 하루에 반응할 수 있는 양에서 거꾸로 나온 숫자다."""

    max_suppression_ratio: float = 0.0
    """원본 신호의 이 비율 이상을 눌러 버리면 너무 조용한 것이다. 0이면 검사하지 않는다.

    시간당 알람 수로는 이걸 못 잡는다 — 관측 구간이 짧으면
    한 건만 나와도 '하루 14건'으로 환산되기 때문이다.
    **비율은 관측 길이에 흔들리지 않는다.**
    """

    def inspect(self, ledger: AlertLedger) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        if ledger.alerts_per_hour > self.max_alerts_per_hour:
            findings.append(
                Finding(
                    code="ALERT_FATIGUE",
                    message=(
                        f"시간당 {ledger.alerts_per_hour:.1f}건이다. "
                        "**이 속도면 2주 안에 알람을 끈다.** "
                        "그 다음 진짜 사고가 나면 아무도 못 본다 — "
                        "알람을 줄이는 것은 편의가 아니라 안전이다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=ledger.rule.describe(),
                    measured=ledger.alerts_per_hour,
                    threshold=self.max_alerts_per_hour,
                )
            )

        if ledger.rule.dwell == 1 and ledger.raw_signal_count:
            findings.append(
                Finding(
                    code="ALERT_NO_DWELL",
                    message=(
                        "연속 확인 없이 한 번 튄 것으로 알람을 낸다. "
                        "**현장 신호는 원래 가끔 튄다** — "
                        "센서 하나가 한 표본 흔들린 것으로 사람을 부르게 된다."
                    ),
                    severity=Severity.WARNING,
                    subject="dwell",
                    measured=1.0,
                    threshold=2.0,
                )
            )

        if ledger.merged_by_budget:
            findings.append(
                Finding(
                    code="ALERT_BUDGET_EXCEEDED",
                    message=(
                        f"상한을 넘어 {ledger.merged_by_budget}건을 묶었다. "
                        "**묶은 것이지 버린 것이 아니다** — "
                        "그러나 이 상황 자체가 '무언가 크게 잘못되었다'는 신호다. "
                        "규율을 조이기 전에 모델과 현장을 먼저 봐야 한다."
                    ),
                    severity=Severity.WARNING,
                    subject="budget",
                    measured=float(ledger.merged_by_budget),
                )
            )

        if (
            self.max_suppression_ratio > 0
            and ledger.raw_signal_count > 0
            and ledger.suppression_ratio > self.max_suppression_ratio
        ):
            findings.append(
                Finding(
                    code="ALERT_TOO_QUIET",
                    message=(
                        f"신호 {ledger.raw_signal_count:,}건 중 "
                        f"{ledger.suppression_ratio:.1%} 를 눌렀다 "
                        f"(알람 {ledger.alert_count}건). "
                        "**규율이 너무 세면 알람도 조용해지고 사고도 조용해진다.** "
                        "양쪽 끝이 모두 위험하다."
                    ),
                    severity=Severity.WARNING,
                    subject="dwell/cooldown",
                    measured=ledger.suppression_ratio,
                    threshold=self.max_suppression_ratio,
                )
            )

        return tuple(findings)


class StreamingAlertGate:
    """디바이스에서 도는 알람 게이트. (참조 구현 — edge-agent)

    `AlertGate` 는 신호 묶음을 통째로 받는다. 사고를 재현할 때는 그게 맞다.
    그러나 **디바이스에는 미래가 없다.** 표본 하나가 들어올 때마다 지금 판단해야 한다.

    그래서 같은 규칙을 증분으로 다시 쓴다. 지키는 약속은 하나다.

        같은 신호를 같은 순서로 넣으면 **AlertGate 와 같은 결과가 나온다.**

    그 약속을 테스트가 매번 확인한다. 확인하지 않으면
    "서버에서는 3건인데 디바이스에서는 40건"이 되고, 그때는 어느 쪽이 맞는지 모른다.

    들고 있는 상태는 네 개뿐이다 — 보드의 RAM 은 넉넉하지 않다.
    """

    __slots__ = (
        "_rule",
        "_streak",
        "_last_alert_at",
        "_bucket_start",
        "_in_bucket",
        "_overflow",
        "_raw",
        "_withheld",
        "_dwell_blocked",
        "_cooled",
        "_merged",
        "_emitted",
        "_first_at",
        "_last_at",
    )

    def __init__(self, rule: AlertRule) -> None:
        self._rule = rule
        self._streak = 0
        self._last_alert_at: float | None = None
        self._bucket_start: float | None = None
        self._in_bucket = 0
        self._overflow = 0
        self._raw = 0
        self._withheld = 0
        self._dwell_blocked = 0
        self._cooled = 0
        self._merged = 0
        self._emitted: list[Alert] = []
        self._first_at: float | None = None
        self._last_at: float | None = None

    @property
    def rule(self) -> AlertRule:
        return self._rule

    def offer(self, signal: Signal) -> Alert | None:
        """판단 하나를 넣는다. 사람을 불러야 하면 Alert 를, 아니면 None 을 준다.

        **시간은 되돌아가지 않는다.** 순서가 어긋난 신호는 거부한다 —
        디바이스 시계가 NTP 로 뒤로 점프하면 억제가 통째로 풀리기 때문이다.
        """
        if self._last_at is not None and signal.at_seconds < self._last_at:
            raise InvariantViolation(
                f"시간이 되돌아갔다 ({self._last_at:g} → {signal.at_seconds:g}). "
                "**시계가 뒤로 뛰면 억제가 풀린다** — 그대로 두면 알람이 쏟아진다.",
                subject="at_seconds",
            )
        if self._first_at is None:
            self._first_at = signal.at_seconds
        self._last_at = signal.at_seconds

        if signal.label not in self._rule.alert_labels:
            self._streak = 0
            return None

        self._raw += 1
        if signal.confidence < self._rule.min_confidence:
            self._withheld += 1
            self._streak = 0
            return None

        self._streak += 1
        if self._streak < self._rule.dwell:
            self._dwell_blocked += 1
            return None
        if (
            self._last_alert_at is not None
            and signal.at_seconds - self._last_alert_at < self._rule.cooldown_seconds
        ):
            self._cooled += 1
            return None

        self._last_alert_at = signal.at_seconds
        return self._through_budget(
            Alert(
                at_seconds=signal.at_seconds,
                label=signal.label,
                confidence=signal.confidence,
            )
        )

    def ledger(self) -> AlertLedger:
        """지금까지의 장부. 업링크에 이대로 실어 보낸다 (실습 6-1)."""
        observed = (
            self._last_at - self._first_at
            if self._first_at is not None and self._last_at is not None
            else 0.0
        )
        return AlertLedger(
            rule=self._rule,
            raw_signal_count=self._raw,
            alerts=tuple(self._emitted),
            withheld_low_confidence=self._withheld,
            suppressed_by_dwell=self._dwell_blocked,
            suppressed_by_cooldown=self._cooled,
            merged_by_budget=self._merged,
            observed_seconds=max(observed, 1.0),
        )

    # -- 내부 --------------------------------------------------------------
    def _through_budget(self, alert: Alert) -> Alert | None:
        """시간당 상한. 넘으면 **버리지 않고 다음 묶음으로 미룬다.**"""
        if self._bucket_start is None:
            self._bucket_start = alert.at_seconds

        rolled: Alert | None = None
        if alert.at_seconds - self._bucket_start >= 3600.0:
            if self._overflow:
                rolled = Alert(
                    at_seconds=self._bucket_start + 3600.0,
                    label=alert.label,
                    confidence=alert.confidence,
                    merged_count=self._overflow,
                )
                self._emitted.append(rolled)
                self._merged += self._overflow
            self._bucket_start = alert.at_seconds
            self._in_bucket = 0
            self._overflow = 0

        if self._in_bucket < self._rule.hourly_budget:
            self._in_bucket += 1
            self._emitted.append(alert)
            return alert

        self._overflow += 1
        return rolled

    def close(self) -> Alert | None:
        """관측을 끝낸다. 아직 안 나간 묶음이 있으면 마지막으로 내보낸다."""
        if not self._overflow:
            return None
        assert self._bucket_start is not None
        last = self._emitted[-1] if self._emitted else None
        merged = Alert(
            at_seconds=self._bucket_start + 3600.0,
            label=last.label if last else self._rule.alert_labels[0],
            confidence=last.confidence if last else 1.0,
            merged_count=self._overflow,
        )
        self._emitted.append(merged)
        self._merged += self._overflow
        self._overflow = 0
        return merged
