"""재학습이 필요한 순간을 직접 정의하라. (실습 5-11)

가장 흔한 답: **"정확도가 떨어지면 재학습한다."**

이 문장은 실행할 수 없다. 현장에 정답이 없어서 정확도를 못 재기 때문이다.
모듈 5 전체가 그 사실 위에 서 있었다.

실행 가능한 기준은 이렇게 생겼다.

    입력 분포가 변했고         (실습 5-7)
    그 상태가 지속됐고         (실습 5-4 — 한 번 튄 게 아니다)
    새 라벨이 충분히 모였다    (없으면 재학습해도 같은 것을 배운다)

세 번째가 자주 잊힌다. **재학습은 새 데이터가 아니라 새 라벨을 요구한다.**
드리프트가 아무리 심해도 라벨이 없으면 학습시킬 것이 없다.

그리고 재학습은 공짜가 아니다.

    라벨링 비용 + 검증 비용 + 배포 위험 + 다시 무너질 위험

그래서 이 판정은 "필요하다/아니다"가 아니라
**"지금 시작해야 하는가, 무엇부터 해야 하는가"** 에 답해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.operations.health import HealthMetric, HealthTimeline
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class TriggerReason(Enum):
    INPUT_DRIFT = "INPUT_DRIFT"
    """들어오는 값이 학습 때와 달라졌다."""

    PREDICTION_SHIFT = "PREDICTION_SHIFT"
    """나가는 답의 구성이 달라졌다."""

    CONFIDENCE_DECAY = "CONFIDENCE_DECAY"
    """모델이 점점 덜 확신한다."""

    MEASURED_ACCURACY_DROP = "MEASURED_ACCURACY_DROP"
    """정답이 붙은 표본에서 실제로 확인된 하락. **가장 강한 근거다.**"""

    NEW_LABELS_AVAILABLE = "NEW_LABELS_AVAILABLE"
    """새 라벨이 충분히 모였다. 그 자체로는 이유가 아니지만 조건이다."""

    SCHEDULE = "SCHEDULE"
    """정기 재학습. 아무 일이 없어도 주기적으로 한다."""


class Urgency(Enum):
    NONE = "NONE"
    WATCH = "WATCH"
    """지금 시작할 필요는 없다. 다음 창을 본다."""

    PLAN = "PLAN"
    """라벨링을 시작한다. 모델은 아직 돌려도 된다."""

    NOW = "NOW"
    """멈추고 시작한다."""


@dataclass(frozen=True, slots=True)
class LabelSupply:
    """재학습에 쓸 수 있는 라벨이 얼마나 모였는가.

    이것이 없으면 재학습 논의는 시작되지 않는다.
    """

    total_records: int
    labeled_records: int
    labeled_since_deploy: int = 0
    minority_label_counts: dict[str, int] = field(default_factory=dict)
    """클래스별 새 라벨 수. **FAULT 가 0건이면 재학습해도 FAULT 를 못 배운다.**"""

    def __post_init__(self) -> None:
        if self.labeled_records > self.total_records:
            raise InvariantViolation(
                "라벨이 전체보다 많다.", subject="labeled_records"
            )

    @property
    def labeled_ratio(self) -> float:
        return self.labeled_records / self.total_records if self.total_records else 0.0

    def starved_labels(self, minimum: int) -> tuple[str, ...]:
        return tuple(
            label
            for label, count in self.minority_label_counts.items()
            if count < minimum
        )

    def describe(self) -> str:
        counts = "  ".join(
            f"{label} {count}" for label, count in sorted(self.minority_label_counts.items())
        )
        return (
            f"라벨 {self.labeled_records:,}/{self.total_records:,} "
            f"({self.labeled_ratio:.1%})  배포 후 {self.labeled_since_deploy:,}건"
            + (f"  [{counts}]" if counts else "")
        )


@dataclass(frozen=True, slots=True)
class RetrainingDecision:
    """재학습 판정."""

    needed: bool
    urgency: Urgency
    reasons: tuple[TriggerReason, ...] = field(default_factory=tuple)
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    blockers: tuple[str, ...] = field(default_factory=tuple)
    """재학습해야 하는데 지금은 할 수 없는 이유. **이것이 진짜 할 일 목록이다.**"""

    supply: LabelSupply | None = None

    @property
    def can_start(self) -> bool:
        return self.needed and not self.blockers

    def render(self) -> str:
        lines = [
            f"재학습 판정: {'필요' if self.needed else '불필요'}  "
            f"(긴급도 {self.urgency.value})"
        ]
        if self.reasons:
            lines.append(
                "  근거 : " + ", ".join(reason.value for reason in self.reasons)
            )
        if self.supply:
            lines.append(f"  라벨 : {self.supply.describe()}")
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        if self.blockers:
            lines.append("")
            lines.append("  지금 시작할 수 없는 이유:")
            lines += [f"    ✗ {blocker}" for blocker in self.blockers]
        elif self.needed:
            lines.append("")
            lines.append("  지금 시작할 수 있다. → 모듈 1 로 돌아간다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RetrainingPolicy:
    """언제 재학습을 시작할 것인가.

    이 숫자들은 통계가 아니라 **현장 사정**이 정한다.
    라벨링에 2주가 걸리면 기준을 낮게 잡아 미리 시작해야 한다.
    """

    drift_psi_threshold: float = 0.2
    prediction_shift_threshold: float = 0.15
    sustained_windows: int = 3
    """이만큼 연속으로 넘겨야 사건이다. 한 번 튄 것으로 재학습하지 않는다."""

    min_new_labels: int = 500
    min_labels_per_class: int = 30
    confidence_floor: float = 0.6
    measured_accuracy_floor: float | None = None
    """정답이 붙은 표본에서 잰 정확도의 하한. 정답이 있을 때만 쓴다."""

    def decide(
        self,
        timeline: HealthTimeline,
        supply: LabelSupply,
        *,
        measured_accuracy: float | None = None,
    ) -> RetrainingDecision:
        if len(timeline) == 0:
            raise InvariantViolation(
                "관측 없이 재학습을 판단할 수 없다.", subject="timeline"
            )

        findings: list[Finding] = []
        reasons: list[TriggerReason] = []
        urgency = Urgency.NONE

        drift = timeline.onset_of(
            HealthMetric.INPUT_PSI,
            self.drift_psi_threshold,
            consecutive=self.sustained_windows,
        )
        if drift.is_sustained:
            reasons.append(TriggerReason.INPUT_DRIFT)
            urgency = Urgency.PLAN
            findings.append(
                Finding(
                    code="RETRAIN_SUSTAINED_INPUT_DRIFT",
                    message=(
                        f"입력 분포가 '{drift.sustained_from.window_label}' 부터 "
                        f"{self.sustained_windows}창 연속으로 기준을 넘었다. "
                        "현실이 학습 데이터에서 멀어진 것이다."
                    ),
                    severity=Severity.WARNING,
                    subject=HealthMetric.INPUT_PSI.value,
                    measured=drift.sustained_from.value,
                    threshold=self.drift_psi_threshold,
                )
            )
        elif drift.spike_only:
            findings.append(
                Finding(
                    code="RETRAIN_DRIFT_SPIKE_ONLY",
                    message=(
                        f"'{drift.first_exceeded.window_label}' 에서 한 번 튀었지만 "
                        "이어지지 않았다. 재학습 사유가 아니다."
                    ),
                    severity=Severity.INFO,
                    subject=HealthMetric.INPUT_PSI.value,
                )
            )
            urgency = Urgency.WATCH

        shift = timeline.onset_of(
            HealthMetric.PREDICTION_SHIFT,
            self.prediction_shift_threshold,
            consecutive=self.sustained_windows,
        )
        if shift.is_sustained:
            reasons.append(TriggerReason.PREDICTION_SHIFT)
            urgency = Urgency.PLAN
            findings.append(
                Finding(
                    code="RETRAIN_SUSTAINED_PREDICTION_SHIFT",
                    message=(
                        f"예측 분포가 '{shift.sustained_from.window_label}' 부터 "
                        "계속 달라져 있다."
                    ),
                    severity=Severity.WARNING,
                    subject=HealthMetric.PREDICTION_SHIFT.value,
                    measured=shift.sustained_from.value,
                    threshold=self.prediction_shift_threshold,
                )
            )

        latest = timeline.latest
        confidence = latest.value_of(HealthMetric.CONFIDENCE) if latest else None
        if confidence is not None and confidence < self.confidence_floor:
            reasons.append(TriggerReason.CONFIDENCE_DECAY)
            urgency = Urgency.PLAN
            findings.append(
                Finding(
                    code="RETRAIN_CONFIDENCE_DECAY",
                    message=(
                        f"평균 확신도가 {confidence:.2f} 까지 내려갔다. "
                        "모델이 지금 보고 있는 것을 학습 때 별로 못 봤다는 신호다."
                    ),
                    severity=Severity.WARNING,
                    subject=HealthMetric.CONFIDENCE.value,
                    measured=confidence,
                    threshold=self.confidence_floor,
                )
            )

        if (
            self.measured_accuracy_floor is not None
            and measured_accuracy is not None
            and measured_accuracy < self.measured_accuracy_floor
        ):
            reasons.append(TriggerReason.MEASURED_ACCURACY_DROP)
            urgency = Urgency.NOW
            findings.append(
                Finding(
                    code="RETRAIN_MEASURED_ACCURACY_DROP",
                    message=(
                        f"정답이 붙은 표본에서 실제 정확도가 {measured_accuracy:.4f} 다. "
                        "**추정이 아니라 확인된 하락이다.**"
                    ),
                    severity=Severity.CRITICAL,
                    subject="measured_accuracy",
                    measured=measured_accuracy,
                    threshold=self.measured_accuracy_floor,
                )
            )

        needed = bool(reasons)
        blockers = self._blockers(supply) if needed else ()
        if needed and TriggerReason.NEW_LABELS_AVAILABLE not in reasons and not blockers:
            reasons.append(TriggerReason.NEW_LABELS_AVAILABLE)

        return RetrainingDecision(
            needed=needed,
            urgency=urgency,
            reasons=tuple(reasons),
            findings=tuple(findings),
            blockers=blockers,
            supply=supply,
        )

    def _blockers(self, supply: LabelSupply) -> tuple[str, ...]:
        """재학습해야 하는데 지금 시작할 수 없는 이유."""
        blockers: list[str] = []
        if supply.labeled_since_deploy < self.min_new_labels:
            blockers.append(
                f"배포 후 새로 붙은 라벨이 {supply.labeled_since_deploy:,}건뿐이다 "
                f"({self.min_new_labels:,}건 필요). "
                "지금 학습하면 예전 데이터를 다시 배운다."
            )
        starved = supply.starved_labels(self.min_labels_per_class)
        if starved:
            blockers.append(
                f"'{', '.join(starved)}' 의 새 라벨이 {self.min_labels_per_class}건 미만이다. "
                "이 클래스는 재학습해도 나아지지 않는다."
            )
        return tuple(blockers)
