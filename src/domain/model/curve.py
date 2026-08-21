"""학습 곡선 — 모델이 무엇을 하고 있는지 읽는다. (실습 3-6, 3-7)

Loss 숫자 하나만 보면 아무것도 알 수 없다. **곡선의 모양**을 봐야 한다.

    학습 손실이 안 떨어진다              → 배우지 못하고 있다 (3-6)
    학습은 떨어지는데 검증이 올라간다      → 외우기 시작했다 (3-7)
    둘 다 떨어지다 검증만 정체            → 여기가 멈출 자리다

이 판단은 사람이 그래프를 눈으로 보고 하는 일이었다.
눈으로 하면 사람마다 다르고, 기록에 남지 않는다. 그래서 규칙으로 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """한 epoch 이 남긴 사실."""

    epoch: int
    train_loss: float
    validation_loss: float
    train_accuracy: float
    validation_accuracy: float
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise InvariantViolation("epoch 은 1부터 센다.", subject="epoch")
        for name in ("train_loss", "validation_loss"):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)
        for name in ("train_accuracy", "validation_accuracy"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 이어야 한다.", subject=name)

    @property
    def generalization_gap(self) -> float:
        """학습과 검증의 거리. 벌어질수록 외우고 있다는 뜻이다."""
        return self.train_accuracy - self.validation_accuracy


@dataclass(frozen=True, slots=True)
class TrainingCurve:
    """epoch 들의 나열. 여기서 모든 판단이 나온다."""

    records: tuple[EpochRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        epochs = [r.epoch for r in self.records]
        if epochs != sorted(epochs):
            raise InvariantViolation("epoch 이 순서대로가 아니다.", subject="records")
        if len(epochs) != len(set(epochs)):
            raise InvariantViolation("같은 epoch 이 두 번 있다.", subject="records")

    def __len__(self) -> int:
        return len(self.records)

    @property
    def is_empty(self) -> bool:
        return not self.records

    @property
    def first(self) -> EpochRecord | None:
        return self.records[0] if self.records else None

    @property
    def last(self) -> EpochRecord | None:
        return self.records[-1] if self.records else None

    @property
    def best_epoch(self) -> EpochRecord | None:
        """검증 손실이 가장 낮았던 지점. **여기가 저장해야 할 모델이다.**"""
        if not self.records:
            return None
        return min(self.records, key=lambda r: r.validation_loss)

    @property
    def train_loss_drop(self) -> float:
        """학습 손실이 처음 대비 얼마나 떨어졌는가 (비율)."""
        if len(self.records) < 2 or self.records[0].train_loss <= 0:
            return 0.0
        first, last = self.records[0].train_loss, self.records[-1].train_loss
        return (first - last) / first

    @property
    def overfitting_epoch(self) -> int | None:
        """검증 손실이 최저를 찍고 다시 올라가기 시작한 지점.

        **모델이 데이터를 외우기 시작한 순간이다.**
        """
        best = self.best_epoch
        if best is None or best.epoch == self.records[-1].epoch:
            return None
        after = [r for r in self.records if r.epoch > best.epoch]
        rising = [r for r in after if r.validation_loss > best.validation_loss]
        return rising[0].epoch if rising else None

    @property
    def final_gap(self) -> float:
        return self.records[-1].generalization_gap if self.records else 0.0

    @property
    def wasted_epochs(self) -> int:
        """최저점 이후로 더 돈 횟수. 외우는 데 쓴 시간이다."""
        best = self.best_epoch
        if best is None:
            return 0
        return self.records[-1].epoch - best.epoch

    @property
    def total_seconds(self) -> float:
        return sum(r.duration_seconds for r in self.records)

    def render(self) -> str:
        lines = [
            f"{'epoch':>6}{'train_loss':>13}{'val_loss':>12}"
            f"{'train_acc':>12}{'val_acc':>10}{'gap':>9}",
            "-" * 62,
        ]
        best = self.best_epoch
        for record in self.records:
            mark = "  ← best" if best and record.epoch == best.epoch else ""
            lines.append(
                f"{record.epoch:>6}{record.train_loss:>13.4f}"
                f"{record.validation_loss:>12.4f}{record.train_accuracy:>12.3f}"
                f"{record.validation_accuracy:>10.3f}"
                f"{record.generalization_gap:>9.3f}{mark}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class LearningPolicy:
    """모델이 실제로 배우고 있는지에 대한 기준. (실습 3-6)"""

    min_train_loss_drop: float = 0.2
    """학습 손실이 20% 도 안 떨어졌다면 배우고 있는 것이 아니다."""

    min_accuracy_over_baseline: float = 0.02
    """다수 클래스만 찍는 것보다 나은가."""

    max_epochs_without_progress: int = 5

    def inspect(
        self, curve: TrainingCurve, baseline_accuracy: float = 0.0
    ) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        if curve.is_empty:
            return (
                Finding(
                    code="LEARNING_NO_EPOCH",
                    message="epoch 이 하나도 기록되지 않았다.",
                    severity=Severity.CRITICAL,
                    subject="curve",
                ),
            )

        if curve.train_loss_drop < self.min_train_loss_drop:
            findings.append(
                Finding(
                    code="LEARNING_LOSS_FLAT",
                    message=(
                        f"학습 손실이 {curve.train_loss_drop:.1%} 밖에 떨어지지 않았다. "
                        "learning rate, 입력 정규화, 라벨 연결을 먼저 의심한다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="train_loss",
                    measured=curve.train_loss_drop,
                    threshold=self.min_train_loss_drop,
                )
            )

        best = curve.best_epoch
        if best is not None and baseline_accuracy > 0:
            margin = best.validation_accuracy - baseline_accuracy
            if margin < self.min_accuracy_over_baseline:
                findings.append(
                    Finding(
                        code="LEARNING_NO_BETTER_THAN_BASELINE",
                        message=(
                            f"검증 정확도 {best.validation_accuracy:.1%} 가 "
                            f"다수 클래스만 찍는 {baseline_accuracy:.1%} 보다 "
                            "의미 있게 낫지 않다."
                        ),
                        severity=Severity.CRITICAL,
                        subject="validation_accuracy",
                        measured=margin,
                        threshold=self.min_accuracy_over_baseline,
                    )
                )

        if curve.wasted_epochs > self.max_epochs_without_progress:
            findings.append(
                Finding(
                    code="LEARNING_WASTED_EPOCHS",
                    message=(
                        f"최저점(epoch {best.epoch}) 이후 {curve.wasted_epochs} epoch 을 "
                        "더 돌았다. 조기 종료를 걸면 그만큼의 시간과 과적합을 아낀다."
                    ),
                    severity=Severity.WARNING,
                    subject="epochs",
                    measured=float(curve.wasted_epochs),
                    threshold=float(self.max_epochs_without_progress),
                )
            )

        return tuple(findings)


@dataclass(frozen=True, slots=True)
class OverfittingPolicy:
    """모델이 외우기 시작한 순간을 잡는 기준. (실습 3-7)"""

    max_generalization_gap: float = 0.10
    """학습 정확도가 검증보다 10%p 넘게 높으면 외우고 있다."""

    max_validation_loss_rise: float = 0.05
    """최저점 대비 검증 손실이 5% 넘게 올라가면 지나쳤다."""

    def inspect(self, curve: TrainingCurve) -> tuple[Finding, ...]:
        if curve.is_empty:
            return ()

        findings: list[Finding] = []
        best = curve.best_epoch
        last = curve.records[-1]

        onset = curve.overfitting_epoch
        if onset is not None:
            findings.append(
                Finding(
                    code="OVERFIT_ONSET",
                    message=(
                        f"epoch {onset} 부터 검증 손실이 다시 올라간다. "
                        f"저장해야 할 모델은 마지막이 아니라 epoch {best.epoch} 의 것이다."
                    ),
                    severity=Severity.WARNING,
                    subject="validation_loss",
                    measured=float(onset),
                    threshold=float(best.epoch),
                )
            )

        if best is not None and best.validation_loss > 0:
            rise = (last.validation_loss - best.validation_loss) / best.validation_loss
            if rise > self.max_validation_loss_rise:
                findings.append(
                    Finding(
                        code="OVERFIT_LOSS_RISING",
                        message=(
                            f"검증 손실이 최저점 대비 {rise:.1%} 올라간 채로 끝났다. "
                            "마지막 가중치를 그대로 배포하면 최고 성능을 버리는 것이다."
                        ),
                        severity=Severity.CRITICAL,
                        subject="validation_loss",
                        measured=rise,
                        threshold=self.max_validation_loss_rise,
                    )
                )

        if last.generalization_gap > self.max_generalization_gap:
            findings.append(
                Finding(
                    code="OVERFIT_GAP_WIDE",
                    message=(
                        f"학습 정확도 {last.train_accuracy:.1%} vs "
                        f"검증 정확도 {last.validation_accuracy:.1%}. "
                        "모델이 패턴이 아니라 표본을 외우고 있다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="generalization_gap",
                    measured=last.generalization_gap,
                    threshold=self.max_generalization_gap,
                )
            )

        return tuple(findings)
