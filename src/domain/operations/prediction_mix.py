"""Prediction Distribution의 변화를 추적하라. (실습 5-6)

**이 지표의 값어치는 하나다 — 정답이 없어도 잴 수 있다.**

현장에는 라벨이 없다. 그래서 정확도를 못 잰다.
그런데 모델이 **무엇을 얼마나 답하고 있는지**는 셀 수 있다.

    평가 때   NORMAL 74% / OVERLOAD 21% / FAULT 5%
    지금      NORMAL 41% / OVERLOAD 22% / FAULT 37%

정확도는 여전히 모른다. 그러나 무언가 벌어졌다는 것은 안다.
가능성은 둘이고, **둘 다 사람이 확인해야 하는 상황이다.**

    세상이 변했다   → 설비가 실제로 자주 서고 있다 (모델은 잘 하고 있다)
    모델이 무너졌다 → 입력이 변해 엉뚱한 답을 하고 있다

확신도가 함께 내려갔다면 두 번째 쪽이다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.operations.window import ObservationWindow
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class PredictionMix:
    """한 창에서 모델이 낸 답의 구성."""

    window: ObservationWindow
    counts: Mapping[str, int]
    mean_confidence: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(count < 0 for count in self.counts.values()):
            raise InvariantViolation("음수 개수는 없다.", subject="counts")

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.counts)

    def share_of(self, label: str) -> float:
        return self.counts.get(label, 0) / self.total if self.total else 0.0

    @property
    def shares(self) -> dict[str, float]:
        return {label: self.share_of(label) for label in self.counts}

    @property
    def overall_confidence(self) -> float:
        """전체 평균 확신도. 클래스별 개수로 가중한다."""
        if not self.total:
            return 0.0
        weighted = sum(
            self.mean_confidence.get(label, 0.0) * count
            for label, count in self.counts.items()
        )
        return weighted / self.total

    def shift_from(self, baseline: Mapping[str, float]) -> float:
        """분포가 얼마나 달라졌는가 (총변동거리, 0~1).

        PSI 대신 이것을 쓰는 이유: **클래스가 몇 개 안 된다.**
        3개짜리 분포에 PSI 를 쓰면 한 칸이 비는 순간 무한대로 튄다.
        """
        labels = set(self.counts) | set(baseline)
        return 0.5 * sum(
            abs(self.share_of(label) - baseline.get(label, 0.0)) for label in labels
        )

    def vanished_from(self, baseline: Mapping[str, float]) -> tuple[str, ...]:
        """기준에서는 답했는데 이 창에서 한 번도 안 나온 클래스.

        실습 3-9 의 EVAL_CLASS_NEVER_PREDICTED 가 현장에서 다시 나타난 것이다.
        """
        return tuple(
            label
            for label, share in baseline.items()
            if share > 0 and self.counts.get(label, 0) == 0
        )

    def surged_from(
        self, baseline: Mapping[str, float], *, factor: float = 3.0
    ) -> tuple[tuple[str, float], ...]:
        """비율이 몇 배로 뛴 클래스. 알람 폭주는 여기서 잡힌다."""
        surged: list[tuple[str, float]] = []
        for label, share in baseline.items():
            if share <= 0:
                continue
            ratio = self.share_of(label) / share
            if ratio >= factor:
                surged.append((label, ratio))
        return tuple(sorted(surged, key=lambda item: -item[1]))

    def render(self, baseline: Mapping[str, float] | None = None) -> str:
        lines = [f"{'클래스':<12}{'건수':>8}{'비율':>9}{'평균확신':>10}"]
        if baseline:
            lines[0] += f"{'평가때':>9}{'차이':>9}"
        lines.append("-" * (len(lines[0]) + 8))
        for label in sorted(self.counts, key=lambda x: -self.counts[x]):
            row = (
                f"{label:<12}{self.counts[label]:>8,}{self.share_of(label):>9.1%}"
                f"{self.mean_confidence.get(label, 0.0):>10.3f}"
            )
            if baseline:
                expected = baseline.get(label, 0.0)
                row += f"{expected:>9.1%}{self.share_of(label) - expected:>+9.1%}"
            lines.append(row)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PredictionDriftPolicy:
    """예측 분포가 얼마나 변하면 사람을 불러야 하는가."""

    max_shift: float = 0.15
    """총변동거리. 0.15 면 표본의 15% 가 다른 클래스로 옮겨간 셈이다."""

    min_confidence: float = 0.6
    max_confidence_drop: float = 0.15
    surge_factor: float = 3.0
    critical_labels: frozenset[str] = frozenset()
    """놓치면 안 되는 클래스. 이것이 사라지면 WARNING 이 아니라 CRITICAL 이다."""

    def inspect(
        self,
        mix: PredictionMix,
        baseline: Mapping[str, float],
        baseline_confidence: float | None = None,
    ) -> tuple[Finding, ...]:
        if not baseline:
            return (
                Finding(
                    code="OPS_NO_MIX_BASELINE",
                    message=(
                        "기준 예측 분포가 없다. "
                        "현장 분포를 봐도 달라졌는지 말할 수 없다."
                    ),
                    severity=Severity.WARNING,
                    subject=mix.window.label,
                ),
            )

        findings: list[Finding] = []
        shift = mix.shift_from(baseline)
        if shift > self.max_shift:
            findings.append(
                Finding(
                    code="OPS_PREDICTION_SHIFT",
                    message=(
                        f"예측 분포가 기준과 {shift:.1%} 달라졌다. "
                        "세상이 변했거나 모델이 무너졌다 — **둘 다 사람이 봐야 한다.**"
                    ),
                    severity=Severity.WARNING,
                    subject=mix.window.label,
                    measured=shift,
                    threshold=self.max_shift,
                )
            )

        for label in mix.vanished_from(baseline):
            findings.append(
                Finding(
                    code="OPS_LABEL_VANISHED",
                    message=(
                        f"'{label}' 을 현장에서 한 번도 예측하지 않았다. "
                        "이 클래스에 대해서는 모델이 없는 것과 같다."
                    ),
                    severity=Severity.CRITICAL
                    if label in self.critical_labels
                    else Severity.WARNING,
                    subject=label,
                    measured=0.0,
                    threshold=baseline[label],
                )
            )

        for label, ratio in mix.surged_from(baseline, factor=self.surge_factor):
            findings.append(
                Finding(
                    code="OPS_LABEL_SURGE",
                    message=(
                        f"'{label}' 비율이 기준의 {ratio:.1f}배다. "
                        "현장에 알람이 그만큼 더 울리고 있다는 뜻이다 — "
                        "사람이 곧 알람을 꺼 버린다."
                    ),
                    severity=Severity.WARNING,
                    subject=label,
                    measured=mix.share_of(label),
                    threshold=baseline[label] * self.surge_factor,
                )
            )

        confidence = mix.overall_confidence
        if confidence and confidence < self.min_confidence:
            findings.append(
                Finding(
                    code="OPS_LOW_CONFIDENCE",
                    message=(
                        f"평균 확신도가 {confidence:.2f} 다. "
                        "모델이 답은 하고 있지만 스스로도 헷갈리고 있다."
                    ),
                    severity=Severity.WARNING,
                    subject=mix.window.label,
                    measured=confidence,
                    threshold=self.min_confidence,
                )
            )

        if baseline_confidence:
            drop = baseline_confidence - confidence
            if drop > self.max_confidence_drop:
                findings.append(
                    Finding(
                        code="OPS_CONFIDENCE_DROP",
                        message=(
                            f"확신도가 {drop:.2f} 내려갔다. "
                            "분포 변화와 **함께** 나타나면 세상이 변한 게 아니라 "
                            "모델이 무너진 쪽이다."
                        ),
                        severity=Severity.WARNING,
                        subject=mix.window.label,
                        measured=drop,
                        threshold=self.max_confidence_drop,
                    )
                )

        return tuple(findings)
