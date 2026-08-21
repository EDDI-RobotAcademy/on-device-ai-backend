"""새 모델과 기존 모델을 실제 데이터로 비교하라. (실습 5-9)

오프라인 평가에서 새 모델이 더 좋았다. 그래서 배포한다?

**아니다.** 오프라인 평가는 학습 때 모아 둔 데이터로 한 것이다.
현장은 그 데이터가 아니다 — 그 사실이 이 모듈 전체의 전제였다.

그래서 **같은 현장 입력을 둘 다에게 넣는다.** 새 모델의 답은 **쓰지 않는다.**
설비는 여전히 기존 모델의 답으로 움직인다. 새 모델은 옆에서 조용히 답만 남긴다.

이것을 shadow(그림자) 또는 canary 라고 부른다.

그리고 여기서 정직해야 하는 지점이 하나 있다.

    두 모델이 **다르다**는 것은 정답 없이도 말할 수 있다.
    두 모델 중 어느 쪽이 **낫다**는 것은 정답 없이 말할 수 없다.

라벨이 붙은 표본이 없으면 이 비교는 "다르다"까지만 답한다.
그것을 "낫다"로 바꿔 읽는 순간, 근거 없는 배포가 된다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.operations.window import ObservationWindow
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity, Verdict, derive_verdict


@dataclass(frozen=True, slots=True)
class ShadowRun:
    """같은 입력을 두 모델에 넣어 본 결과."""

    window: ObservationWindow
    incumbent_label: str
    candidate_label: str
    sample_count: int
    agreement_count: int
    """두 모델의 답이 같았던 건수."""

    incumbent_p95_ms: float
    candidate_p95_ms: float
    incumbent_mix: Mapping[str, int] = field(default_factory=dict)
    candidate_mix: Mapping[str, int] = field(default_factory=dict)

    labeled_count: int = 0
    """정답이 붙은 표본 수. **대개 전체의 일부다.**"""

    incumbent_correct: int = 0
    candidate_correct: int = 0

    def __post_init__(self) -> None:
        if self.sample_count < 1:
            raise InvariantViolation(
                "표본이 없으면 비교한 것이 아니다.", subject="sample_count"
            )
        if self.agreement_count > self.sample_count:
            raise InvariantViolation(
                "일치 건수가 전체보다 많다.", subject="agreement_count"
            )
        if self.labeled_count > self.sample_count:
            raise InvariantViolation(
                "정답 붙은 표본이 전체보다 많다.", subject="labeled_count"
            )
        for name in ("incumbent_correct", "candidate_correct"):
            if getattr(self, name) > self.labeled_count:
                raise InvariantViolation(
                    f"{name} 가 정답 붙은 표본보다 많다.", subject=name
                )

    @property
    def agreement_ratio(self) -> float:
        return self.agreement_count / self.sample_count

    @property
    def disagreement_count(self) -> int:
        return self.sample_count - self.agreement_count

    @property
    def has_labels(self) -> bool:
        """'낫다'고 말할 수 있는가."""
        return self.labeled_count > 0

    @property
    def incumbent_accuracy(self) -> float | None:
        if not self.labeled_count:
            return None
        return self.incumbent_correct / self.labeled_count

    @property
    def candidate_accuracy(self) -> float | None:
        if not self.labeled_count:
            return None
        return self.candidate_correct / self.labeled_count

    @property
    def accuracy_gain(self) -> float | None:
        """정답이 있을 때만 답할 수 있다."""
        if not self.labeled_count:
            return None
        return (self.candidate_correct - self.incumbent_correct) / self.labeled_count

    @property
    def speedup(self) -> float:
        if self.candidate_p95_ms <= 0:
            return 0.0
        return self.incumbent_p95_ms / self.candidate_p95_ms

    def render(self) -> str:
        # 결과물 id 는 길다. 표가 무너지지 않게 뒤쪽만 남긴다 — 구분되는 부분이 뒤에 있다.
        incumbent = _short(self.incumbent_label)
        candidate = _short(self.candidate_label)
        lines = [
            f"그림자 비교 — {self.window.label} ({self.sample_count:,}건)",
            "-" * 66,
            f"  {'':<14}{incumbent:>22}{candidate:>22}",
            f"  {'p95(ms)':<14}{self.incumbent_p95_ms:>22.3f}{self.candidate_p95_ms:>22.3f}",
        ]
        labels = sorted(set(self.incumbent_mix) | set(self.candidate_mix))
        for label in labels:
            lines.append(
                f"  {label:<14}{self.incumbent_mix.get(label, 0):>22,}"
                f"{self.candidate_mix.get(label, 0):>22,}"
            )
        lines.append("-" * 66)
        lines.append(
            f"  답이 같았던 비율 : {self.agreement_ratio:.1%} "
            f"({self.disagreement_count:,}건에서 갈렸다)"
        )
        if self.has_labels:
            lines.append(
                f"  정답 붙은 {self.labeled_count:,}건 기준 정확도 : "
                f"{self.incumbent_accuracy:.4f} → {self.candidate_accuracy:.4f} "
                f"({self.accuracy_gain:+.4f})"
            )
        else:
            lines.append(
                "  정답이 붙은 표본이 없다 — **다르다**까지만 말할 수 있다"
            )
        return "\n".join(lines)


def _short(label: str, width: int = 20) -> str:
    """긴 결과물 id 의 뒤쪽만 남긴다. 구분되는 부분은 대개 뒤에 있다."""
    return label if len(label) <= width else "…" + label[-(width - 1) :]


@dataclass(frozen=True, slots=True)
class PromotionVerdict:
    """새 모델로 갈아탈 것인가."""

    run: ShadowRun
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> Verdict:
        return derive_verdict(self.findings)

    @property
    def promote(self) -> bool:
        return self.verdict is not Verdict.FAILED

    def render(self) -> str:
        lines = [
            self.run.render(),
            "",
            f"승격 판정: {self.verdict.value}",
        ]
        if self.findings:
            lines += [f"  - {f.describe()}" for f in self.findings]
        else:
            lines.append("  걸리는 것이 없다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """무엇이 확인돼야 갈아탈 수 있는가."""

    min_sample_count: int = 500
    min_labeled_count: int = 100
    min_accuracy_gain: float = 0.0
    max_slowdown_ratio: float = 1.2
    min_agreement_ratio: float = 0.0
    """0 이면 요구하지 않는다. 두 모델이 다른 것은 그 자체로 문제가 아니다."""

    require_labels: bool = True
    """정답 없이 승격을 허용할 것인가. **기본은 허용하지 않는다.**"""

    def evaluate(self, run: ShadowRun) -> PromotionVerdict:
        findings: list[Finding] = []

        if run.sample_count < self.min_sample_count:
            findings.append(
                Finding(
                    code="SHADOW_TOO_FEW_SAMPLES",
                    message=(
                        f"{run.sample_count}건으로 판단하고 있다. "
                        "현장 조건은 시간대마다 다르다 — 한나절은 돌려 봐야 한다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=run.window.label,
                    measured=float(run.sample_count),
                    threshold=float(self.min_sample_count),
                )
            )

        if self.require_labels and not run.has_labels:
            findings.append(
                Finding(
                    code="SHADOW_NO_LABELS",
                    message=(
                        "정답이 붙은 표본이 하나도 없다. "
                        "두 모델이 **다르다**는 것만 확인했지 "
                        "어느 쪽이 **낫다**는 것은 확인하지 못했다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=run.window.label,
                )
            )
        elif run.has_labels and run.labeled_count < self.min_labeled_count:
            findings.append(
                Finding(
                    code="SHADOW_TOO_FEW_LABELS",
                    message=(
                        f"정답이 {run.labeled_count}건뿐이다. "
                        "이 숫자로 계산한 정확도 차이는 다음에 재면 뒤집힐 수 있다."
                    ),
                    severity=Severity.WARNING,
                    subject=run.window.label,
                    measured=float(run.labeled_count),
                    threshold=float(self.min_labeled_count),
                )
            )

        gain = run.accuracy_gain
        if gain is not None and gain < self.min_accuracy_gain:
            findings.append(
                Finding(
                    code="SHADOW_NOT_BETTER",
                    message=(
                        f"현장 데이터에서 새 모델이 더 낫지 않다 ({gain:+.4f}). "
                        "오프라인 평가에서 좋았던 것은 그 데이터에서 좋았다는 뜻이다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=run.candidate_label,
                    measured=gain,
                    threshold=self.min_accuracy_gain,
                )
            )

        if run.incumbent_p95_ms > 0:
            slowdown = run.candidate_p95_ms / run.incumbent_p95_ms
            if slowdown > self.max_slowdown_ratio:
                findings.append(
                    Finding(
                        code="SHADOW_SLOWER",
                        message=(
                            f"새 모델이 {slowdown:.2f}배 느리다. "
                            "정확도를 조금 얻고 사이클 타임을 잃는 거래일 수 있다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=run.candidate_label,
                        measured=slowdown,
                        threshold=self.max_slowdown_ratio,
                    )
                )

        if (
            self.min_agreement_ratio > 0
            and run.agreement_ratio < self.min_agreement_ratio
        ):
            findings.append(
                Finding(
                    code="SHADOW_HIGH_DISAGREEMENT",
                    message=(
                        f"두 모델의 답이 {run.disagreement_count:,}건에서 갈렸다. "
                        "그 자체가 나쁜 것은 아니지만, 갈린 건들을 사람이 봐야 한다."
                    ),
                    severity=Severity.WARNING,
                    subject=run.window.label,
                    measured=run.agreement_ratio,
                    threshold=self.min_agreement_ratio,
                )
            )

        return PromotionVerdict(run=run, findings=tuple(findings))
