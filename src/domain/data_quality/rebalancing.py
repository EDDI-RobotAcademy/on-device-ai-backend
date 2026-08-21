"""불균형을 줄이는 방법마다 잃는 것이 다르다. (실습 2-11)

실습 2-5 에서 "불균형 데이터는 AI의 판단을 왜곡한다"를 **측정**했다.
그 다음 질문은 하나다.

    "그래서 어떻게 하죠?"

인터넷 답변은 대개 "오버샘플링 하세요"다. 그런데 방법마다 대가가 다르다.

    복제      표본이 늘어난 것처럼 보이지만 **새 정보는 0이다.**
              모델은 그 몇 장을 통째로 외운다.
    버리기    비율은 맞지만 **정상의 다양성을 버린다.**
              현장의 정상은 한 가지가 아니다.
    가중치    데이터를 건드리지 않는다. 대신 **없는 것은 여전히 없다.**
              40장으로 배운 것은 여전히 40장짜리다.
    합성      없던 표본을 만든다. 그런데 **현장에 없는 패턴도 만든다.**

그리고 어느 방법을 쓰든 반드시 지켜야 하는 규칙이 하나 있다.

    **분할한 뒤에, train 에만 적용한다.**

먼저 리샘플링하고 나중에 나누면 같은 표본이 train 과 test 양쪽에 들어간다.
그러면 정확도가 올라간다 — 시험지가 유출되었기 때문에.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class RebalancingStrategy(Enum):
    NONE = "NONE"
    """아무것도 하지 않는다. 기준선이다 — 이것 없이는 비교가 안 된다."""

    OVERSAMPLE = "OVERSAMPLE"
    """적은 쪽을 복제한다."""

    UNDERSAMPLE = "UNDERSAMPLE"
    """많은 쪽을 버린다."""

    CLASS_WEIGHT = "CLASS_WEIGHT"
    """데이터는 그대로 두고 손실 함수에 가중치를 건다."""

    SYNTHETIC = "SYNTHETIC"
    """이웃 사이를 보간해 없던 표본을 만든다 (SMOTE 계열)."""


@dataclass(frozen=True, slots=True)
class RebalancingPlan:
    """무엇을 어떻게 맞출 것인가."""

    strategy: RebalancingStrategy
    target_ratio: float = 1.0
    """가장 많은 클래스 대비 목표 비율. 1.0 이면 완전히 같게 맞춘다."""

    applied_after_split: bool = True
    """분할한 뒤에 적용했는가. **False 면 시험지가 유출된다.**"""

    def __post_init__(self) -> None:
        if not 0.0 < self.target_ratio <= 1.0:
            raise InvariantViolation(
                "목표 비율은 0 초과 1 이하여야 한다.", subject="target_ratio"
            )

    def describe(self) -> str:
        when = "분할 후" if self.applied_after_split else "분할 전"
        return f"{self.strategy.value} (목표 비율 {self.target_ratio:g}, {when})"


@dataclass(frozen=True, slots=True)
class RebalancingOutcome:
    """실제로 적용해 본 결과. Infrastructure 가 채운다."""

    strategy: RebalancingStrategy
    before: Mapping[str, int]
    after: Mapping[str, int]
    duplicated_rows: int = 0
    """복제로 늘어난 행 수. 새 정보는 0이다."""

    discarded_rows: int = 0
    """버려진 행 수. 되돌릴 수 없다."""

    synthesized_rows: int = 0
    """만들어 낸 행 수. 현장에서 관측된 적 없는 값이다."""

    distinct_minority_samples: int = 0
    """가장 적은 클래스의 **서로 다른** 원본 표본 수.

    복제를 아무리 해도 이 숫자는 늘지 않는다. **여기가 핵심이다.**
    """

    def __post_init__(self) -> None:
        for name in ("duplicated_rows", "discarded_rows", "synthesized_rows"):
            if getattr(self, name) < 0:
                raise InvariantViolation("행 수는 음수일 수 없다.", subject=name)

    @staticmethod
    def _ratio(counts: Mapping[str, int]) -> float:
        present = [c for c in counts.values() if c > 0]
        if len(present) < 2:
            return 1.0
        return max(present) / min(present)

    @property
    def imbalance_before(self) -> float:
        return self._ratio(self.before)

    @property
    def imbalance_after(self) -> float:
        return self._ratio(self.after)

    @property
    def total_before(self) -> int:
        return sum(self.before.values())

    @property
    def total_after(self) -> int:
        return sum(self.after.values())

    @property
    def information_gain(self) -> int:
        """정말로 늘어난 **서로 다른** 원본 표본 수. 대개 0이다."""
        return 0 if self.strategy in (
            RebalancingStrategy.NONE,
            RebalancingStrategy.OVERSAMPLE,
            RebalancingStrategy.UNDERSAMPLE,
            RebalancingStrategy.CLASS_WEIGHT,
        ) else self.synthesized_rows

    def describe(self) -> str:
        return (
            f"{self.strategy.value:<14}"
            f"{self.total_before:>8,} → {self.total_after:>8,}행  "
            f"불균형 {self.imbalance_before:>5.1f}배 → {self.imbalance_after:>5.1f}배  "
            f"복제 {self.duplicated_rows:>6,} 버림 {self.discarded_rows:>6,} "
            f"합성 {self.synthesized_rows:>6,}"
        )


@dataclass(frozen=True, slots=True)
class RebalancingPolicy:
    """이 방법을 써도 되는가. (실습 2-11)

    **어떤 것도 무조건 옳지 않다.** 그래서 이 Policy 는 순위를 매기지 않는다.
    무엇을 잃는지 적어 줄 뿐이다.
    """

    max_duplication_ratio: float = 3.0
    """원본 대비 이 배수를 넘게 복제하면 그 몇 장을 외운다."""

    max_discard_ratio: float = 0.5
    min_distinct_minority: int = 50

    def inspect(
        self, plan: RebalancingPlan, outcome: RebalancingOutcome
    ) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        if not plan.applied_after_split:
            findings.append(
                Finding(
                    code="REBALANCE_BEFORE_SPLIT",
                    message=(
                        "분할하기 **전에** 리샘플링했다. "
                        "복제된 표본이 train 과 test 양쪽에 들어간다 — "
                        "**정확도는 올라가고 현장에서는 그대로다.** "
                        "이 실수는 지표가 좋아지기 때문에 아무도 의심하지 않는다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=plan.strategy.value,
                )
            )

        if outcome.duplicated_rows:
            base = max(outcome.distinct_minority_samples, 1)
            factor = (outcome.duplicated_rows + base) / base
            if factor > self.max_duplication_ratio:
                findings.append(
                    Finding(
                        code="REBALANCE_HEAVY_DUPLICATION",
                        message=(
                            f"같은 표본을 평균 {factor:.1f}배로 늘렸다. "
                            f"**서로 다른 표본은 여전히 "
                            f"{outcome.distinct_minority_samples}개뿐이다** — "
                            "모델은 그 몇 개를 외운다. "
                            "학습 정확도만 올라가고 검증은 따라오지 않는다 (실습 3-7)."
                        ),
                        severity=Severity.WARNING,
                        subject=plan.strategy.value,
                        measured=factor,
                        threshold=self.max_duplication_ratio,
                    )
                )

        if outcome.total_before:
            discard_ratio = outcome.discarded_rows / outcome.total_before
            if discard_ratio > self.max_discard_ratio:
                findings.append(
                    Finding(
                        code="REBALANCE_HEAVY_DISCARD",
                        message=(
                            f"{outcome.discarded_rows:,}행({discard_ratio:.0%})을 버렸다. "
                            "**현장의 '정상'은 한 가지가 아니다** — "
                            "버린 것 중에 다음 달에 필요할 정상이 있다. "
                            "그리고 버린 데이터는 되돌릴 수 없다."
                        ),
                        severity=Severity.WARNING,
                        subject=plan.strategy.value,
                        measured=discard_ratio,
                        threshold=self.max_discard_ratio,
                    )
                )

        if outcome.synthesized_rows:
            findings.append(
                Finding(
                    code="REBALANCE_SYNTHETIC",
                    message=(
                        f"{outcome.synthesized_rows:,}행을 만들어 냈다. "
                        "**현장에서 관측된 적 없는 값이다.** "
                        "물리적으로 불가능한 조합이 섞이면 "
                        "모델은 있지도 않은 패턴을 배운다 (실습 2-3 의 물리 범위)."
                    ),
                    severity=Severity.WARNING,
                    subject=plan.strategy.value,
                    measured=float(outcome.synthesized_rows),
                )
            )

        if 0 < outcome.distinct_minority_samples < self.min_distinct_minority:
            severity = (
                Severity.CRITICAL
                if plan.strategy is not RebalancingStrategy.NONE
                else Severity.WARNING
            )
            findings.append(
                Finding(
                    code="REBALANCE_CANNOT_FIX_SHORTAGE",
                    message=(
                        f"가장 적은 클래스의 서로 다른 표본이 "
                        f"{outcome.distinct_minority_samples}개다. "
                        "**어떤 리샘플링도 이 숫자를 늘리지 못한다.** "
                        "비율을 맞춰서 지표가 좋아 보이면 "
                        "'데이터를 더 모아야 한다'는 결론이 가려진다."
                    ),
                    severity=severity,
                    subject=plan.strategy.value,
                    measured=float(outcome.distinct_minority_samples),
                    threshold=float(self.min_distinct_minority),
                )
            )

        if (
            plan.strategy is RebalancingStrategy.CLASS_WEIGHT
            and outcome.total_after == outcome.total_before
        ):
            findings.append(
                Finding(
                    code="REBALANCE_WEIGHT_ONLY",
                    message=(
                        "가중치는 데이터를 건드리지 않았다. "
                        "**시험지 유출도 없고 버린 것도 없다** — "
                        "그래서 대개 여기서 시작한다. "
                        "다만 없는 것은 여전히 없다."
                    ),
                    severity=Severity.INFO,
                    subject=plan.strategy.value,
                )
            )

        return tuple(findings)


@dataclass(frozen=True, slots=True)
class RebalancingComparison:
    """여러 전략을 나란히 놓은 표. (실습 2-11)"""

    rows: tuple[tuple[RebalancingPlan, RebalancingOutcome, tuple[Finding, ...]], ...] = (
        field(default_factory=tuple)
    )

    @property
    def safe_strategies(self) -> tuple[RebalancingStrategy, ...]:
        """막는 소견이 없는 전략들."""
        return tuple(
            plan.strategy
            for plan, _, findings in self.rows
            if not any(f.is_blocking for f in findings)
        )

    def render(self) -> str:
        lines = ["[불균형을 줄이는 방법과 그 대가]"]
        for plan, outcome, findings in self.rows:
            lines.append(f"  {outcome.describe()}")
            for finding in findings:
                lines.append(f"      {finding.describe()}")
        lines.append("")
        lines.append(
            "  막는 소견이 없는 전략: "
            + (", ".join(s.value for s in self.safe_strategies) or "없음")
        )
        return "\n".join(lines)
