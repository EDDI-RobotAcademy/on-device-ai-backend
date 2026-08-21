"""정확도와 Latency를 직접 싸움 붙여라. (실습 4-8)

최적화는 **교환**이다. 무엇을 주고 무엇을 받았는지 한 표에 놓지 않으면
"빨라졌습니다"만 남고 무엇을 잃었는지는 아무도 모른다.

그리고 이 표에서 자주 발견되는 사실:
    **더 작게 만들었는데 더 느려진 후보가 있다.**
    양자화한 값을 다시 풀어 쓰는 비용이 계산 이득보다 클 때 그렇다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.optimization.benchmark import BenchmarkResult
from domain.optimization.conversion import ConversionRecord
from domain.optimization.runtime import ModelArtifact
from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class ArtifactAccuracy:
    """변환된 결과물을 같은 평가 집합으로 다시 잰 정확도.

    모듈 3 의 ConfusionMatrix 를 그대로 들고 오지 않는다.
    이 Context 가 알아야 하는 것은 **숫자 몇 개**뿐이다.
    """

    accuracy: float
    macro_recall: float
    macro_f1: float = 0.0
    """재현율과 정밀도를 함께 본 값. (실습 4-8)

    양자화는 **재현율은 지키면서 정밀도만 무너뜨리는** 방식으로 망가질 수 있다.
    재현율만 보면 그게 안 보인다 — 오탐이 늘어난 것은 현장에서 알람 폭주로 나타난다.
    """

    per_class_recall: Mapping[str, float] = field(default_factory=dict)

    predicted_mix: Mapping[str, float] = field(default_factory=dict)
    """이 결과물이 각 클래스를 **답한** 비율. 재현율이 아니다.

    모듈 5 가 현장 예측 분포를 여기에 견준다 (실습 5-6).
    현장에서 셀 수 있는 것은 정답이 아니라 예측이므로,
    비교 기준도 정답 기준(재현율)이 아니라 예측 기준이어야 한다.
    """

    def __post_init__(self) -> None:
        for name in ("accuracy", "macro_recall", "macro_f1"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 이어야 한다.", subject=name)

    def drop_from(self, baseline: ArtifactAccuracy) -> float:
        return baseline.accuracy - self.accuracy

    def macro_recall_drop_from(self, baseline: ArtifactAccuracy) -> float:
        return baseline.macro_recall - self.macro_recall

    def worst_class_drop_from(
        self, baseline: ArtifactAccuracy
    ) -> tuple[str | None, float]:
        """가장 크게 떨어진 클래스. **평균은 이것을 숨긴다.**"""
        worst_label, worst_drop = None, 0.0
        for label, recall in baseline.per_class_recall.items():
            drop = recall - self.per_class_recall.get(label, 0.0)
            if drop > worst_drop:
                worst_label, worst_drop = label, drop
        return worst_label, worst_drop


@dataclass(frozen=True, slots=True)
class OptimizationCandidate:
    """후보 하나. 결과물 + 변환 기록 + 속도 + 정확도가 한 묶음이다."""

    artifact: ModelArtifact
    conversion: ConversionRecord
    benchmark: BenchmarkResult
    accuracy: ArtifactAccuracy

    @property
    def label(self) -> str:
        return self.artifact.label

    def speedup_over(self, baseline: OptimizationCandidate) -> float:
        return self.benchmark.speedup_over(baseline.benchmark)

    def size_ratio_to(self, baseline: OptimizationCandidate) -> float:
        return self.artifact.size_ratio_to(baseline.artifact)

    def accuracy_drop_from(self, baseline: OptimizationCandidate) -> float:
        return self.accuracy.drop_from(baseline.accuracy)


@dataclass(frozen=True, slots=True)
class TradeoffTable:
    """후보들을 한 표에 놓는다."""

    baseline: OptimizationCandidate
    candidates: tuple[OptimizationCandidate, ...] = field(default_factory=tuple)

    @property
    def all_candidates(self) -> tuple[OptimizationCandidate, ...]:
        return (self.baseline, *self.candidates)

    @property
    def fastest(self) -> OptimizationCandidate:
        return min(self.all_candidates, key=lambda c: c.benchmark.p95_ms)

    @property
    def smallest(self) -> OptimizationCandidate:
        return min(self.all_candidates, key=lambda c: c.artifact.size_bytes)

    @property
    def most_accurate(self) -> OptimizationCandidate:
        return max(self.all_candidates, key=lambda c: c.accuracy.macro_recall)

    @property
    def slower_than_baseline(self) -> tuple[OptimizationCandidate, ...]:
        """최적화했는데 오히려 느려진 후보들.

        드물지 않다. 특히 작은 모델에서 INT8 은 자주 여기 들어온다.
        """
        return tuple(
            c
            for c in self.candidates
            if c.benchmark.p95_ms > self.baseline.benchmark.p95_ms
        )

    @property
    def pareto_front(self) -> tuple[OptimizationCandidate, ...]:
        """다른 후보에게 **모든 면에서** 지지 않는 후보들.

        더 빠르면서 더 정확한 후보가 있으면 그 후보는 볼 필요가 없다.
        """
        front: list[OptimizationCandidate] = []
        for candidate in self.all_candidates:
            dominated = any(
                other is not candidate
                and other.benchmark.p95_ms <= candidate.benchmark.p95_ms
                and other.accuracy.macro_recall >= candidate.accuracy.macro_recall
                and (
                    other.benchmark.p95_ms < candidate.benchmark.p95_ms
                    or other.accuracy.macro_recall > candidate.accuracy.macro_recall
                )
                for other in self.all_candidates
            )
            if not dominated:
                front.append(candidate)
        return tuple(front)

    def render(self) -> str:
        lines = [
            f"{'후보':<18}{'크기':>10}{'p95(ms)':>10}{'배속':>8}"
            f"{'정확도':>9}{'macroR':>9}{'macroF1':>9}{'Δ정확도':>10}",
            "-" * 85,
        ]
        for candidate in self.all_candidates:
            is_baseline = candidate is self.baseline
            speedup = "—" if is_baseline else f"{candidate.speedup_over(self.baseline):.2f}x"
            drop = (
                "—"
                if is_baseline
                else f"{-candidate.accuracy_drop_from(self.baseline):+.4f}"
            )
            mark = "  (기준)" if is_baseline else ""
            lines.append(
                f"{candidate.label:<18}{candidate.artifact.size_kib:>9.1f}K"
                f"{candidate.benchmark.p95_ms:>10.4f}{speedup:>8}"
                f"{candidate.accuracy.accuracy:>9.4f}"
                f"{candidate.accuracy.macro_recall:>9.4f}"
                f"{candidate.accuracy.macro_f1:>9.4f}{drop:>10}{mark}"
            )
        return "\n".join(lines)
