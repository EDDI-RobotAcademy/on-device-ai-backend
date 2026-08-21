"""Accuracy 뒤에 숨어 있는 실패. (실습 3-9)

불량률 1.7% 인 라인에서 정확도 98% 짜리 모델은 아무것도 안 하는 모델일 수 있다.
실습 2-5 에서 이미 이 숫자를 봤다. 여기서는 **학습된 모델이 실제로 그런지** 확인한다.

정확도 하나만 보면 이 실패가 안 보인다. 봐야 하는 것은 혼동 행렬이다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """무엇을 무엇으로 착각했는가.

    labels[i] 를 labels[j] 로 예측한 횟수가 counts[i][j] 다.
    """

    labels: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        size = len(self.labels)
        if size < 2:
            raise InvariantViolation("클래스가 둘 미만이다.", subject="labels")
        if len(set(self.labels)) != size:
            raise InvariantViolation("라벨 이름이 중복된다.", subject="labels")
        if len(self.counts) != size or any(len(row) != size for row in self.counts):
            raise InvariantViolation(
                f"혼동 행렬은 {size}×{size} 여야 한다.", subject="counts"
            )
        if any(value < 0 for row in self.counts for value in row):
            raise InvariantViolation("개수는 음수일 수 없다.", subject="counts")

    @classmethod
    def from_pairs(
        cls, labels: Sequence[str], pairs: Sequence[tuple[str, str]]
    ) -> ConfusionMatrix:
        """(정답, 예측) 쌍의 나열로부터 만든다."""
        index = {label: i for i, label in enumerate(labels)}
        size = len(labels)
        grid = [[0] * size for _ in range(size)]
        for actual, predicted in pairs:
            if actual not in index or predicted not in index:
                raise InvariantViolation(
                    f"모르는 라벨이 있다: {actual!r} / {predicted!r}", subject="pairs"
                )
            grid[index[actual]][index[predicted]] += 1
        return cls(labels=tuple(labels), counts=tuple(tuple(row) for row in grid))

    @property
    def total(self) -> int:
        return sum(value for row in self.counts for value in row)

    def support_of(self, label: str) -> int:
        i = self.labels.index(label)
        return sum(self.counts[i])

    def predicted_count_of(self, label: str) -> int:
        j = self.labels.index(label)
        return sum(row[j] for row in self.counts)

    def correct_of(self, label: str) -> int:
        i = self.labels.index(label)
        return self.counts[i][i]

    def count_of(self, actual: str, predicted: str) -> int:
        """actual 을 predicted 로 예측한 횟수. (실습 3-13)

        칸 하나를 이름으로 꺼낸다. 행렬을 다시 접을 때 쓴다.
        """
        return self.counts[self.labels.index(actual)][self.labels.index(predicted)]

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return sum(self.counts[i][i] for i in range(len(self.labels))) / self.total

    def recall_of(self, label: str) -> float:
        """이 클래스를 **놓치지 않았는가.** 현장에서 가장 중요한 숫자다."""
        support = self.support_of(label)
        return self.correct_of(label) / support if support else 0.0

    def precision_of(self, label: str) -> float:
        """이 클래스라고 한 것이 맞았는가. 헛경보의 반대말이다."""
        predicted = self.predicted_count_of(label)
        return self.correct_of(label) / predicted if predicted else 0.0

    def f1_of(self, label: str) -> float:
        precision, recall = self.precision_of(label), self.recall_of(label)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def macro_recall(self) -> float:
        """클래스마다 같은 무게로 센 재현율. 불균형에 속지 않는 지표다."""
        present = [label for label in self.labels if self.support_of(label) > 0]
        if not present:
            return 0.0
        return sum(self.recall_of(label) for label in present) / len(present)

    @property
    def macro_f1(self) -> float:
        present = [label for label in self.labels if self.support_of(label) > 0]
        if not present:
            return 0.0
        return sum(self.f1_of(label) for label in present) / len(present)

    @property
    def baseline_accuracy(self) -> float:
        """전부 다수 클래스라고 찍었을 때의 정확도."""
        if self.total == 0:
            return 0.0
        return max(self.support_of(label) for label in self.labels) / self.total

    @property
    def never_predicted(self) -> tuple[str, ...]:
        """모델이 **한 번도 예측하지 않은** 클래스."""
        return tuple(
            label
            for label in self.labels
            if self.support_of(label) > 0 and self.predicted_count_of(label) == 0
        )

    def render(self) -> str:
        width = max(len(label) for label in self.labels) + 2
        header = " " * (width + 8) + "".join(f"{label:>{width}}" for label in self.labels)
        lines = ["실제 \\ 예측", header, "-" * len(header)]
        for i, label in enumerate(self.labels):
            row = "".join(f"{value:>{width}}" for value in self.counts[i])
            lines.append(f"{label:<{width + 8}}{row}")
        lines.append("")
        lines.append(
            f"{'class':<{width}}{'support':>9}{'recall':>9}"
            f"{'precision':>11}{'f1':>8}"
        )
        for label in self.labels:
            lines.append(
                f"{label:<{width}}{self.support_of(label):>9}"
                f"{self.recall_of(label):>9.3f}{self.precision_of(label):>11.3f}"
                f"{self.f1_of(label):>8.3f}"
            )
        lines.append(
            f"\n정확도 {self.accuracy:.3f} / macro recall {self.macro_recall:.3f} "
            f"/ baseline {self.baseline_accuracy:.3f}"
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """한 분할에 대한 평가 결과."""

    split: str
    matrix: ConfusionMatrix
    loss: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0

    def __post_init__(self) -> None:
        if self.loss < 0:
            raise InvariantViolation("loss 는 음수일 수 없다.", subject="loss")
        if self.latency_ms_p95 < 0 or self.latency_ms_p50 < 0:
            raise InvariantViolation("지연시간은 음수일 수 없다.", subject="latency")

    @property
    def accuracy(self) -> float:
        return self.matrix.accuracy

    @property
    def macro_recall(self) -> float:
        return self.matrix.macro_recall


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    """정확도 뒤를 보는 기준."""

    min_accuracy_over_baseline: float = 0.03
    min_recall_per_class: float = 0.5
    critical_labels: frozenset[str] = field(default_factory=frozenset)
    """놓치면 안 되는 클래스. 여기 있는 클래스는 재현율 기준이 더 세다."""

    min_critical_recall: float = 0.7
    max_precision_recall_gap: float = 0.5

    def inspect(self, result: EvaluationResult) -> tuple[Finding, ...]:
        matrix = result.matrix
        findings: list[Finding] = []

        margin = matrix.accuracy - matrix.baseline_accuracy
        if margin < self.min_accuracy_over_baseline:
            findings.append(
                Finding(
                    code="EVAL_NO_BETTER_THAN_BASELINE",
                    message=(
                        f"정확도 {matrix.accuracy:.1%} 인데 "
                        f"다수 클래스만 찍어도 {matrix.baseline_accuracy:.1%} 다. "
                        "모델이 실제로 벌어들인 것은 거의 없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="accuracy",
                    measured=margin,
                    threshold=self.min_accuracy_over_baseline,
                )
            )

        for label in matrix.never_predicted:
            findings.append(
                Finding(
                    code="EVAL_CLASS_NEVER_PREDICTED",
                    message=(
                        f"'{label}' 를 한 번도 예측하지 않았다. "
                        "이 클래스에 대해서는 모델이 존재하지 않는 것과 같다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=label,
                    measured=0.0,
                    threshold=1.0,
                )
            )

        for label in matrix.labels:
            if matrix.support_of(label) == 0:
                continue
            recall = matrix.recall_of(label)
            is_critical = label in self.critical_labels
            threshold = (
                self.min_critical_recall if is_critical else self.min_recall_per_class
            )
            if recall < threshold:
                findings.append(
                    Finding(
                        code="EVAL_RECALL_TOO_LOW",
                        message=(
                            f"'{label}' 재현율 {recall:.1%} — "
                            f"{matrix.support_of(label)} 건 중 "
                            f"{matrix.support_of(label) - matrix.correct_of(label)} 건을 놓쳤다."
                            + ("  (놓치면 안 되는 클래스다)" if is_critical else "")
                        ),
                        severity=Severity.CRITICAL
                        if is_critical
                        else Severity.WARNING,
                        subject=label,
                        measured=recall,
                        threshold=threshold,
                    )
                )

            gap = abs(matrix.precision_of(label) - recall)
            if matrix.predicted_count_of(label) > 0 and gap > self.max_precision_recall_gap:
                findings.append(
                    Finding(
                        code="EVAL_PRECISION_RECALL_IMBALANCE",
                        message=(
                            f"'{label}' 정밀도 {matrix.precision_of(label):.2f} vs "
                            f"재현율 {recall:.2f}. 한쪽으로 치우친 모델이다."
                        ),
                        severity=Severity.WARNING,
                        subject=label,
                        measured=gap,
                        threshold=self.max_precision_recall_gap,
                    )
                )

        return tuple(findings)
