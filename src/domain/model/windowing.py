"""센서의 시간을 자르는 규칙. (실습 3-4)

실습 1-7 에서 "몇 초를 한 덩어리로 볼 것인가"를 정했다.
여기서는 그 창을 실제로 자를 때 생기는 두 가지 문제를 다룬다.

    1. **창의 라벨은 무엇인가?**
       30 표본짜리 창 안에 NORMAL 25개와 FAULT 5개가 있다면 이 창은 무엇인가?
       다수결로 하면 짧은 사고가 전부 사라진다. 하나라도 있으면 FAULT 로 하면
       거의 모든 창이 FAULT 가 된다. 기준은 현장이 정해야 한다.

    2. **창이 겹치면 분할이 새어 나간다.**
       stride 가 length 보다 작으면 인접한 두 창이 표본을 공유한다.
       그 둘이 train 과 test 로 갈리면, 시험 문제의 일부를 이미 본 것이다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class WindowLabelPolicy:
    """창 하나에 어떤 라벨을 붙일 것인가.

    우선순위 규칙이다. 위에 있는 라벨이 임계 비율을 넘으면 그것으로 확정한다.
    "짧게 스쳐도 사고는 사고다"라는 현장 판단을 비율로 옮긴 것이다.
    """

    priority: tuple[tuple[str, float], ...]
    """(라벨, 최소 비율). 순서대로 검사한다."""

    default_label: str

    def __post_init__(self) -> None:
        if not self.priority:
            raise InvariantViolation(
                "우선순위 규칙이 비어 있다. 창의 라벨을 정할 방법이 없다.",
                subject="priority",
            )
        if not self.default_label.strip():
            raise InvariantViolation(
                "어느 것도 해당하지 않을 때의 라벨이 없다.", subject="default_label"
            )
        for label, ratio in self.priority:
            if not label.strip():
                raise InvariantViolation("라벨 이름이 비어 있다.", subject="priority")
            if not 0.0 < ratio <= 1.0:
                raise InvariantViolation(
                    f"'{label}' 의 임계 비율은 0 초과 1 이하여야 한다. (받은 값 {ratio})",
                    subject=label,
                )

    def label_for(self, counts: Mapping[str, int]) -> str:
        """창 안의 라벨 개수 분포로부터 창의 라벨을 결정한다."""
        total = sum(counts.values())
        if total == 0:
            return self.default_label
        for label, threshold in self.priority:
            if counts.get(label, 0) / total >= threshold:
                return label
        return self.default_label

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.priority) + (self.default_label,)

    def describe(self) -> str:
        rules = " → ".join(
            f"{label} {ratio:.0%} 이상" for label, ratio in self.priority
        )
        return f"{rules} → 그 외 {self.default_label}"


@dataclass(frozen=True, slots=True)
class WindowingPlan:
    """창을 어떻게 자를 것인가. 학습·평가·배포에서 **같은 값**이어야 한다.

    window_length 는 모델 입력의 시간 축과 반드시 같다.
    다르면 학습은 그냥 돌아가고, 배포할 때 모양이 안 맞아 터진다.
    """

    window_length: int
    stride: int
    label_policy: WindowLabelPolicy

    def __post_init__(self) -> None:
        if self.window_length < 1:
            raise InvariantViolation(
                "창 길이는 1 이상이어야 한다.", subject="window_length"
            )
        if self.stride < 1:
            raise InvariantViolation("stride 는 1 이상이어야 한다.", subject="stride")
        if self.stride > self.window_length:
            raise InvariantViolation(
                f"stride({self.stride}) 가 창 길이({self.window_length}) 보다 크면 "
                "표본 사이가 통째로 버려진다.",
                subject="stride",
            )

    @property
    def overlap_ratio(self) -> float:
        return 1.0 - self.stride / self.window_length

    def describe(self) -> str:
        return (
            f"length={self.window_length} stride={self.stride} "
            f"(겹침 {self.overlap_ratio:.0%}) / {self.label_policy.describe()}"
        )


@dataclass(frozen=True, slots=True)
class WindowingSummary:
    """실제로 잘라 본 결과. Infrastructure 가 채운다."""

    source_row_count: int
    window_length: int
    stride: int
    window_count: int
    label_counts: Mapping[str, int] = field(default_factory=dict)
    shared_sample_count: int = 0
    """인접 창끼리 공유하는 표본 수 (겹침 때문에 생긴다)."""

    def __post_init__(self) -> None:
        for name in ("source_row_count", "window_length", "stride", "window_count"):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)

    @property
    def overlap_ratio(self) -> float:
        if self.window_length == 0:
            return 0.0
        return max(0.0, 1.0 - self.stride / self.window_length)

    @property
    def coverage_ratio(self) -> float:
        """원본 표본 중 어느 하나의 창에라도 **실제로 들어간** 비율.

        창이 이어진 구간의 길이가 아니라, 창 안에 들어간 표본 수로 센다.
        stride 가 창보다 크면 창과 창 사이가 통째로 버려지는데,
        구간 길이로 재면 그 사실이 사라진다.
        """
        if self.source_row_count == 0 or self.window_count == 0:
            return 0.0
        if self.stride >= self.window_length:
            covered = self.window_count * self.window_length
        else:
            covered = (self.window_count - 1) * self.stride + self.window_length
        return min(covered, self.source_row_count) / self.source_row_count

    @property
    def effective_sample_count(self) -> int:
        """겹침을 걷어낸, 서로 독립인 창의 수.

        "표본이 862개"라고 말할 때 실제로 독립인 것은 이 숫자다.
        """
        if self.window_length == 0 or self.stride == 0:
            return self.window_count
        if self.stride >= self.window_length:
            return self.window_count
        return max(1, (self.window_count * self.stride) // self.window_length)

    def describe(self) -> str:
        classes = ", ".join(
            f"{name} {count}" for name, count in sorted(self.label_counts.items())
        )
        return (
            f"{self.source_row_count:,}행 → 창 {self.window_count:,}개 "
            f"(length={self.window_length}, stride={self.stride}, "
            f"겹침 {self.overlap_ratio:.0%})\n"
            f"  독립 표본 환산 {self.effective_sample_count:,}개 / 라벨 {classes}"
        )


@dataclass(frozen=True, slots=True)
class WindowingPolicy:
    """창 설계가 학습을 왜곡하지 않는지에 대한 기준."""

    max_overlap_ratio: float = 0.5
    min_coverage_ratio: float = 0.9
    min_windows_per_class: int = 10
    require_disjoint_split: bool = True
    """분할 경계에서 창이 겹치면 안 된다."""

    def inspect(self, summary: WindowingSummary) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        if summary.overlap_ratio > self.max_overlap_ratio:
            findings.append(
                Finding(
                    code="WINDOW_OVERLAP_HIGH",
                    message=(
                        f"창이 {summary.overlap_ratio:.0%} 겹친다. "
                        f"창 {summary.window_count:,}개처럼 보이지만 독립 표본은 "
                        f"{summary.effective_sample_count:,}개다."
                    ),
                    severity=Severity.WARNING,
                    subject="stride",
                    measured=summary.overlap_ratio,
                    threshold=self.max_overlap_ratio,
                )
            )

        if summary.coverage_ratio < self.min_coverage_ratio:
            findings.append(
                Finding(
                    code="WINDOW_COVERAGE_LOW",
                    message=(
                        "원본의 상당 부분이 어떤 창에도 들어가지 않았다. "
                        "stride 가 length 보다 커서 사이가 버려졌을 수 있다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="coverage",
                    measured=summary.coverage_ratio,
                    threshold=self.min_coverage_ratio,
                )
            )

        for label, count in sorted(summary.label_counts.items()):
            if 0 < count < self.min_windows_per_class:
                findings.append(
                    Finding(
                        code="WINDOW_CLASS_TOO_FEW",
                        message=(
                            f"'{label}' 창이 {count}개뿐이다. "
                            "원본에는 표본이 있어도 창으로 자르면 사라질 수 있다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=label,
                        measured=float(count),
                        threshold=float(self.min_windows_per_class),
                    )
                )

        return tuple(findings)
