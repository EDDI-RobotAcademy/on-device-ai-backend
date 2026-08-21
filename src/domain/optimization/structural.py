"""구조를 줄이는 것과 숫자를 줄이는 것은 다르다. (실습 4-11)

실습 4-5 ~ 4-7 은 **숫자를 좁혔다.** FP32 → FP16 → INT8.
구조는 그대로다. 층도 채널도 그대로고, 곱셈 횟수도 그대로다.

경량화에는 다른 축이 하나 더 있다. **구조 자체를 줄이는 것.**

    폭 줄이기        채널 수를 절반으로. 파라미터도 곱셈도 같이 준다.
    깊이 줄이기      층을 뺀다. 표현력이 준다.
    가지치기         작은 가중치를 0으로 만든다.

세 축의 성질이 다르다.

    양자화     재학습이 필요 없다. 곱셈 횟수는 **안 줄어든다.**
    구조 축소  **재학습이 반드시 필요하다.** 곱셈 횟수가 실제로 준다.

그리고 가지치기에는 함정이 하나 있다.

    비구조적 가지치기(가중치를 0으로)는
    **파일도 안 줄고 지연시간도 안 준다.**

0도 저장되고 0도 곱해지기 때문이다.
희소 연산을 지원하는 런타임이 있어야 비로소 이득이 생긴다.
이 사실을 모르면 "50% 가지치기 했는데 왜 그대로죠?"에서 하루를 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class ReductionKind(Enum):
    WIDTH = "WIDTH"
    """채널 수를 줄인다. 구조가 바뀌므로 **재학습해야 한다.**"""

    DEPTH = "DEPTH"
    """층을 뺀다. 구조가 바뀌므로 **재학습해야 한다.**"""

    PRUNE_UNSTRUCTURED = "PRUNE_UNSTRUCTURED"
    """작은 가중치를 0으로 만든다. 모양은 그대로다.

    **희소 커널이 없으면 파일도 속도도 그대로다.**
    """

    PRUNE_STRUCTURED = "PRUNE_STRUCTURED"
    """채널을 통째로 0으로 만든다. 실제로 떼어 내면 모양이 줄어든다."""

    @property
    def changes_shape(self) -> bool:
        return self in (ReductionKind.WIDTH, ReductionKind.DEPTH)

    @property
    def requires_retraining(self) -> bool:
        """구조를 건드리면 배운 것이 흐트러진다. 미세조정 없이는 정확도가 무너진다."""
        return True


@dataclass(frozen=True, slots=True)
class StructuralReduction:
    """무엇을 얼마나 줄일 것인가."""

    kind: ReductionKind
    ratio: float
    """줄이는 비율. 0.5 면 절반."""

    fine_tuned: bool = False
    """줄인 뒤에 다시 학습했는가.

    **이 한 줄이 정확도를 가른다.** 그런데 실무에서 가장 자주 빠뜨린다.
    """

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio < 1.0:
            raise InvariantViolation(
                "줄이는 비율은 0 초과 1 미만이어야 한다.", subject="ratio"
            )

    def describe(self) -> str:
        return (
            f"{self.kind.value} {self.ratio:.0%} "
            f"({'미세조정 함' if self.fine_tuned else '미세조정 없음'})"
        )


@dataclass(frozen=True, slots=True)
class StructuralOutcome:
    """실제로 줄여 본 결과. Infrastructure 가 채운다."""

    reduction: StructuralReduction
    label: str

    parameter_count_before: int
    parameter_count_after: int
    """**0이 된 가중치도 여전히 파라미터다.** 비구조적 가지치기에서는 이 값이 안 준다."""

    nonzero_parameter_count: int
    mac_count_before: int
    mac_count_after: int
    size_bytes_before: int
    size_bytes_after: int
    accuracy_before: float
    accuracy_after: float
    macro_recall_after: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "parameter_count_before",
            "parameter_count_after",
            "nonzero_parameter_count",
            "mac_count_before",
            "mac_count_after",
            "size_bytes_before",
            "size_bytes_after",
        ):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)
        for name in ("accuracy_before", "accuracy_after", "macro_recall_after"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 이어야 한다.", subject=name)

    @property
    def sparsity(self) -> float:
        """가중치 중 0인 비율."""
        if self.parameter_count_after == 0:
            return 0.0
        zeros = self.parameter_count_after - self.nonzero_parameter_count
        return zeros / self.parameter_count_after

    @property
    def mac_reduction(self) -> float:
        """곱셈 횟수가 몇 % 줄었는가. **이것이 진짜 속도 이득의 상한이다.**"""
        if self.mac_count_before == 0:
            return 0.0
        return 1.0 - self.mac_count_after / self.mac_count_before

    @property
    def size_reduction(self) -> float:
        if self.size_bytes_before == 0:
            return 0.0
        return 1.0 - self.size_bytes_after / self.size_bytes_before

    @property
    def accuracy_drop(self) -> float:
        return self.accuracy_before - self.accuracy_after

    def describe(self) -> str:
        return (
            f"{self.label:<26}"
            f"params {self.parameter_count_after:>7,} "
            f"(0 아닌 것 {self.nonzero_parameter_count:>7,})  "
            f"MAC {self.mac_reduction:>6.1%}↓  "
            f"크기 {self.size_reduction:>6.1%}↓  "
            f"정확도 {self.accuracy_after:.3f} ({-self.accuracy_drop:+.3f})"
        )


@dataclass(frozen=True, slots=True)
class StructuralPolicy:
    """이 축소를 받아들일 수 있는가. (실습 4-11)"""

    max_accuracy_drop: float = 0.03
    min_mac_reduction: float = 0.10
    """이만큼도 안 줄면 '경량화했다'고 말할 수 없다."""

    def inspect(self, outcome: StructuralOutcome) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        reduction = outcome.reduction

        if not reduction.fine_tuned and reduction.kind.requires_retraining:
            severity = (
                Severity.CRITICAL
                if outcome.accuracy_drop > self.max_accuracy_drop
                else Severity.WARNING
            )
            findings.append(
                Finding(
                    code="STRUCT_NO_FINE_TUNE",
                    message=(
                        "줄인 뒤에 다시 학습하지 않았다. "
                        "**구조를 건드리면 배운 것이 흐트러진다** — "
                        "양자화와 여기서 갈린다. 양자화는 재학습이 필요 없지만, "
                        "구조 축소는 미세조정이 절차의 일부다."
                    ),
                    severity=severity,
                    subject=outcome.label,
                    measured=outcome.accuracy_drop,
                    threshold=self.max_accuracy_drop,
                )
            )

        if outcome.accuracy_drop > self.max_accuracy_drop:
            findings.append(
                Finding(
                    code="STRUCT_ACCURACY_LOST",
                    message=(
                        f"정확도가 {outcome.accuracy_drop:.3f} 떨어졌다 "
                        f"({outcome.accuracy_before:.3f} → {outcome.accuracy_after:.3f}). "
                        "**줄여서 얻은 것보다 잃은 것이 큰지 따져야 한다.**"
                    ),
                    severity=Severity.CRITICAL,
                    subject=outcome.label,
                    measured=outcome.accuracy_drop,
                    threshold=self.max_accuracy_drop,
                )
            )

        if outcome.sparsity > 0.1 and outcome.size_reduction < 0.05:
            findings.append(
                Finding(
                    code="STRUCT_SPARSITY_NOT_REALIZED",
                    message=(
                        f"가중치의 {outcome.sparsity:.0%} 가 0인데 "
                        f"파일은 {outcome.size_reduction:.1%} 밖에 안 줄었다. "
                        "**0도 저장되고 0도 곱해진다.** "
                        "희소 연산을 지원하는 런타임이 없으면 "
                        "비구조적 가지치기는 장부상의 경량화다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=outcome.label,
                    measured=outcome.size_reduction,
                    threshold=0.05,
                )
            )

        if (
            reduction.fine_tuned
            and reduction.kind in (
                ReductionKind.PRUNE_UNSTRUCTURED,
                ReductionKind.PRUNE_STRUCTURED,
            )
            and outcome.sparsity < reduction.ratio * 0.5
        ):
            findings.append(
                Finding(
                    code="STRUCT_PRUNE_UNDONE",
                    message=(
                        f"{reduction.ratio:.0%} 를 0으로 만들었는데 "
                        f"미세조정 뒤에 0이 {outcome.sparsity:.0%} 만 남았다. "
                        "**미세조정이 0을 다시 채웠다.** "
                        "가지치기한 자리를 마스크로 붙잡아 두지 않으면 "
                        "다시 학습하는 순간 가지치기가 없던 일이 된다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=outcome.label,
                    measured=outcome.sparsity,
                    threshold=reduction.ratio * 0.5,
                )
            )

        if outcome.mac_reduction < self.min_mac_reduction:
            findings.append(
                Finding(
                    code="STRUCT_MAC_UNCHANGED",
                    message=(
                        f"곱셈 횟수가 {outcome.mac_reduction:.1%} 줄었다. "
                        "**연산량이 그대로면 지연시간도 그대로다** — "
                        "파라미터 수가 줄었다는 말과 빨라졌다는 말은 다르다."
                    ),
                    severity=Severity.WARNING,
                    subject=outcome.label,
                    measured=outcome.mac_reduction,
                    threshold=self.min_mac_reduction,
                )
            )

        return tuple(findings)


@dataclass(frozen=True, slots=True)
class ReductionComparison:
    """축소 방법들을 나란히 놓은 표. (실습 4-11)"""

    rows: tuple[tuple[StructuralOutcome, tuple[Finding, ...]], ...]

    @property
    def usable(self) -> tuple[str, ...]:
        return tuple(
            outcome.label
            for outcome, findings in self.rows
            if not any(f.is_blocking for f in findings)
        )

    def render(self) -> str:
        lines = ["[구조를 줄이는 방법과 실제로 줄어든 것]"]
        for outcome, findings in self.rows:
            lines.append(f"  {outcome.describe()}")
            for finding in findings:
                lines.append(f"      {finding.describe()}")
        lines.append("")
        lines.append(
            "  쓸 수 있는 것: " + (", ".join(self.usable) or "없음")
        )
        return "\n".join(lines)
