"""Data Context 의 검사 결과.

판정을 표현하는 문법(Finding / Severity / Verdict)은 Shared Kernel 에 있다.
여기서 정하는 것은 **이 Context 가 무엇을 검사하는가**(InspectionKind)뿐이다.

    측정값(Measurement)  +  기준(Policy)  →  InspectionReport
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.shared.inspection import (
    Finding,
    Severity,
    Verdict,
    blocking_findings,
    derive_verdict,
)

__all__ = [
    "Finding",
    "InspectionKind",
    "InspectionReport",
    "Severity",
    "Verdict",
]


class InspectionKind(Enum):
    """Dataset 이 통과해야 하는 검사의 종류."""

    PROFILE = "PROFILE"
    SCHEMA = "SCHEMA"
    SIGNAL_PLAUSIBILITY = "SIGNAL_PLAUSIBILITY"
    TIME_AXIS = "TIME_AXIS"
    LABEL_SPACE = "LABEL_SPACE"
    TRAINING_SPEC = "TRAINING_SPEC"
    PARTITION = "PARTITION"
    REPRESENTATIVENESS = "REPRESENTATIVENESS"


@dataclass(frozen=True, slots=True)
class InspectionReport:
    """한 종류의 검사에 대한 결과."""

    kind: InspectionKind
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> Verdict:
        return derive_verdict(self.findings)

    @property
    def passed(self) -> bool:
        """학습 진행을 막지 않는가? (경고는 막지 않는다)"""
        return self.verdict is not Verdict.FAILED

    @property
    def blocking_findings(self) -> tuple[Finding, ...]:
        return blocking_findings(self.findings)

    def summary(self) -> str:
        return f"{self.kind.value}: {self.verdict.value} (findings={len(self.findings)})"
