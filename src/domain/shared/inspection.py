"""검사 결과의 공통 언어 (Shared Kernel).

Data Context 와 Data Quality Context 는 서로를 모른다.
그러나 "무엇이 얼마여서 기준 얼마를 넘었다"를 말하는 방식은 같아야 한다.
그래야 두 검사의 결과를 한 화면에 놓고 비교할 수 있다.

여기 있는 것은 '판정을 표현하는 문법'뿐이다.
무엇을 검사할지, 기준이 얼마인지는 각 Context 가 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.shared.errors import InvariantViolation


class Severity(Enum):
    INFO = "INFO"
    """확인이 필요한 선택. 진행을 막지 않는다."""

    WARNING = "WARNING"
    """지금 진행해도 되지만, 이 사실을 알고 있어야 한다."""

    CRITICAL = "CRITICAL"
    """이 상태로 진행하면 모델이 아니라 현장이 대가를 치른다."""


class Verdict(Enum):
    PASSED = "PASSED"
    PASSED_WITH_WARNINGS = "PASSED_WITH_WARNINGS"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class Finding:
    """검사가 발견한 사실 하나.

    "이상하다"가 아니라 "무엇이 얼마여서 기준 얼마를 넘었다"까지 적는다.
    숫자가 없는 Finding 은 현장에서 아무도 신뢰하지 않고, 아무도 고칠 수 없다.
    """

    code: str
    message: str
    severity: Severity
    subject: str | None = None
    measured: float | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise InvariantViolation(
                "Finding 은 추적 가능한 code 를 가져야 한다.", subject="code"
            )

    @property
    def is_blocking(self) -> bool:
        return self.severity is Severity.CRITICAL

    def describe(self) -> str:
        parts = [f"{self.severity.value} {self.code}"]
        if self.subject:
            parts.append(f"({self.subject})")
        parts.append(self.message)
        if self.measured is not None and self.threshold is not None:
            parts.append(f"[측정 {self.measured:.4g} / 기준 {self.threshold:.4g}]")
        elif self.measured is not None:
            parts.append(f"[측정 {self.measured:.4g}]")
        return " ".join(parts)


def derive_verdict(findings: tuple[Finding, ...]) -> Verdict:
    """판정은 저장하지 않고 Finding 들로부터 유도한다.

    누군가 `report.verdict = PASSED` 로 덮어쓰는 경로를 원천적으로 없애기 위해서다.
    현장에서 검사 결과를 '일단 통과로 바꿔 놓는' 일은 실제로 일어난다.
    """
    if any(f.severity is Severity.CRITICAL for f in findings):
        return Verdict.FAILED
    if any(f.severity is Severity.WARNING for f in findings):
        return Verdict.PASSED_WITH_WARNINGS
    return Verdict.PASSED


def blocking_findings(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    return tuple(f for f in findings if f.is_blocking)


def warning_findings(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    return tuple(f for f in findings if f.severity is Severity.WARNING)
