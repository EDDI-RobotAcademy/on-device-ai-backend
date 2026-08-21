"""학습 착수 판정(Domain Policy).

실습 1-10 "학습시키기 전에 데이터부터 검증하라".

앞의 9개 실습에서 만든 검사 결과를 모아 단 하나의 질문에 답한다.

    "이 데이터로 학습을 시작해도 되는가?"

이 판정은 사람의 감이 아니라 재현 가능한 규칙이어야 한다.
그래야 3개월 뒤에 "그때 왜 학습을 시작했나"를 설명할 수 있다.

이것은 data_quality Context 의 Data Quality Gate 와 다르다.
    여기(Data)          : 데이터를 이해했고 계약이 서 있는가 — 구조/시간/라벨/분할/대표성
    Data Quality Gate   : 그 데이터가 얼마나 오염되었는가 — 결측/이상치/중복/노이즈
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.data.identifiers import DatasetId
from domain.data.inspection import (
    Finding,
    InspectionKind,
    InspectionReport,
    Severity,
    Verdict,
)
from domain.shared.errors import InvariantViolation

DEFAULT_REQUIRED_KINDS: frozenset[InspectionKind] = frozenset(
    {
        InspectionKind.SCHEMA,
        InspectionKind.SIGNAL_PLAUSIBILITY,
        InspectionKind.LABEL_SPACE,
        InspectionKind.TRAINING_SPEC,
        InspectionKind.PARTITION,
        InspectionKind.REPRESENTATIVENESS,
    }
)


@dataclass(frozen=True, slots=True)
class ReadinessCertificate:
    """판정의 결과이자 근거 기록.

    통과했든 막혔든 남는다. "왜 막혔는지"가 남지 않으면 아무도 고칠 수 없다.
    """

    dataset_id: DatasetId
    verdict: Verdict
    evaluated_kinds: tuple[InspectionKind, ...]
    missing_kinds: tuple[InspectionKind, ...]
    blocking_findings: tuple[Finding, ...]
    warning_findings: tuple[Finding, ...]

    @property
    def is_ready(self) -> bool:
        return self.verdict is not Verdict.FAILED

    def summary(self) -> str:
        return (
            f"{self.dataset_id} → {self.verdict.value} "
            f"(검사 {len(self.evaluated_kinds)}종, 차단 {len(self.blocking_findings)}건, "
            f"경고 {len(self.warning_findings)}건)"
        )

    def reasons(self) -> tuple[str, ...]:
        lines = [f"검사 누락: {k.value}" for k in self.missing_kinds]
        lines += [f.describe() for f in self.blocking_findings]
        return tuple(lines)


@dataclass(frozen=True, slots=True)
class ReadinessPolicy:
    """무엇을 다 통과해야 학습을 시작할 수 있는가."""

    required_kinds: frozenset[InspectionKind] = field(default=DEFAULT_REQUIRED_KINDS)
    allow_warnings: bool = True
    """경고를 안고 갈 것인가. 안전 관련 라인에서는 False 로 조인다."""

    max_warning_count: int = 10

    def __post_init__(self) -> None:
        if not self.required_kinds:
            raise InvariantViolation(
                "아무 검사도 요구하지 않는 판정 기준은 기준이 아니다.", subject="required_kinds"
            )
        if self.max_warning_count < 0:
            raise InvariantViolation(
                "max_warning_count 는 음수일 수 없다.", subject="max_warning_count"
            )

    def evaluate(
        self,
        dataset_id: DatasetId,
        reports: Mapping[InspectionKind, InspectionReport],
    ) -> ReadinessCertificate:
        missing = tuple(sorted(
            (k for k in self.required_kinds if k not in reports), key=lambda k: k.value
        ))

        blocking: list[Finding] = []
        warnings: list[Finding] = []
        for kind in sorted(reports, key=lambda k: k.value):
            for finding in reports[kind].findings:
                if finding.severity is Severity.CRITICAL:
                    blocking.append(finding)
                elif finding.severity is Severity.WARNING:
                    warnings.append(finding)

        if missing:
            blocking.extend(
                Finding(
                    code="READINESS_INSPECTION_MISSING",
                    message=f"{kind.value} 검사를 수행하지 않았다.",
                    severity=Severity.CRITICAL,
                    subject=kind.value,
                )
                for kind in missing
            )

        too_many_warnings = len(warnings) > self.max_warning_count
        if too_many_warnings:
            blocking.append(
                Finding(
                    code="READINESS_TOO_MANY_WARNINGS",
                    message="경고가 너무 많다. 개별로는 넘어갈 수 있어도 합치면 데이터를 신뢰할 수 없다.",
                    severity=Severity.CRITICAL,
                    subject="warnings",
                    measured=float(len(warnings)),
                    threshold=float(self.max_warning_count),
                )
            )

        if blocking:
            verdict = Verdict.FAILED
        elif warnings:
            verdict = Verdict.PASSED if self.allow_warnings else Verdict.FAILED
            if not self.allow_warnings:
                blocking.extend(warnings)
            else:
                verdict = Verdict.PASSED_WITH_WARNINGS
        else:
            verdict = Verdict.PASSED

        return ReadinessCertificate(
            dataset_id=dataset_id,
            verdict=verdict,
            evaluated_kinds=tuple(sorted(reports, key=lambda k: k.value)),
            missing_kinds=missing,
            blocking_findings=tuple(blocking),
            warning_findings=tuple(warnings),
        )
