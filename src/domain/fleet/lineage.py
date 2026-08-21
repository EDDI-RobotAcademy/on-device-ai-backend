"""Edge → Cloud → Edge 순환 구조를 완성하라. (실습 6-10)

순환이 닫혔다는 것을 무엇으로 증명하는가?

**계보(lineage)로 증명한다.**

    지금 DEV-02 에서 돌고 있는 모델
      ← 어느 릴리스인가
      ← 어느 학습이 만들었나
      ← 어느 데이터셋으로 학습했나
      ← 어느 디바이스의, 어느 구간 데이터인가

이 사슬이 한 칸이라도 끊기면 순환이 아니라 **일방통행**이다.
데이터는 올라갔고 모델은 내려왔는데, 둘이 이어져 있다는 증거가 없다.

그리고 이 질문은 6개월 뒤에 반드시 나온다.

    "이 모델 뭐로 만들었죠?"
    "그때 그 불량이 학습 데이터에 들어갔나요?"
    "이 디바이스만 이상한데 얘 데이터로 학습한 적 있나요?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.shared.inspection import Finding, Severity, Verdict, derive_verdict


@dataclass(frozen=True, slots=True)
class LineageLink:
    """사슬 한 칸."""

    stage: str
    identifier: str
    detail: str = ""

    @property
    def is_broken(self) -> bool:
        return not self.identifier.strip()

    def describe(self) -> str:
        value = self.identifier or "(끊김)"
        note = f"  {self.detail}" if self.detail else ""
        return f"{self.stage:<16}{value}{note}"


@dataclass(frozen=True, slots=True)
class LineageTrace:
    """한 디바이스에서 시작해 데이터까지 거슬러 올라간 사슬."""

    device_id: str
    links: tuple[LineageLink, ...] = field(default_factory=tuple)

    @property
    def broken(self) -> tuple[LineageLink, ...]:
        return tuple(link for link in self.links if link.is_broken)

    @property
    def is_complete(self) -> bool:
        return bool(self.links) and not self.broken

    def render(self) -> str:
        lines = [f"계보 — {self.device_id}", "-" * 62]
        for index, link in enumerate(self.links):
            arrow = "  " if index == 0 else "↑ "
            lines.append(f"  {arrow}{link.describe()}")
        lines.append("-" * 62)
        lines.append(
            "  사슬이 이어져 있다." if self.is_complete else "  **사슬이 끊겼다.**"
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class LoopClosure:
    """순환이 닫혔는지에 대한 판정."""

    trace: LineageTrace
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> Verdict:
        return derive_verdict(self.findings)

    @property
    def closed(self) -> bool:
        return self.verdict is not Verdict.FAILED

    def render(self) -> str:
        lines = [self.trace.render(), "", f"순환 판정: {self.verdict.value}"]
        if self.findings:
            lines += [f"  - {f.describe()}" for f in self.findings]
        else:
            lines.append("  Edge → Cloud → Edge 가 이어져 있다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class LineagePolicy:
    """무엇이 이어져 있어야 순환이라고 부를 것인가."""

    required_stages: tuple[str, ...] = (
        "디바이스",
        "릴리스",
        "학습",
        "데이터셋",
        "원본 구간",
    )

    require_source_devices: bool = True
    """그 데이터가 **어느 디바이스에서 왔는지**까지 남아야 한다."""

    def inspect(
        self, trace: LineageTrace, *, source_devices: tuple[str, ...] = ()
    ) -> LoopClosure:
        findings: list[Finding] = []

        present = {link.stage for link in trace.links if not link.is_broken}
        for stage in self.required_stages:
            if stage not in present:
                findings.append(
                    Finding(
                        code="LINEAGE_BROKEN",
                        message=(
                            f"'{stage}' 칸이 비어 있다. "
                            "여기서 사슬이 끊기면 그 위로는 못 올라간다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=stage,
                    )
                )

        if self.require_source_devices and not source_devices:
            findings.append(
                Finding(
                    code="LINEAGE_NO_SOURCE_DEVICES",
                    message=(
                        "학습 데이터가 어느 디바이스에서 왔는지 남아 있지 않다. "
                        "'이 디바이스 데이터로 학습한 적 있나요?'에 답할 수 없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="source_devices",
                )
            )
        elif trace.device_id and trace.device_id not in source_devices:
            findings.append(
                Finding(
                    code="LINEAGE_SELF_EXCLUDED",
                    message=(
                        f"'{trace.device_id}' 는 이 모델의 학습 데이터에 들어가지 않았다. "
                        "**틀린 것은 아니다** — 격리됐거나 나중에 설치된 디바이스일 수 있다. "
                        "다만 그 사실을 알고 있어야 한다."
                    ),
                    severity=Severity.INFO,
                    subject=trace.device_id,
                )
            )

        return LoopClosure(trace=trace, findings=tuple(findings))


def trace_of(
    *,
    device_id: str,
    version: str,
    job_id: str,
    build_id: str,
    window: str,
    detail: str = "",
) -> LineageTrace:
    """다섯 칸짜리 사슬을 만든다.

    빈 문자열이 들어오면 그 칸이 '끊김'으로 남는다 — **숨기지 않는다.**
    """
    return LineageTrace(
        device_id=device_id,
        links=(
            LineageLink("디바이스", device_id),
            LineageLink("릴리스", version),
            LineageLink("학습", job_id),
            LineageLink("데이터셋", build_id),
            LineageLink("원본 구간", window, detail),
        ),
    )
