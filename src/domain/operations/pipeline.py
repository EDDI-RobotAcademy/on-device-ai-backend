"""디바이스에서 도는 추론 파이프라인. (실습 5-12)

지금까지 모듈 5 는 **로그를 읽는 쪽**만 봤다.
그 로그를 만드는 쪽, 즉 디바이스 안에서 실제로 도는 순서는 이렇다.

    입력 수집 → 전처리 → 추론 → 후처리 → 알람/저장

다섯 단계 전부에 실패 방식이 있고, **대부분의 현장 사고는 추론이 아닌 곳에서 난다.**

    입력 수집   센서가 값을 안 준다. 카메라가 검은 화면을 준다.
    전처리      학습 때와 다른 정규화를 쓴다. 아무 에러도 안 난다.
    추론        여기는 대개 멀쩡하다.
    후처리      확신도가 낮은데 그대로 답으로 쓴다.
    알람/저장   알람이 너무 많아서 아무도 안 본다 (실습 5-13).

그래서 이 파일은 "추론이 몇 ms 걸리는가"가 아니라
**"각 단계에서 무엇이 빠질 수 있는가"**를 모델링한다.

계약(전처리 명세)은 학습 때 정한 것과 **같아야 한다**.
그것을 지키는 장치가 `PipelineContract` 다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class PipelineStage(Enum):
    ACQUIRE = "ACQUIRE"
    """센서·카메라에서 원본을 받는다."""

    PREPROCESS = "PREPROCESS"
    """학습 때와 **똑같이** 자르고 정규화한다."""

    INFER = "INFER"
    POSTPROCESS = "POSTPROCESS"
    """점수를 라벨로 바꾸고, 확신이 부족하면 답을 보류한다."""

    EMIT = "EMIT"
    """알람을 내보내고 로그를 남긴다."""

    @property
    def order(self) -> int:
        return list(PipelineStage).index(self)


@dataclass(frozen=True, slots=True)
class PipelineContract:
    """학습 때 정한 것과 **같아야 하는** 것들.

    이 값이 하나라도 어긋나면 모델은 아무 에러 없이 다른 것을 본다.
    그래서 배포 시점에 대조한다 — 실행 중에 알아채는 것은 이미 늦다.
    """

    input_shape: tuple[int, ...]
    sample_interval_seconds: float
    """표본 간격. **모델의 일부다** (실습 5-1, 6-6)."""

    feature_fields: tuple[str, ...] = ()
    normalization: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    class_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.input_shape:
            raise InvariantViolation("입력 모양이 없다.", subject="input_shape")
        if self.sample_interval_seconds <= 0:
            raise InvariantViolation(
                "표본 간격은 0보다 커야 한다.", subject="sample_interval_seconds"
            )

    def differences_from(self, other: PipelineContract) -> tuple[str, ...]:
        """두 계약이 어디서 다른가."""
        gaps: list[str] = []
        if tuple(self.input_shape) != tuple(other.input_shape):
            gaps.append(f"입력 모양 {other.input_shape} → {self.input_shape}")
        if self.sample_interval_seconds != other.sample_interval_seconds:
            gaps.append(
                f"표본 간격 {other.sample_interval_seconds:g}s → "
                f"{self.sample_interval_seconds:g}s"
            )
        if tuple(self.feature_fields) != tuple(other.feature_fields):
            gaps.append("입력 열 구성")
        if dict(self.normalization) != dict(other.normalization):
            gaps.append("정규화 통계")
        if tuple(self.class_labels) != tuple(other.class_labels):
            gaps.append(f"라벨 순서 {other.class_labels} → {self.class_labels}")
        return tuple(gaps)

    def describe(self) -> str:
        return (
            f"입력 {self.input_shape} / {self.sample_interval_seconds:g}s 간격 / "
            f"열 {len(self.feature_fields)}개 / 라벨 {self.class_labels}"
        )


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """단계 하나가 남긴 것."""

    stage: PipelineStage
    attempted: int
    succeeded: int
    duration_ms: float = 0.0
    reason_counts: Mapping[str, int] = field(default_factory=dict)
    """실패한 것들의 이유별 개수. **'몇 개 실패'만으로는 못 고친다.**"""

    def __post_init__(self) -> None:
        if self.attempted < 0 or self.succeeded < 0:
            raise InvariantViolation("개수는 음수일 수 없다.", subject=self.stage.value)
        if self.succeeded > self.attempted:
            raise InvariantViolation(
                "성공이 시도보다 많을 수 없다.", subject=self.stage.value
            )

    @property
    def dropped(self) -> int:
        return self.attempted - self.succeeded

    @property
    def drop_ratio(self) -> float:
        return self.dropped / self.attempted if self.attempted else 0.0

    def describe(self) -> str:
        reasons = ", ".join(
            f"{name} {count}" for name, count in sorted(self.reason_counts.items())
        )
        return (
            f"{self.stage.value:<12}{self.attempted:>7,} → {self.succeeded:>7,}"
            f"  ({self.drop_ratio:>6.2%} 빠짐)  {self.duration_ms:>8.1f}ms"
            + (f"  [{reasons}]" if reasons else "")
        )


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """파이프라인을 한 번 돌린 결과. (실습 5-12)"""

    device_id: str
    contract: PipelineContract
    stages: tuple[StageOutcome, ...]
    emitted_alerts: int = 0
    withheld: int = 0
    """확신이 부족해 답을 보류한 건수."""

    def __post_init__(self) -> None:
        if not self.stages:
            raise InvariantViolation("단계가 하나도 없다.", subject="stages")
        order = [s.stage.order for s in self.stages]
        if order != sorted(order):
            raise InvariantViolation(
                "단계가 순서대로가 아니다. **추론 뒤에 전처리를 하는 파이프라인은 없다.**",
                subject="stages",
            )

    def stage_of(self, stage: PipelineStage) -> StageOutcome | None:
        return next((s for s in self.stages if s.stage is stage), None)

    @property
    def acquired(self) -> int:
        """센서가 준 표본 수. 판단 횟수가 아니다 — 창은 stride 마다 만든다."""
        return self.stages[0].attempted

    @property
    def decision_opportunities(self) -> int:
        """만들 수 있었던 판단의 수. 여기가 끝까지 가는 비율의 분모다."""
        stage = self.stage_of(PipelineStage.PREPROCESS)
        return stage.attempted if stage else self.acquired

    @property
    def answered(self) -> int:
        emit = self.stage_of(PipelineStage.EMIT)
        return emit.succeeded if emit else 0

    @property
    def end_to_end_ratio(self) -> float:
        """만들 수 있었던 판단 중 **답이 남은** 비율.

        추론 성공률이 아니다. **끝까지 간 비율**이다.
        """
        return (
            self.answered / self.decision_opportunities
            if self.decision_opportunities
            else 0.0
        )

    @property
    def total_ms(self) -> float:
        return sum(s.duration_ms for s in self.stages)

    @property
    def slowest(self) -> StageOutcome:
        return max(self.stages, key=lambda s: s.duration_ms)

    def render(self) -> str:
        lines = [
            f"[{self.device_id}] {self.contract.describe()}",
            f"{'단계':<12}{'시도':>7}   {'성공':>7}    {'빠짐':>8}  {'시간':>10}",
            "-" * 62,
        ]
        lines += [f"  {s.describe()}" for s in self.stages]
        lines.append("-" * 62)
        lines.append(
            f"  센서 {self.acquired:,}표본 → 판단 기회 "
            f"{self.decision_opportunities:,}건 → 답 {self.answered:,}건 "
            f"({self.end_to_end_ratio:.1%})  "
            f"보류 {self.withheld:,}  알람 {self.emitted_alerts:,}"
        )
        lines.append(
            f"  전체 {self.total_ms:.1f}ms / 가장 느린 단계 {self.slowest.stage.value}"
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PipelinePolicy:
    """이 파이프라인을 현장에 둬도 되는가. (실습 5-12)"""

    max_stage_drop_ratio: float = 0.05
    min_end_to_end_ratio: float = 0.9
    max_withheld_ratio: float = 0.3

    def inspect(
        self, run: PipelineRun, *, trained_contract: PipelineContract | None = None
    ) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        if trained_contract is not None:
            gaps = run.contract.differences_from(trained_contract)
            if gaps:
                findings.append(
                    Finding(
                        code="PIPE_CONTRACT_MISMATCH",
                        message=(
                            "학습 때의 계약과 다르다: " + "; ".join(gaps) + ". "
                            "**아무 에러도 안 난다.** 모델은 다른 것을 보면서 "
                            "그럴듯한 답을 계속 낸다 — 현장에서만 틀린다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=run.device_id,
                        measured=float(len(gaps)),
                        threshold=0.0,
                    )
                )

        for stage in run.stages:
            if stage.drop_ratio > self.max_stage_drop_ratio:
                findings.append(
                    Finding(
                        code="PIPE_STAGE_DROP",
                        message=(
                            f"{stage.stage.value} 에서 "
                            f"{stage.dropped:,}건({stage.drop_ratio:.1%})이 빠진다. "
                            + (
                                "이유별로 세어 두지 않으면 고칠 수 없다."
                                if not stage.reason_counts
                                else "이유: "
                                + ", ".join(
                                    f"{k} {v}"
                                    for k, v in sorted(stage.reason_counts.items())
                                )
                            )
                        ),
                        severity=(
                            Severity.CRITICAL
                            if stage.stage
                            in (PipelineStage.ACQUIRE, PipelineStage.INFER)
                            else Severity.WARNING
                        ),
                        subject=stage.stage.value,
                        measured=stage.drop_ratio,
                        threshold=self.max_stage_drop_ratio,
                    )
                )

        if run.end_to_end_ratio < self.min_end_to_end_ratio:
            findings.append(
                Finding(
                    code="PIPE_END_TO_END_LOSS",
                    message=(
                        f"판단 기회 {run.decision_opportunities:,}건 중 "
                        f"{run.answered:,}건({run.end_to_end_ratio:.1%})만 답이 되었다. "
                        "**추론 성공률이 100%여도 이 숫자는 낮을 수 있다** — "
                        "빠지는 곳은 대개 추론이 아니다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=run.device_id,
                    measured=run.end_to_end_ratio,
                    threshold=self.min_end_to_end_ratio,
                )
            )

        if (
            run.answered
            and run.withheld / max(run.decision_opportunities, 1)
            > self.max_withheld_ratio
        ):
            findings.append(
                Finding(
                    code="PIPE_TOO_MANY_WITHHELD",
                    message=(
                        f"{run.withheld:,}건을 확신 부족으로 보류했다. "
                        "**보류가 이만큼이면 사람이 그만큼 봐야 한다** — "
                        "자동화했다고 말할 수 없다. 확신 문턱을 다시 정하거나 "
                        "모델을 다시 봐야 한다."
                    ),
                    severity=Severity.WARNING,
                    subject=run.device_id,
                    measured=run.withheld / max(run.decision_opportunities, 1),
                    threshold=self.max_withheld_ratio,
                )
            )

        return tuple(findings)
