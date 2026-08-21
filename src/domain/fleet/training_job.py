"""SageMaker에서 새로운 모델을 학습시켜라. (실습 6-5)

**이 파일에 SageMaker 라는 단어가 없다.** 원격 학습은 어디서 하든 같은 모양이다.

    입력은 저장소에 있고
    계산은 남의 기계에서 돌고
    결과는 다시 저장소로 나온다
    그리고 **언제 끝나는지 모른다**

마지막이 로컬 학습(모듈 3)과의 결정적인 차이다.
`trainer.fit()` 은 돌아올 때까지 기다리면 됐다. 여기서는 기다릴 수 없다.

    Command → Job → Status → Result   (CLAUDE.md §11)

그리고 하나 더. **원격 학습은 실패하는 방식이 다르다.**
로컬에서는 예외가 올라온다. 원격에서는 상태가 FAILED 로 바뀌고,
이유는 로그 어딘가에 있다. 그 이유를 Job 에 붙여 두지 않으면 아무도 못 찾는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class RemoteJobStatus(Enum):
    PENDING = "PENDING"
    """제출했다. 아직 기계가 안 잡혔다."""

    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    """사람이 멈췄다. 대개 비용 때문이다."""

    @property
    def is_terminal(self) -> bool:
        return self in (
            RemoteJobStatus.SUCCEEDED,
            RemoteJobStatus.FAILED,
            RemoteJobStatus.STOPPED,
        )


@dataclass(frozen=True, slots=True)
class ComputeSpec:
    """어떤 기계에서 얼마나 돌릴 것인가.

    이 세 줄이 곧 청구서다. 학습 코드보다 이쪽을 먼저 본다.
    """

    instance_type: str
    instance_count: int = 1
    max_runtime_seconds: int = 3_600
    hourly_cost_usd: float = 0.0
    """0 이면 모른다는 뜻이다. 모르면 예산 검사를 못 한다."""

    def __post_init__(self) -> None:
        if not self.instance_type.strip():
            raise InvariantViolation(
                "어떤 기계인지 없으면 비용을 알 수 없다.", subject="instance_type"
            )
        if self.instance_count < 1:
            raise InvariantViolation("기계 0대는 없다.", subject="instance_count")
        if self.max_runtime_seconds < 60:
            raise InvariantViolation(
                "1분 안에 끝나는 학습은 학습이 아니다.", subject="max_runtime_seconds"
            )

    @property
    def worst_case_cost_usd(self) -> float:
        """최악의 경우 얼마인가. **평균이 아니라 상한을 본다.**"""
        hours = self.max_runtime_seconds / 3600
        return self.hourly_cost_usd * self.instance_count * hours

    def describe(self) -> str:
        cost = (
            f"  최대 ${self.worst_case_cost_usd:,.2f}"
            if self.hourly_cost_usd
            else "  (비용 미상)"
        )
        return (
            f"{self.instance_type} × {self.instance_count}  "
            f"최대 {self.max_runtime_seconds // 60}분{cost}"
        )


@dataclass(frozen=True, slots=True)
class RemoteTrainingJob:
    """클라우드에 맡긴 학습 한 번."""

    job_id: str
    dataset_uri: str
    output_uri: str
    compute: ComputeSpec
    status: RemoteJobStatus = RemoteJobStatus.PENDING
    submitted_at: str = ""
    finished_at: str = ""
    failure_reason: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifact_uri: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_uri.strip():
            raise InvariantViolation(
                "입력 데이터가 어디 있는지 없다.", subject="dataset_uri"
            )
        if not self.output_uri.strip():
            raise InvariantViolation(
                "결과를 어디에 둘지 없다. 학습이 끝나도 못 찾는다.",
                subject="output_uri",
            )
        if self.status is RemoteJobStatus.FAILED and not self.failure_reason.strip():
            raise InvariantViolation(
                "실패했는데 이유가 없다. 로그 어딘가에 있는 것은 없는 것과 같다.",
                subject="failure_reason",
            )
        if self.status is RemoteJobStatus.SUCCEEDED and not self.artifact_uri.strip():
            raise InvariantViolation(
                "성공했는데 결과물 위치가 없다.", subject="artifact_uri"
            )

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def succeeded(self) -> bool:
        return self.status is RemoteJobStatus.SUCCEEDED

    def describe(self) -> str:
        line = f"{self.job_id}  {self.status.value}  {self.compute.describe()}"
        if self.failure_reason:
            line += f"\n    실패 이유: {self.failure_reason}"
        if self.metrics:
            line += "\n    " + "  ".join(
                f"{k}={v:.4f}" for k, v in sorted(self.metrics.items())
            )
        return line


@dataclass(frozen=True, slots=True)
class TrainingBudgetPolicy:
    """이 학습을 시켜도 되는가.

    학습 자체의 품질은 모듈 3 이 본다. 여기서 보는 것은 **비용과 시간**이다.
    """

    max_cost_usd: float = 20.0
    max_runtime_seconds: int = 7_200
    require_cost_estimate: bool = True
    min_metrics: Mapping[str, float] = field(default_factory=dict)
    """끝난 뒤에 이 값들을 넘어야 결과물로 인정한다."""

    def inspect_submission(self, job: RemoteTrainingJob) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        compute = job.compute

        if self.require_cost_estimate and not compute.hourly_cost_usd:
            findings.append(
                Finding(
                    code="TRAIN_NO_COST_ESTIMATE",
                    message=(
                        "시간당 비용을 모른 채 제출하고 있다. "
                        "**클라우드 학습에서 가장 흔한 사고는 정확도가 아니라 청구서다.**"
                    ),
                    severity=Severity.WARNING,
                    subject=job.job_id,
                )
            )
        elif compute.worst_case_cost_usd > self.max_cost_usd:
            findings.append(
                Finding(
                    code="TRAIN_OVER_BUDGET",
                    message=(
                        f"최악의 경우 ${compute.worst_case_cost_usd:,.2f} 다. "
                        "최대 실행 시간을 줄이거나 기계를 낮춘다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=job.job_id,
                    measured=compute.worst_case_cost_usd,
                    threshold=self.max_cost_usd,
                )
            )

        if compute.max_runtime_seconds > self.max_runtime_seconds:
            findings.append(
                Finding(
                    code="TRAIN_RUNTIME_TOO_LONG",
                    message=(
                        "최대 실행 시간이 너무 길다. "
                        "**멈추지 않는 학습은 끝나지 않고 과금만 된다** — "
                        "상한이 곧 안전장치다."
                    ),
                    severity=Severity.WARNING,
                    subject=job.job_id,
                    measured=float(compute.max_runtime_seconds),
                    threshold=float(self.max_runtime_seconds),
                )
            )

        return tuple(findings)

    def inspect_result(self, job: RemoteTrainingJob) -> tuple[Finding, ...]:
        if not job.is_terminal:
            return (
                Finding(
                    code="TRAIN_STILL_RUNNING",
                    message="아직 끝나지 않았다. 결과를 판단할 수 없다.",
                    severity=Severity.INFO,
                    subject=job.job_id,
                ),
            )

        if not job.succeeded:
            return (
                Finding(
                    code="TRAIN_FAILED",
                    message=f"{job.status.value}: {job.failure_reason}",
                    severity=Severity.CRITICAL,
                    subject=job.job_id,
                ),
            )

        findings: list[Finding] = []
        for name, floor in self.min_metrics.items():
            value = job.metrics.get(name)
            if value is None:
                findings.append(
                    Finding(
                        code="TRAIN_METRIC_MISSING",
                        message=(
                            f"'{name}' 지표가 결과에 없다. "
                            "학습은 끝났는데 좋은지 나쁜지 말할 수 없다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=name,
                    )
                )
            elif value < floor:
                findings.append(
                    Finding(
                        code="TRAIN_METRIC_BELOW_FLOOR",
                        message=(
                            f"'{name}' 가 기준 아래다. "
                            "이 결과물을 디바이스로 내보내지 않는다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=name,
                        measured=value,
                        threshold=floor,
                    )
                )
        return tuple(findings)
