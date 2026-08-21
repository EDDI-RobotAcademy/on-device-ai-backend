"""실험을 기록하지 않으면 다시 만들 수 없다. (실습 6-12)

실습 3-14 는 실험을 **비교**했다. 그건 한 사람의 노트북 안에서였다.
클라우드에서 학습을 돌리기 시작하면 문제가 달라진다.

    학습 잡이 30개 돈다.
    아티팩트는 S3 에 있다.
    파라미터는 각자의 노트북에 있다.
    두 달 뒤 그 사람이 퇴사한다.

그러면 남는 것은 s3://.../model.tar.gz 파일 하나다.
**그 파일이 무엇으로 만들어졌는지 아무도 모른다.**

재현에 필요한 것은 넷이다.

    데이터    어떤 데이터셋의 어느 버전인가
    코드      어떤 커밋인가
    설정      어떤 하이퍼파라미터인가
    결과      무엇이 나왔는가

넷 중 하나라도 없으면 그 실험은 **다시 만들 수 없다.**
그리고 다시 만들 수 없는 모델은 고칠 수도 없다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """학습 한 번의 전모. S3 에 이대로 남는다."""

    experiment_id: str
    trial_id: str
    dataset_version: str = ""
    code_version: str = ""
    parameters: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifact_uri: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        for name in ("experiment_id", "trial_id"):
            if not getattr(self, name).strip():
                raise InvariantViolation(f"{name} 가 없다.", subject=name)

    @property
    def key(self) -> str:
        """S3 에서의 자리. **실험 → 시행 순서로 접두어를 잡는다** (실습 6-2)."""
        return f"experiments/{self.experiment_id}/trials/{self.trial_id}/record.json"

    @property
    def missing_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.dataset_version.strip():
            missing.append("데이터 버전")
        if not self.code_version.strip():
            missing.append("코드 버전")
        if not self.parameters:
            missing.append("하이퍼파라미터")
        if not self.metrics:
            missing.append("결과 지표")
        if not self.artifact_uri.strip():
            missing.append("아티팩트 위치")
        return tuple(missing)

    @property
    def is_reproducible(self) -> bool:
        return not self.missing_fields

    def describe(self) -> str:
        params = " ".join(f"{k}={v}" for k, v in sorted(self.parameters.items()))
        metrics = " ".join(f"{k}={v:.4g}" for k, v in sorted(self.metrics.items()))
        return (
            f"{self.trial_id:<18}data={self.dataset_version:<14}"
            f"code={self.code_version:<10}\n"
            f"    {params}\n    {metrics}"
        )


@dataclass(frozen=True, slots=True)
class ExperimentLedger:
    """한 실험에 속한 시행들. (실습 6-12)"""

    experiment_id: str
    records: tuple[ExperimentRecord, ...] = field(default_factory=tuple)
    missing_artifacts: tuple[str, ...] = field(default_factory=tuple)
    """기록은 있는데 **S3 에 실제 파일이 없는** 것들."""

    def best_by(self, metric: str) -> ExperimentRecord | None:
        scored = [r for r in self.records if metric in r.metrics]
        if not scored:
            return None
        return max(scored, key=lambda r: r.metrics[metric])

    @property
    def reproducible_count(self) -> int:
        return sum(1 for r in self.records if r.is_reproducible)

    def render(self, metric: str = "macro_f1") -> str:
        best = self.best_by(metric)
        lines = [f"[{self.experiment_id}] 시행 {len(self.records)}건"]
        for record in self.records:
            mark = " ←" if best and record.trial_id == best.trial_id else ""
            lines.append(f"  {record.describe()}{mark}")
        lines.append(
            f"  재현 가능 {self.reproducible_count}/{len(self.records)}건"
            + (
                f" / 아티팩트 없음 {len(self.missing_artifacts)}건"
                if self.missing_artifacts
                else ""
            )
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ReproducibilityPolicy:
    """이 기록으로 다시 만들 수 있는가. (실습 6-12)"""

    def inspect(self, ledger: ExperimentLedger) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        for record in ledger.records:
            missing = record.missing_fields
            if missing:
                findings.append(
                    Finding(
                        code="EXPR_NOT_REPRODUCIBLE",
                        message=(
                            f"'{record.trial_id}' 에 {', '.join(missing)} 가 없다. "
                            "**이 시행은 다시 만들 수 없다** — "
                            "그리고 다시 만들 수 없는 모델은 고칠 수도 없다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=record.trial_id,
                        measured=float(len(missing)),
                        threshold=0.0,
                    )
                )

        if ledger.missing_artifacts:
            findings.append(
                Finding(
                    code="EXPR_ARTIFACT_MISSING",
                    message=(
                        f"{len(ledger.missing_artifacts)}건이 **없는 파일**을 가리킨다: "
                        + ", ".join(ledger.missing_artifacts[:3])
                        + ". 기록만 있고 결과물이 없으면 그 기록은 종이다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=ledger.experiment_id,
                    measured=float(len(ledger.missing_artifacts)),
                )
            )

        versions = {r.dataset_version for r in ledger.records if r.dataset_version}
        declared = any("data" in r.parameters for r in ledger.records)
        if len(versions) > 1 and not declared:
            findings.append(
                Finding(
                    code="EXPR_DATA_VERSION_MIXED",
                    message=(
                        f"데이터 버전이 {len(versions)}종류인데 "
                        "파라미터에 적혀 있지 않다. "
                        "**모델을 비교한 것이 아니라 데이터를 비교한 것이다** (실습 3-14)."
                    ),
                    severity=Severity.CRITICAL,
                    subject=ledger.experiment_id,
                    measured=float(len(versions)),
                    threshold=1.0,
                )
            )

        if len(ledger.records) < 2:
            findings.append(
                Finding(
                    code="EXPR_SINGLE_TRIAL",
                    message=(
                        "시행이 하나뿐이다. 기록은 남았지만 **비교는 없다.**"
                    ),
                    severity=Severity.WARNING,
                    subject=ledger.experiment_id,
                    measured=float(len(ledger.records)),
                    threshold=2.0,
                )
            )

        return tuple(findings)
