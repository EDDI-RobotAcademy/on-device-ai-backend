"""모델을 처음으로 현장에 배포하라 — 그 전에 확인할 것. (실습 5-1)

모듈 4 를 통과한 결과물이 있다. 그러면 바로 올려도 되는가?

**아니다.** 파일이 준비된 것과 배포가 준비된 것은 다르다.
아래 것들이 빠진 채 나가는 일이 실제로 자주 있다.

    전처리 통계    없으면 디바이스가 다른 전처리를 한다 — 조용히 다른 모델이 된다
    기준 숫자      없으면 나중에 "느려졌다"를 말할 수 없다
    관측 경로      없으면 배포하고 아무것도 못 본다
    되돌릴 곳      첫 배포에는 없다 — **그래서 첫 배포는 좁게 한다**

마지막 항목이 이 검사에서 가장 중요하다.
첫 배포에는 롤백 대상이 없다. 문제가 나면 격리밖에 못 한다.
그러니 처음부터 수천 대에 올리면 안 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.operations.artifact_ref import DeployedArtifactRef
from domain.operations.target import DeploymentTarget
from domain.shared.inspection import Finding, Severity, Verdict, derive_verdict


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    """배포 전 점검 결과."""

    artifact: DeployedArtifactRef
    target: DeploymentTarget
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    previous_versions: int = 0

    @property
    def verdict(self) -> Verdict:
        return derive_verdict(self.findings)

    @property
    def can_release(self) -> bool:
        return self.verdict is not Verdict.FAILED

    @property
    def is_first_release(self) -> bool:
        return self.previous_versions == 0

    def render(self) -> str:
        lines = [
            f"배포 전 점검: {self.verdict.value}",
            f"  대상 : {self.target.describe()}",
            f"  결과물 : {self.artifact.describe()}",
        ]
        if self.is_first_release:
            lines.append("  **첫 배포다 — 되돌릴 곳이 없다.**")
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        else:
            lines.append("  걸리는 것이 없다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ReleasePolicy:
    """무엇이 갖춰져야 현장에 내보낼 수 있는가."""

    require_selected: bool = True
    require_preprocessing: bool = True
    require_baseline: bool = True
    max_first_release_devices: int = 10
    """첫 배포의 상한. 되돌릴 곳이 없는 배포는 좁아야 한다."""

    def inspect(
        self,
        artifact: DeployedArtifactRef,
        target: DeploymentTarget,
        *,
        previous_versions: int = 0,
    ) -> ReleaseCheck:
        findings: list[Finding] = []

        if self.require_selected and not artifact.selected:
            findings.append(
                Finding(
                    code="RELEASE_NOT_SELECTED",
                    message=(
                        "모듈 4 의 선택 판정을 통과하지 않았다. "
                        "예산 안에 드는지 확인하지 않은 결과물이다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=artifact.artifact_id,
                )
            )

        if self.require_preprocessing and not artifact.has_preprocessing:
            findings.append(
                Finding(
                    code="RELEASE_NO_PREPROCESSING",
                    message=(
                        "전처리 통계가 함께 나가지 않는다. "
                        "**전처리는 모델의 일부다** — 디바이스가 다른 전처리를 하면 "
                        "변환 동등성을 아무리 확인해도 소용없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="normalization",
                )
            )

        if self.require_baseline and not artifact.has_baseline:
            findings.append(
                Finding(
                    code="RELEASE_NO_BASELINE",
                    message=(
                        "기준 지연시간이나 기준 예측 분포가 없다. "
                        "배포한 뒤에 '느려졌다' '이상해졌다'를 말할 근거가 없어진다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="baseline",
                )
            )

        if previous_versions == 0 and target.device_count > self.max_first_release_devices:
            findings.append(
                Finding(
                    code="RELEASE_FIRST_TOO_WIDE",
                    message=(
                        f"첫 배포인데 {target.device_count}대에 한 번에 올린다. "
                        "**되돌릴 버전이 없다** — 문제가 나면 격리밖에 못 한다. "
                        "몇 대로 먼저 확인한 뒤 넓힌다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=target.identifier,
                    measured=float(target.device_count),
                    threshold=float(self.max_first_release_devices),
                )
            )
        elif previous_versions == 0:
            findings.append(
                Finding(
                    code="RELEASE_FIRST_NO_ROLLBACK_TARGET",
                    message=(
                        "첫 배포에는 돌아갈 버전이 없다. "
                        "문제가 나면 격리(실습 5-8)로 멈추는 것이 유일한 수단이다."
                    ),
                    severity=Severity.INFO,
                    subject=target.identifier,
                )
            )

        if not artifact.input_fields:
            findings.append(
                Finding(
                    code="RELEASE_NO_INPUT_SCHEMA",
                    message=(
                        "입력 채널 목록이 없다. 나중에 입력이 변했는지(실습 5-7) 볼 수 없고, "
                        "전처리가 맞는지도 확인할 방법이 없다 — "
                        "**전처리에 대해 아무것도 모르는 채로 내보내는 것**이다."
                    ),
                    severity=Severity.CRITICAL
                    if self.require_preprocessing
                    else Severity.WARNING,
                    subject="input_fields",
                )
            )

        return ReleaseCheck(
            artifact=artifact,
            target=target,
            findings=tuple(findings),
            previous_versions=previous_versions,
        )
