"""배포된 모델에 Version을 부여하라. (실습 5-2)

세 가지 버전이 있다. **서로 다른 것이다.**

    모델 버전    mv-power-cnn1d-s42   어떤 학습이 만들어 낸 가중치인가
    결과물       TFLITE/INT8          그 가중치를 어떤 형식으로 바꿨는가
    배포 버전    v3                   그것이 언제 어디에 올라갔는가

같은 모델을 두 번 배포하면 모델 버전은 같고 배포 버전은 다르다.
현장에서 물어보는 질문은 대개 **배포 버전으로만** 답할 수 있다.

    "어제 3시에 3라인에서 돌던 게 뭐였죠?"   → v3
    "그거 언제부터 돌았어요?"                 → v3 이 올라간 시각
    "그 전 건요?"                             → v2
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.operations.artifact_ref import DeployedArtifactRef
from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class DeploymentVersion:
    """한 번의 릴리스."""

    number: int
    artifact: DeployedArtifactRef
    released_at: str
    """Domain 은 시계를 모른다. Application 이 Clock 으로 찍어서 넣는다."""

    note: str = ""
    rolled_back_from: int | None = None
    """롤백으로 만들어진 버전이면, 어느 버전에서 되돌아왔는지. (실습 5-10)"""

    def __post_init__(self) -> None:
        if self.number < 1:
            raise InvariantViolation("배포 버전은 1부터다.", subject="number")
        if not self.released_at.strip():
            raise InvariantViolation(
                "언제 올라갔는지 없으면 '언제부터'에 답할 수 없다.",
                subject="released_at",
            )
        if self.rolled_back_from is not None and self.rolled_back_from >= self.number:
            raise InvariantViolation(
                "롤백은 앞선 버전으로만 돌아간다.", subject="rolled_back_from"
            )

    @property
    def label(self) -> str:
        return f"v{self.number}"

    @property
    def is_rollback(self) -> bool:
        return self.rolled_back_from is not None

    @property
    def model_version_id(self) -> str:
        return self.artifact.model_version_id

    def describe(self) -> str:
        origin = (
            f"  ← v{self.rolled_back_from} 에서 롤백" if self.is_rollback else ""
        )
        return (
            f"{self.label:<5}{self.released_at}  "
            f"{self.artifact.model_version_id} ({self.artifact.label}){origin}"
        )
