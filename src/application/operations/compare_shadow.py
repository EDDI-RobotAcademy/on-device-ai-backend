"""새 모델과 기존 모델을 실제 데이터로 비교하라. (실습 5-9)

새 모델의 답은 **쓰지 않는다.** 설비는 여전히 기존 모델로 움직인다.
새 모델은 옆에서 같은 입력을 받아 답만 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.operations.dto import ShadowView
from application.operations.support import load_deployment
from domain.operations.ports import DeploymentRepository, ShadowRunner
from domain.operations.shadow import PromotionPolicy
from domain.operations.window import ObservationWindow


@dataclass(frozen=True, slots=True)
class CompareShadowCommand:
    deployment_id: str
    window: ObservationWindow
    candidate_artifact_id: str
    policy: PromotionPolicy = field(default_factory=PromotionPolicy)


class CompareShadow:
    def __init__(
        self, deployments: DeploymentRepository, runner: ShadowRunner
    ) -> None:
        self._deployments = deployments
        self._runner = runner

    def execute(self, command: CompareShadowCommand) -> ShadowView:
        deployment = load_deployment(self._deployments, command.deployment_id)
        run = self._runner.run(
            deployment.id, command.window, command.candidate_artifact_id
        )
        verdict = command.policy.evaluate(run)
        return ShadowView.of(str(deployment.id), verdict)
