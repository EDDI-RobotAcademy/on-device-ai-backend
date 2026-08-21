"""Deployment Aggregate 의 불변식.

파일도, 모델도, 로그도 없이 돌아간다.
**결정의 기록**은 기술을 하나도 모른다.
"""

from __future__ import annotations

import pytest

from domain.operations.artifact_ref import DeployedArtifactRef
from domain.operations.deployment import Deployment, DeploymentStatus
from domain.operations.errors import NotDeployable, VersionNotFound
from domain.operations.identifiers import DeploymentId
from domain.operations.target import DeploymentTarget, TargetKind
from domain.shared.errors import IllegalStateTransition, InvariantViolation

LABELS = ("FAULT", "OVERLOAD", "NORMAL")


def artifact(**overrides) -> DeployedArtifactRef:  # noqa: ANN003
    base: dict[str, object] = dict(
        artifact_id="mv-1-tflite-fp16",
        optimization_run_ref="opt-1",
        model_version_id="mv-1",
        runtime="TFLITE",
        precision="FP16",
        size_bytes=11_724,
        class_labels=LABELS,
        input_fields=("active_power_kw",),
        expected_p95_ms=0.0031,
        expected_accuracy=0.97,
        expected_class_mix={"NORMAL": 0.79, "OVERLOAD": 0.17, "FAULT": 0.04},
        normalization={"active_power_kw": (147.8, 39.8)},
        selected=True,
    )
    base.update(overrides)
    return DeployedArtifactRef(**base)  # type: ignore[arg-type]


def target(**overrides) -> DeploymentTarget:  # noqa: ANN003
    base: dict[str, object] = dict(
        kind=TargetKind.DEVICE_GROUP, identifier="LINE-3", device_count=3
    )
    base.update(overrides)
    return DeploymentTarget(**base)  # type: ignore[arg-type]


def deployed(**overrides) -> Deployment:  # noqa: ANN003
    return Deployment.deploy(
        DeploymentId.of("dep-1"),
        target(),
        artifact(**overrides),
        "2026-05-19 23:00:00",
    )


class Test배포:
    def test_선택되지_않은_결과물은_배포_대상이_아니다(self) -> None:
        with pytest.raises(NotDeployable):
            deployed(selected=False)

    def test_게이트를_끄면_배포할_수_있다(self) -> None:
        run = Deployment.deploy(
            DeploymentId.of("dep-1"),
            target(),
            artifact(selected=False),
            "2026-05-19 23:00:00",
            require_selected=False,
        )
        assert run.status is DeploymentStatus.DEPLOYED

    def test_첫_배포는_v1_이다(self) -> None:
        assert deployed().current_version.number == 1

    def test_배포하면_사건이_남는다(self) -> None:
        events = deployed().pull_events()
        assert [e.event_name for e in events] == ["ModelDeployed"]


class Test버전:
    def test_새_버전은_번호가_하나_올라간다(self) -> None:
        deployment = deployed()
        deployment.release(artifact(artifact_id="a2"), "2026-05-21 12:00:00")
        assert deployment.current_version.number == 2
        assert len(deployment.versions) == 2

    def test_그_시각에_돌던_버전을_찾는다(self) -> None:
        deployment = deployed()
        deployment.release(artifact(artifact_id="a2"), "2026-05-21 12:00:00")

        assert deployment.version_at("2026-05-20 12:00:00").number == 1
        assert deployment.version_at("2026-05-22 12:00:00").number == 2

    def test_배포_전_시각에는_아무것도_없다(self) -> None:
        assert deployed().version_at("2026-05-01 00:00:00") is None


class Test격리:
    def test_이유_없이_멈추지_않는다(self) -> None:
        with pytest.raises(InvariantViolation):
            deployed().quarantine("  ", "2026-05-23 00:00:00")

    def test_격리해도_버전은_그대로다(self) -> None:
        deployment = deployed()
        deployment.quarantine("입력 드리프트", "2026-05-23 00:00:00")

        assert deployment.status is DeploymentStatus.QUARANTINED
        assert deployment.current_version.number == 1
        assert not deployment.is_serving

    def test_두_번_멈추지_않는다(self) -> None:
        deployment = deployed()
        deployment.quarantine("첫 번째", "2026-05-23 00:00:00")
        deployment.pull_events()
        deployment.quarantine("두 번째", "2026-05-23 01:00:00")

        assert deployment.quarantine_reason == "첫 번째"
        assert deployment.pull_events() == ()

    def test_격리_중에는_새_버전을_올릴_수_없다(self) -> None:
        deployment = deployed()
        deployment.quarantine("입력 드리프트", "2026-05-23 00:00:00")
        with pytest.raises(IllegalStateTransition):
            deployment.release(artifact(artifact_id="a2"), "2026-05-23 01:00:00")

    def test_확인한_것을_적어야_다시_켠다(self) -> None:
        deployment = deployed()
        deployment.quarantine("입력 드리프트", "2026-05-23 00:00:00")

        with pytest.raises(InvariantViolation):
            deployment.resume("", "2026-05-23 02:00:00")

        deployment.resume("팬 교체 완료", "2026-05-23 02:00:00")
        assert deployment.status is DeploymentStatus.DEPLOYED
        assert deployment.quarantine_reason == ""

    def test_격리_상태가_아니면_해제할_것이_없다(self) -> None:
        with pytest.raises(IllegalStateTransition):
            deployed().resume("아무거나", "2026-05-23 00:00:00")


class Test롤백:
    def three(self) -> Deployment:
        deployment = deployed()
        deployment.release(artifact(artifact_id="a2"), "2026-05-21 12:00:00")
        deployment.release(artifact(artifact_id="a3"), "2026-05-22 12:00:00")
        return deployment

    def test_롤백은_새_버전을_만든다(self) -> None:
        deployment = self.three()
        version = deployment.rollback(1, "v3 문제", "2026-05-23 09:00:00")

        assert version.number == 4
        assert version.rolled_back_from == 3
        assert version.is_rollback
        assert version.artifact.artifact_id == "mv-1-tflite-fp16"
        assert deployment.status is DeploymentStatus.ROLLED_BACK

    def test_번호를_되감지_않으므로_시간에_답할_수_있다(self) -> None:
        deployment = self.three()
        deployment.rollback(1, "v3 문제", "2026-05-23 09:00:00")

        assert deployment.version_at("2026-05-22 18:00:00").number == 3
        assert deployment.version_at("2026-05-23 12:00:00").number == 4

    def test_돌아갈_곳이_없으면_롤백이_아니다(self) -> None:
        with pytest.raises(VersionNotFound):
            deployed().rollback(9, "없는 버전", "2026-05-23 00:00:00")

    def test_이유가_필요하다(self) -> None:
        with pytest.raises(InvariantViolation):
            self.three().rollback(1, "   ", "2026-05-23 00:00:00")

    def test_현재_버전으로는_돌아갈_수_없다(self) -> None:
        with pytest.raises(IllegalStateTransition):
            self.three().rollback(3, "현재 버전", "2026-05-23 00:00:00")

    def test_격리를_해소한다(self) -> None:
        deployment = self.three()
        deployment.quarantine("드리프트", "2026-05-23 00:00:00")
        deployment.rollback(1, "v1 로 복귀", "2026-05-23 09:00:00")

        assert deployment.quarantine_reason == ""
        assert deployment.is_serving

    def test_롤백하면_사건이_남는다(self) -> None:
        deployment = self.three()
        deployment.pull_events()
        deployment.rollback(1, "v3 문제", "2026-05-23 09:00:00")

        events = deployment.pull_events()
        assert [e.event_name for e in events] == ["DeploymentRolledBack"]
        assert events[0].from_version == 3
        assert events[0].to_version == 1
        assert events[0].new_version == 4

    def test_횟수를_센다(self) -> None:
        deployment = self.three()
        deployment.rollback(1, "첫 롤백", "2026-05-23 09:00:00")
        deployment.rollback(3, "다시 앞으로", "2026-05-23 12:00:00")
        assert deployment.rollback_count == 2


class Test내림:
    def test_내려간_배포는_아무것도_못_한다(self) -> None:
        deployment = deployed()
        deployment.retire("설비 폐기", "2026-06-01 00:00:00")

        with pytest.raises(IllegalStateTransition):
            deployment.quarantine("이제 와서", "2026-06-02 00:00:00")
        with pytest.raises(IllegalStateTransition):
            deployment.rollback(1, "이제 와서", "2026-06-02 00:00:00")


class Test대상:
    def test_DEVICE_는_한_대다(self) -> None:
        with pytest.raises(InvariantViolation):
            DeploymentTarget(
                kind=TargetKind.DEVICE, identifier="D-1", device_count=5
            )

    def test_0대짜리_배포는_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            target(device_count=0)

    def test_넓은_배포는_되돌리기_어렵다(self) -> None:
        assert target(kind=TargetKind.FLEET, identifier="ALL", device_count=800).is_wide
        assert not target().is_wide


class Test이력:
    def test_언제_무엇을_했는지_전부_남는다(self) -> None:
        deployment = deployed()
        deployment.release(artifact(artifact_id="a2"), "2026-05-21 12:00:00")
        deployment.quarantine("드리프트", "2026-05-23 00:00:00")
        deployment.rollback(1, "v1 로 복귀", "2026-05-23 09:00:00")

        history = deployment.history
        assert len(history) == 4
        assert [at for at, _ in history] == sorted(at for at, _ in history)
