"""실습 5-2 — 배포된 모델에 Version을 부여하라.

    pytest -m lesson_5_2 -s

현장에서 실제로 나오는 질문은 이런 것들이다.

    "어제 3시에 3라인에서 돌던 게 뭐였죠?"
    "그거 언제부터 돌았어요?"
    "그 전 건요?"

**이 질문들은 배포 버전으로만 답할 수 있다.**
"""

from __future__ import annotations

import pytest

from domain.operations.version import DeploymentVersion
from domain.shared.errors import InvariantViolation
from tests.support import operations_scenario as os5
from tests.support import report

pytestmark = pytest.mark.lesson_5_2


def test_버전은_세_가지가_있고_서로_다르다(deployed) -> None:
    report.section("실습 5-2 · 배포된 모델에 Version을 부여하라")

    view = deployed.deploy_result.deployment
    report.block(
        "세 가지 버전",
        "\n".join(
            [
                f"  모델 버전  : {view.model_version_id}   어떤 학습이 만든 가중치인가",
                f"  결과물     : {view.current_artifact}   어떤 형식으로 바꿨는가",
                f"  배포 버전  : v{view.current_version}"
                "                          언제 어디에 올라갔는가",
            ]
        ),
    )
    report.note(
        "같은 모델을 두 번 배포하면 모델 버전은 같고 배포 버전은 다르다. "
        "현장 질문은 대개 배포 버전으로만 답할 수 있다."
    )
    assert view.current_version == 1


def test_새_버전을_올리면_번호가_올라간다(operations_container, deployed) -> None:
    os5.release(
        operations_container,
        deployed.optimized,
        deployed.trained,
        note="INT8 로 교체",
        artifact_label="TFLITE/INT8",
        released_at="2026-05-21 12:00:00",
        require_selected=False,
    )
    view = os5.release(
        operations_container,
        deployed.optimized,
        deployed.trained,
        note="ONNX 로 교체",
        artifact_label="ONNX/FP32",
        released_at="2026-05-22 12:00:00",
        require_selected=False,
    ).deployment

    report.block("배포 이력", view.render())
    assert view.current_version == 3
    assert view.version_count == 3


def test_그_시각에_무엇이_돌고_있었는지_답할_수_있다(operations_container, deployed) -> None:
    """**이것이 버전을 매기는 이유다.**"""
    from domain.operations.identifiers import DeploymentId

    os5.release(
        operations_container,
        deployed.optimized,
        deployed.trained,
        artifact_label="TFLITE/INT8",
        released_at="2026-05-21 12:00:00",
        require_selected=False,
        note="INT8 로 교체",
    )
    deployment = operations_container.deployments.find_by_id(
        DeploymentId.of(deployed.deployment_id)
    )

    before = deployment.version_at("2026-05-21 03:00:00")
    after = deployment.version_at("2026-05-22 03:00:00")

    report.block(
        "그 시각에 돌던 버전",
        "\n".join(
            [
                f"  2026-05-21 03:00 → {before.label}  ({before.artifact.label})",
                f"  2026-05-22 03:00 → {after.label}  ({after.artifact.label})",
            ]
        ),
    )
    assert before.number == 1
    assert after.number == 2
    report.note(
        "이 질문에 답할 수 있는 이유는 하나다 — **버전 번호를 되감지 않기 때문이다.**"
    )


def test_격리된_배포에는_새_버전을_올릴_수_없다(operations_container, deployed) -> None:
    from domain.shared.errors import IllegalStateTransition

    os5.quarantine(operations_container, reason="실습용 격리")
    with pytest.raises(IllegalStateTransition) as caught:
        os5.release(
            operations_container,
            deployed.optimized,
            deployed.trained,
            require_selected=False,
        )
    report.note(str(caught.value))
    report.note(
        "멈춰 세운 상태에서 새 것을 밀어 넣으면 무엇이 문제였는지 영영 모른다."
    )


def test_언제_올라갔는지_없으면_버전이_아니다() -> None:
    with pytest.raises(InvariantViolation):
        DeploymentVersion(
            number=1,
            artifact=_artifact(),
            released_at="   ",
        )
    report.note("시각 없는 버전으로는 '언제부터'에 답할 수 없다.")


def test_롤백_버전은_앞선_버전에서만_온다() -> None:
    with pytest.raises(InvariantViolation):
        DeploymentVersion(
            number=2, artifact=_artifact(), released_at="2026-05-21", rolled_back_from=5
        )


def _artifact():  # noqa: ANN202
    from domain.operations.artifact_ref import DeployedArtifactRef

    return DeployedArtifactRef(
        artifact_id="a",
        optimization_run_ref="opt",
        model_version_id="mv-1",
        runtime="TFLITE",
        precision="FP16",
        size_bytes=1,
        class_labels=("A", "B"),
        selected=True,
    )
