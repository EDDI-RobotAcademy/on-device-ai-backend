"""실습 5-10 — 망가진 모델을 Rollback하라.

    pytest -m lesson_5_10 -s

이 실습의 설계 결정 하나가 나머지를 전부 설명한다.

    **롤백은 되돌리기가 아니라 새 배포다.**

v3 에서 v1 로 돌아가면 결과는 v4 다 (내용은 v1).
버전 번호는 줄어들지 않는다 — 그래야 "어제 3시에 무엇이 돌았는가"에 답할 수 있다.
번호를 되감으면 그 시간의 기록이 사라진다.
"""

from __future__ import annotations

import pytest

from domain.operations.errors import VersionNotFound
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from tests.support import operations_scenario as os5
from tests.support import report

pytestmark = pytest.mark.lesson_5_10


@pytest.fixture
def three_versions(operations_container, deployed):  # noqa: ANN001, ANN201
    """v1(FP16) → v2(INT8) → v3(ONNX) 까지 올려 둔 배포."""
    os5.release(
        operations_container,
        deployed.optimized,
        deployed.trained,
        artifact_label="TFLITE/INT8",
        released_at="2026-05-21 12:00:00",
        require_selected=False,
        note="크기를 줄여 본다",
    )
    os5.release(
        operations_container,
        deployed.optimized,
        deployed.trained,
        artifact_label="ONNX/FP32",
        released_at="2026-05-22 12:00:00",
        require_selected=False,
        note="정확도를 되찾아 본다",
    )
    return operations_container


def test_롤백은_새_버전을_만든다(three_versions) -> None:
    report.section("실습 5-10 · 망가진 모델을 Rollback하라")

    view = os5.rollback(
        three_versions,
        to_version=1,
        reason="v3 배포 후 FAULT 재현율 붕괴",
        occurred_at="2026-05-23 09:00:00",
    )
    report.block("롤백 후 배포 이력", view.render())

    assert view.current_version == 4
    assert view.version_count == 4
    assert view.rollback_count == 1
    assert view.status == "ROLLED_BACK"
    report.note(
        "**v3 → v1 로 돌아갔는데 결과는 v4 다.** 내용은 v1 과 같고 번호는 새것이다."
    )


def test_번호를_되감지_않기_때문에_시간에_답할_수_있다(three_versions) -> None:
    """**이것이 새 버전을 만드는 이유의 전부다.**"""
    from domain.operations.identifiers import DeploymentId

    os5.rollback(
        three_versions,
        to_version=1,
        reason="v3 배포 후 FAULT 재현율 붕괴",
        occurred_at="2026-05-23 09:00:00",
    )
    deployment = three_versions.deployments.find_by_id(
        DeploymentId.of(os5.DEPLOYMENT_ID)
    )

    moments = [
        ("2026-05-20 12:00:00", 1),
        ("2026-05-21 18:00:00", 2),
        ("2026-05-22 18:00:00", 3),
        ("2026-05-23 12:00:00", 4),
    ]
    report.block(
        "그 시각에 돌던 버전",
        "\n".join(
            f"  {moment} → {deployment.version_at(moment).label} "
            f"({deployment.version_at(moment).artifact.label})"
            for moment, _ in moments
        ),
    )
    for moment, expected in moments:
        assert deployment.version_at(moment).number == expected
    report.note(
        "번호를 v1 로 되감았다면 '5-22 18시' 와 '5-23 12시' 를 구분할 수 없다. "
        "둘 다 v1 이 되기 때문이다."
    )


def test_돌아갈_곳이_없으면_롤백이_아니다(operations_container) -> None:
    with pytest.raises(VersionNotFound) as caught:
        os5.rollback(operations_container, to_version=7, reason="아무거나")
    report.note(str(caught.value))
    report.note(
        "첫 배포 직후에는 이 상황이 된다. 그래서 첫 배포는 좁게 한다 (실습 5-1)."
    )


def test_이유_없는_롤백은_다음_사람에게_아무것도_알려주지_않는다(three_versions) -> None:
    with pytest.raises(InvariantViolation) as caught:
        os5.rollback(three_versions, to_version=1, reason="  ")
    report.note(str(caught.value))
    report.note(
        "'일단 되돌림' 이라고 적힌 기록은 반년 뒤에 아무 도움이 안 된다."
    )


def test_이미_돌고_있는_버전으로는_돌아갈_수_없다(three_versions) -> None:
    with pytest.raises(IllegalStateTransition) as caught:
        os5.rollback(three_versions, to_version=3, reason="현재 버전으로")
    report.note(str(caught.value))


def test_격리된_상태에서도_롤백은_된다(three_versions) -> None:
    """격리는 멈춤이고 롤백은 조치다. 순서가 자연스럽다."""
    windows = os5.windows(three_versions)
    os5.observe(three_versions, windows[-1])
    os5.quarantine(three_versions)

    view = os5.rollback(
        three_versions,
        to_version=1,
        reason="입력 드리프트 확인, v1 로 복귀",
        occurred_at="2026-05-23 10:00:00",
    )
    assert view.status == "ROLLED_BACK"
    assert not view.quarantine_reason
    report.note(
        "멈춘다(격리) → 원인을 본다 → 되돌린다(롤백). "
        "롤백이 되면 격리 사유는 해소된 것으로 본다."
    )


def test_롤백해도_나아진다는_보장은_없다(three_versions, deployed) -> None:
    """**그래서 격리가 먼저다.**"""
    report.note(
        "v1 로 되돌리면 나아지는가? 입력이 변한 것이라면(실습 5-7) "
        "**v1 도 그 입력을 본 적이 없다.**"
    )
    report.note(
        "v1 과 v3 는 같은 학습 데이터에서 나왔다. "
        "형식만 다를 뿐(TFLITE/FP16 vs ONNX/FP32) 배운 것은 같다."
    )
    os5.rollback(
        three_versions, to_version=1, reason="v3 문제 확인", occurred_at="2026-05-23 09:00:00"
    )
    report.note(
        "롤백이 듣는 경우는 **새 모델 자체가 나빴을 때**다. "
        "세상이 변한 것이면 롤백이 아니라 재학습이다 (실습 5-11)."
    )


def test_되돌린_기록은_이력에_남는다(three_versions) -> None:
    view = os5.rollback(
        three_versions,
        to_version=2,
        reason="ONNX 런타임 메모리 초과",
        occurred_at="2026-05-23 09:00:00",
    )
    report.block(
        "배포 이력 전체",
        "\n".join(f"  {at}  {what}" for at, what in view.history),
    )
    assert any("롤백" in what for _, what in view.history)
    assert view.versions[-1].endswith("에서 롤백") or "롤백" in view.versions[-1]
    report.note("사건 조사는 이 표부터 읽는다.")
