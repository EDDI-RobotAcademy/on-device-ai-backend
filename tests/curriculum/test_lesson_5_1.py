"""실습 5-1 — 모델을 처음으로 현장에 배포하라.

    pytest -m lesson_5_1 -s

모듈 4 를 통과한 결과물이 있다. 파일도 있다. 그러면 올려도 되는가?

**파일이 준비된 것과 배포가 준비된 것은 다르다.**
"""

from __future__ import annotations

import pytest

from domain.operations.artifact_ref import DeployedArtifactRef
from domain.operations.errors import NotDeployable
from domain.operations.release_check import ReleasePolicy
from domain.operations.target import DeploymentTarget, TargetKind
from tests.support import operations_scenario as os5
from tests.support import report

pytestmark = pytest.mark.lesson_5_1


def artifact(**overrides) -> DeployedArtifactRef:  # noqa: ANN003
    base: dict[str, object] = dict(
        artifact_id="mv-1-tflite-fp16",
        optimization_run_ref="opt-power",
        model_version_id="mv-1",
        runtime="TFLITE",
        precision="FP16",
        size_bytes=11_724,
        class_labels=("FAULT", "OVERLOAD", "NORMAL"),
        input_fields=("active_power_kw", "temperature_c"),
        expected_p95_ms=0.0031,
        expected_accuracy=1.0,
        expected_class_mix={"NORMAL": 0.79, "OVERLOAD": 0.17, "FAULT": 0.04},
        normalization={"active_power_kw": (147.8, 39.8)},
        selected=True,
    )
    base.update(overrides)
    return DeployedArtifactRef(**base)  # type: ignore[arg-type]


def codes(check) -> set[str]:  # noqa: ANN001
    return {f.code for f in check.findings}


def test_배포는_파일_하나를_보내는_일이_아니다(deployed) -> None:
    report.section("실습 5-1 · 모델을 처음으로 현장에 배포하라")

    result = deployed.deploy_result
    report.block("배포 전 점검", result.check.render())
    report.block("배포 결과", result.deployment.render())

    assert result.deployment.status == "DEPLOYED"
    assert result.deployment.current_version == 1
    report.note(
        "네 가지가 함께 나갔다 — 결과물, 전처리 통계, 기준 숫자, 그리고 관측."
    )
    assert result.watch_id
    report.note(
        f"관측 '{result.watch_id}' 이 **배포와 동시에** 열렸다. "
        "나중에 켜면 그 사이 구간은 영영 비어 있다."
    )


def test_선택_판정을_통과하지_않으면_배포_대상이_아니다() -> None:
    from domain.operations.deployment import Deployment
    from domain.operations.identifiers import DeploymentId

    with pytest.raises(NotDeployable) as caught:
        Deployment.deploy(
            DeploymentId.of("dep-x"),
            DeploymentTarget(kind=TargetKind.DEVICE, identifier="D-1"),
            artifact(selected=False),
            "2026-05-19 23:00:00",
        )
    report.note(str(caught.value))
    report.note(
        "예산 안에 드는지 확인하지 않은 결과물이 나가면 "
        "모듈 4 의 예산은 처음부터 없었던 것과 같다."
    )


def test_전처리_통계가_빠지면_조용히_다른_모델이_된다() -> None:
    """**전처리는 모델의 일부다.**"""
    check = ReleasePolicy().inspect(
        artifact(normalization={}),
        DeploymentTarget(kind=TargetKind.DEVICE, identifier="D-1"),
    )
    report.block("정규화 통계를 빼고 내보내려 할 때", check.render())

    assert "RELEASE_NO_PREPROCESSING" in codes(check)
    assert not check.can_release
    report.note(
        "TFLite 파일에는 정규화 통계가 없다. 그것을 따로 안 보내면 "
        "디바이스가 다른 전처리를 하고, 모델은 학습 때와 다른 입력을 받는다."
    )
    report.note(
        "실습 4-2 의 변환 동등성을 아무리 확인해도 이건 안 잡힌다 — "
        "**같은 입력을 넣지 않았기 때문이다.**"
    )


def test_기준_숫자가_없으면_나중에_아무것도_말할_수_없다() -> None:
    check = ReleasePolicy().inspect(
        artifact(expected_p95_ms=0.0, expected_class_mix={}),
        DeploymentTarget(kind=TargetKind.DEVICE, identifier="D-1"),
    )
    assert "RELEASE_NO_BASELINE" in codes(check)
    report.note(
        "'느려졌다' 를 말하려면 원래 얼마였는지를 알아야 한다. "
        "그 숫자는 모듈 4 의 벤치마크에서 온다 — 배포할 때 함께 가져가야 한다."
    )


def test_첫_배포는_좁아야_한다() -> None:
    """되돌릴 곳이 없는 배포는 넓으면 안 된다."""
    wide = ReleasePolicy().inspect(
        artifact(),
        DeploymentTarget(
            kind=TargetKind.FLEET, identifier="ALL", device_count=1_200
        ),
        previous_versions=0,
    )
    report.block("첫 배포를 1,200대에 한 번에", wide.render())

    assert "RELEASE_FIRST_TOO_WIDE" in codes(wide)
    assert not wide.can_release
    report.note(
        "**첫 배포에는 롤백 대상이 없다.** 문제가 나면 격리(실습 5-8)밖에 못 한다."
    )

    narrow = ReleasePolicy().inspect(
        artifact(),
        DeploymentTarget(kind=TargetKind.DEVICE_GROUP, identifier="LINE-3", device_count=3),
        previous_versions=0,
    )
    assert narrow.can_release
    assert "RELEASE_FIRST_NO_ROLLBACK_TARGET" in codes(narrow)
    report.note("3대면 통과한다 — 대신 '돌아갈 곳이 없다'는 사실은 기록으로 남는다.")

    # 두 번째 배포부터는 넓혀도 된다. 돌아갈 곳이 생겼기 때문이다.
    later = ReleasePolicy().inspect(
        artifact(),
        DeploymentTarget(kind=TargetKind.FLEET, identifier="ALL", device_count=1_200),
        previous_versions=1,
    )
    assert later.can_release
    report.note("2차 배포는 1,200대도 통과한다 — v1 으로 돌아갈 수 있기 때문이다.")


def test_없는_배포는_없다고_말한다(operations_container) -> None:
    from application.operations.deploy_model import GetDeploymentQuery
    from domain.operations.errors import DeploymentNotFound

    with pytest.raises(DeploymentNotFound):
        operations_container.get_deployment().execute(
            GetDeploymentQuery(deployment_id="없음")
        )
