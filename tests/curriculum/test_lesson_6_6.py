"""실습 6-6 — Cloud에서 만든 모델을 Edge로 보내라.

    pytest -m lesson_6_6 -s

모듈 5 에서 배웠다 — **결과물 파일만 보내면 조용히 다른 모델이 된다.**

수천 대에 보낼 때는 거기에 두 가지가 더 붙는다.

    체크섬   좁은 회선에서 잘려 도착한 파일을 알아채기 위해
    크기     **한 대 크기 × 대수**가 전송 예산에 맞는지
"""

from __future__ import annotations

import pytest

from domain.fleet.release import ReleasePolicy
from tests.support import fleet_scenario as fs
from tests.support import report

pytestmark = pytest.mark.lesson_6_6


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def test_릴리스는_파일이_아니라_묶음이다(fleet_env) -> None:
    report.section("실습 6-6 · Cloud에서 만든 모델을 Edge로 보내라")

    view = fleet_env.published
    report.block("릴리스 점검", view.render())

    bundle = fs.bundle()
    report.block(
        "묶음에 들어 있는 것",
        "\n".join(
            [
                f"  결과물     : {bundle.artifact_uri} ({bundle.artifact_bytes:,}B)",
                f"  체크섬     : {bundle.checksum[:16]}…",
                f"  전처리     : 정규화 {len(bundle.normalization)}채널, "
                f"표본 간격 {bundle.sample_interval_seconds}초, 창 {bundle.window_length}",
                f"  기준       : p95 {bundle.expected_p95_ms}ms, 예측 분포 "
                f"{len(bundle.expected_class_mix)}클래스",
                f"  계보       : {bundle.source_build_id} / {bundle.source_job_id}",
            ]
        ),
    )
    assert view.can_publish
    report.note(
        "**모듈 5 의 DeployedArtifactRef 가 요구하던 것이 전부 들어 있다.** "
        "다른 점은 회선을 건넌다는 것뿐이고, 그래서 체크섬과 크기가 붙는다."
    )


def test_전처리가_빠지면_조용히_다른_모델이_된다(fleet_env) -> None:
    view = fs.publish(
        fleet_env.fleet,
        fs.bundle(version="v2.0.1-no-preproc", normalization={}),
    )
    report.block("정규화 통계를 빼고 내보내려 할 때", view.render())

    assert "RELEASE_NO_PREPROCESSING" in codes(view)
    assert not view.can_publish
    report.note(
        "표본 간격도 마찬가지다. 30초로 솎아 넣으면 "
        "같은 30표본이 5분이 아니라 15분을 덮는다 (실습 5-1)."
    )


def test_표본_간격이_없어도_전처리가_없는_것이다(fleet_env) -> None:
    view = fs.publish(
        fleet_env.fleet,
        fs.bundle(version="v2.0.2-no-interval", sample_interval_seconds=0),
    )
    assert "RELEASE_NO_PREPROCESSING" in codes(view)
    report.note("**표본 간격도 모델의 일부다.** 정규화 통계만 보내는 것으로는 부족하다.")


def test_한_대에_작은_것이_수천_대면_달라진다(fleet_env) -> None:
    """**한 대 기준으로 보면 작아 보이는 것이 대수를 곱하면 달라진다.**"""
    view = fs.publish(
        fleet_env.fleet,
        fs.bundle(version="v2.1.0-big", artifact_bytes=200_000),
        policy=ReleasePolicy(max_fleet_transfer_mib=2.0),
    )
    report.block("200KB × 24대", view.render())

    assert "RELEASE_FLEET_TRANSFER_TOO_LARGE" in codes(view)
    report.note(
        f"한 대에 195KiB 는 작다. 24대면 {view.fleet_transfer_mib:.1f}MiB 다. "
        "3,000대면 572MiB 다 — 그건 회선 계획이 필요한 크기다."
    )


def test_플래시에_안_들어가면_보낼_수_없다(fleet_env) -> None:
    view = fs.publish(
        fleet_env.fleet,
        fs.bundle(version="v2.2.0-huge", artifact_bytes=600_000),
        policy=ReleasePolicy(max_artifact_kib=256.0),
    )
    assert "RELEASE_TOO_LARGE" in codes(view)
    assert not view.can_publish
    report.note(
        "이 숫자는 **모듈 4 의 DeviceBudget.storage_kib 과 같아야 한다.** "
        "배포 전에 지킨 예산을 배포할 때 다른 숫자로 재면 그 예산은 의미가 없다."
    )


def test_계보가_없으면_6개월_뒤에_못_답한다(fleet_env) -> None:
    view = fs.publish(
        fleet_env.fleet,
        fs.bundle(version="v2.3.0-orphan", source_build_id=""),
    )
    assert "RELEASE_NO_LINEAGE" in codes(view)
    assert not view.can_publish
    report.note(
        "'이 모델 뭐로 만들었죠?' 는 반드시 나온다. "
        "그때 답할 수 있게 하는 것이 여기서 한 줄 적어 두는 일이다 (실습 6-10)."
    )


def test_체크섬_없는_묶음은_만들_수도_없다() -> None:
    from domain.shared.errors import InvariantViolation

    with pytest.raises(InvariantViolation) as caught:
        fs.bundle(checksum="")
    report.note(str(caught.value))
    report.note(
        "디바이스가 스스로 검증할 수 있어야 한다. "
        "그래서 OTA 문서에도 체크섬이 들어간다 (실습 6-8)."
    )


def test_같은_버전_이름으로_다른_내용을_올리면_계보가_끊긴다(fleet_env) -> None:
    from domain.shared.errors import InvariantViolation
    from domain.fleet.identifiers import FleetId

    fleet = fleet_env.fleet.fleets.find_by_id(FleetId.of(fs.FLEET_ID))
    with pytest.raises(InvariantViolation) as caught:
        fleet.publish(fs.bundle(version="v2.0.0", artifact_bytes=99))
    report.note(str(caught.value))
    report.note(
        "현장에서 실제로 일어난다 — 급해서 같은 태그로 다시 빌드해 올린다. "
        "그러면 'v2.0.0' 이 두 가지를 뜻하게 된다."
    )


def test_등록만_하고_아직_아무_디바이스에도_안_갔다(fleet_env) -> None:
    from domain.fleet.identifiers import FleetId

    fleet = fleet_env.fleet.fleets.find_by_id(FleetId.of(fs.FLEET_ID))
    on_new = fleet.devices_on("v2.0.0")

    assert fleet.release_of("v2.0.0") is not None
    assert on_new == ()
    report.note(
        "**등록과 배포는 다른 일이다.** 등록은 즉시 끝나고, 배포는 며칠 걸린다 (실습 6-8)."
    )
