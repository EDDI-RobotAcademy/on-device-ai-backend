"""실습 6-10 — Edge → Cloud → Edge 순환 구조를 완성하라.

    pytest -m lesson_6_10 -s

순환이 닫혔다는 것을 무엇으로 증명하는가?

**계보(lineage)로 증명한다.**
사슬이 한 칸이라도 끊기면 순환이 아니라 **일방통행**이다 —
데이터는 올라갔고 모델은 내려왔는데, 둘이 이어져 있다는 증거가 없다.
"""

from __future__ import annotations

import pytest

from domain.fleet.lineage import LineagePolicy, trace_of
from tests.support import fleet_scenario as fs
from tests.support import report

pytestmark = pytest.mark.lesson_6_10


@pytest.fixture
def closed(fleet_env):  # noqa: ANN001, ANN201
    """한 바퀴를 다 돈 상태."""
    fs.plan(fleet_env.fleet)
    fs.collect(fleet_env.fleet)
    fs.advance(fleet_env.fleet)
    fs.collect(fleet_env.fleet)
    fs.apply_outcomes(fleet_env.fleet)
    return fleet_env


def test_한_바퀴가_이어져_있다(closed) -> None:
    report.section("실습 6-10 · Edge → Cloud → Edge 순환 구조를 완성하라")

    view = fs.trace(closed.fleet, "DEV-00")
    report.block("계보", view.render())

    assert view.closed
    assert not view.broken_stages
    report.note(
        "**지금 DEV-00 에서 돌고 있는 모델이 어느 데이터에서 왔는지** 다섯 칸으로 답한다."
    )


def test_여섯_모듈이_한_바퀴를_돈다(closed) -> None:
    report.block(
        "한 바퀴",
        "\n".join(
            [
                "  모듈 1·2  현장 데이터를 검증하고 품질을 판정한다",
                "  모듈 3    모델을 학습하고 승인한다",
                "  모듈 4    디바이스에 올릴 수 있게 바꾸고 고른다",
                "  모듈 5    배포하고 지켜보다 재학습이 필요하다고 판단한다",
                "  모듈 6-1~3  현장 데이터를 클라우드로 올린다",
                "  모듈 6-4    그것으로 다시 학습 데이터를 만든다   ← **모듈 1 로 돌아간다**",
                "  모듈 6-5~9  클라우드에서 학습하고 현장으로 내보낸다",
            ]
        ),
    )
    view = fs.trace(closed.fleet, "DEV-00")
    assert view.closed
    report.note(
        "모듈 5-11 의 RetrainingRequested 가 이 순환의 시작 신호였고, "
        "여기 계보가 그것이 실제로 돌았다는 증거다."
    )


def test_사슬이_끊기면_그_위로_못_올라간다() -> None:
    broken = trace_of(
        device_id="DEV-00",
        version="v2.0.0",
        job_id="train-2026-05-24",
        build_id="",  # 어느 데이터셋에서 왔는지 안 남겼다
        window="2026-05-22 ~ 2026-05-23",
    )
    closure = LineagePolicy().inspect(
        broken, source_devices=("DEV-00", "DEV-01")
    )
    report.block("데이터셋 칸이 빈 계보", closure.render())

    assert not closure.closed
    assert "LINEAGE_BROKEN" in {f.code for f in closure.findings}
    report.note(
        "학습까지는 되짚어 올라갔는데 거기서 멈춘다. "
        "**'이 모델 뭐로 만들었죠?' 에 답할 수 없다.**"
    )


def test_어느_디바이스_데이터로_학습했는지도_남아야_한다() -> None:
    trace = trace_of(
        device_id="DEV-00",
        version="v2.0.0",
        job_id="train-1",
        build_id="build-1",
        window="2026-05-22 ~ 2026-05-23",
    )
    closure = LineagePolicy().inspect(trace, source_devices=())
    assert "LINEAGE_NO_SOURCE_DEVICES" in {f.code for f in closure.findings}
    report.note(
        "'이 디바이스만 이상한데 얘 데이터로 학습한 적 있나요?' — "
        "이 질문은 반드시 나온다."
    )


def test_학습에_안_들어간_디바이스도_알고_있어야_한다(closed) -> None:
    """**틀린 것은 아니다. 다만 알고 있어야 한다.**"""
    view = fs.trace(
        closed.fleet,
        "DEV-00",
        source_devices=("DEV-05", "DEV-06"),  # DEV-00 은 학습에 안 들어갔다
    )
    report.block("자기 데이터가 학습에 안 들어간 디바이스", view.render())

    assert view.closed  # INFO 라서 순환은 닫혀 있다
    assert "LINEAGE_SELF_EXCLUDED" in {f.code for f in view.findings}
    report.note(
        "격리됐거나 나중에 설치된 디바이스라면 정상이다. "
        "그런데 **그 디바이스에서 이상이 나면 이 사실이 첫 번째 단서다.**"
    )


def test_모델을_안_받은_디바이스는_계보가_없다(fleet_env) -> None:
    view = fs.trace(fleet_env.fleet, "DEV-20")
    report.block("아직 v1.0.0 인 디바이스", view.render())

    assert not view.closed
    assert {"학습", "데이터셋"} <= set(view.broken_stages)
    report.note(
        "**버전 이름은 있는데 그 위가 없다.** v1.0.0 은 이 플릿에 "
        "등록된 적이 없는 버전이다 — 공장 초기 이미지로 깔려 나온 것이다."
    )
    report.note(
        "현장에서 실제로 이렇다. 공장 초기 이미지는 대개 추적이 안 된다. "
        "**그 사실을 아는 것과 모르는 것이 다르다.**"
    )


def test_계보는_S3의_명세에서_시작된다(closed) -> None:
    import json

    from domain.fleet.object_key import ObjectKey

    manifest = json.loads(
        closed.fleet.store.get(
            ObjectKey(
                prefix="datasets",
                partitions=(("build", "build-2026-05-24"),),
                filename="manifest.json",
            )
        )
    )
    report.block(
        "사슬의 가장 위 칸",
        "\n".join(
            [
                f"  구간     : {manifest['window']['started_at']} ~ "
                f"{manifest['window']['ended_at']}",
                f"  이유     : {manifest['window']['reason']}",
                f"  디바이스 : {len(manifest['devices'])}대",
            ]
        ),
    )
    assert manifest["devices"]
    report.note(
        "**계보는 기록해 둔 만큼만 있다.** "
        "이 manifest 를 안 남겼으면 여기서 끊긴다."
    )
