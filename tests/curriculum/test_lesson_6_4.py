"""실습 6-4 — Cloud에서 다시 학습 데이터를 만들어라.

    pytest -m lesson_6_4 -s

**여기가 순환이 닫히는 지점이다.** 모듈 1 로 돌아간다.

그런데 올라온 것을 전부 쓰면 안 된다.
현장 데이터에는 학습에 넣으면 안 되는 것이 섞여 있다.
"""

from __future__ import annotations

import json

import pytest

from domain.fleet.dataset_build import (
    DatasetBuildPolicy,
    DatasetBuildSpec,
    SourceWindow,
)
from domain.fleet.device import DeviceStatus
from tests.support import fleet_scenario as fs
from tests.support import report

pytestmark = pytest.mark.lesson_6_4


def spec(**overrides) -> DatasetBuildSpec:  # noqa: ANN003
    base: dict[str, object] = dict(
        build_id="b1",
        window=SourceWindow(
            started_at="2026-05-22 00:00:00", ended_at="2026-05-23 23:59:59"
        ),
        device_ids=("DEV-00", "DEV-01", "DEV-02"),
        record_counts={"DEV-00": 4000, "DEV-01": 4000, "DEV-02": 4000},
        labeled_counts={"DEV-00": 500, "DEV-01": 500, "DEV-02": 500},
        label_distribution={"NORMAL": 1000, "OVERLOAD": 400, "FAULT": 100},
    )
    base.update(overrides)
    return DatasetBuildSpec(**base)  # type: ignore[arg-type]


def codes(check) -> set[str]:  # noqa: ANN001
    return {f.code for f in check.findings}


def test_현장_데이터로_학습_데이터셋을_만든다(fleet_env) -> None:
    report.section("실습 6-4 · Cloud에서 다시 학습 데이터를 만들어라")

    view = fleet_env.dataset
    report.block("데이터셋 계획", view.render())

    assert view.can_build
    assert view.dataset_uri.startswith("s3://")
    report.note(
        "이 명세가 **모듈 1 의 입력**이 된다. "
        "그리고 6개월 뒤 계보의 한 칸이 된다 (실습 6-10)."
    )


def test_격리된_디바이스는_자동으로_빠진다(fleet_env) -> None:
    """**이상한 상태에서 낸 판단으로 다시 학습하면 이상을 학습한다.**"""
    fs.mark(fleet_env.fleet, "DEV-05", DeviceStatus.QUARANTINED, "드리프트로 격리")
    fs.mark(fleet_env.fleet, "DEV-06", DeviceStatus.UNREACHABLE)

    view = fs.build_dataset(fleet_env.fleet, build_id="build-after-quarantine")
    report.block("제외된 디바이스", view.render())

    excluded = {device for device, _ in view.excluded}
    assert "DEV-05" in excluded
    assert "DEV-06" in excluded
    assert all(reason for _, reason in view.excluded)
    report.note(
        "**뺐다는 사실과 이유가 기록에 남는다.** "
        "6개월 뒤 '왜 이 디바이스 데이터가 없죠?'에 답할 수 있다."
    )


def test_이유_없이_뺀_것은_막는다() -> None:
    check = DatasetBuildPolicy().inspect(
        spec(excluded_devices=(("DEV-09", ""),))
    )
    assert "BUILD_UNEXPLAINED_EXCLUSION" in codes(check)
    report.note(
        "**무엇을 뺐는지가 무엇을 넣었는지만큼 중요하다.** "
        "빼는 것으로 결과를 바꿀 수 있기 때문이다."
    )


def test_현장_데이터가_많은_것과_학습할_수_있는_것은_다르다() -> None:
    check = DatasetBuildPolicy().inspect(
        spec(labeled_counts={"DEV-00": 20, "DEV-01": 20, "DEV-02": 20})
    )
    report.block("표본 12,000건, 라벨 60건", check.render())

    assert "BUILD_TOO_FEW_LABELS" in codes(check)
    assert not check.can_build
    report.note(
        "표본은 12,000건이다. 그런데 학습시킬 수 있는 것은 60건이다. "
        "**나머지 11,940건은 정답을 모른다** (실습 5-3)."
    )


def test_소수_클래스가_비면_그_클래스는_안_나아진다() -> None:
    check = DatasetBuildPolicy().inspect(
        spec(label_distribution={"NORMAL": 1400, "OVERLOAD": 90, "FAULT": 10})
    )
    assert "BUILD_MINORITY_LABEL_STARVED" in codes(check)
    report.note(
        "FAULT 가 10건이다. 재학습해도 FAULT 는 나아지지 않는다 — "
        "실습 5-11 의 blocker 가 여기서 실제로 막는다."
    )


def test_한_디바이스가_절반을_넘으면_그_디바이스를_학습한다() -> None:
    """**자주 놓치는 실수.**"""
    check = DatasetBuildPolicy().inspect(
        spec(record_counts={"DEV-00": 12_000, "DEV-01": 500, "DEV-02": 500})
    )
    report.block("DEV-00 이 92%", check.render())

    assert "BUILD_DEVICE_DOMINATED" in codes(check)
    report.note(
        "DEV-00 이 다른 대보다 스무 배 많이 올렸다. "
        "**새 모델은 DEV-00 의 버릇을 학습한다.**"
    )
    report.note(
        "회선이 좋은 디바이스, 가동률이 높은 라인에서 자연스럽게 생긴다. "
        "표본 수를 맞추거나 가중치를 주지 않으면 그대로 편향이 된다."
    )


def test_한_대에서만_모으면_그_한_대에_맞춘다() -> None:
    check = DatasetBuildPolicy().inspect(
        spec(
            device_ids=("DEV-00",),
            record_counts={"DEV-00": 12_000},
            labeled_counts={"DEV-00": 1_500},
        )
    )
    assert "BUILD_TOO_FEW_DEVICES" in codes(check)
    report.note("설비마다 조금씩 다르다. 한 대만 보면 그 한 대에 맞춘 모델이 된다.")


def test_명세가_그대로_S3에_남는다(fleet_env) -> None:
    """계보의 한 칸이다."""
    from domain.fleet.object_key import ObjectKey

    manifest = fleet_env.fleet.store.get(
        ObjectKey(
            prefix="datasets",
            partitions=(("build", "build-2026-05-24"),),
            filename="manifest.json",
        )
    )
    body = json.loads(manifest)
    report.block(
        "manifest.json",
        "\n".join(
            [
                f"  build_id : {body['build_id']}",
                f"  구간     : {body['window']['started_at']} ~ {body['window']['ended_at']}",
                f"  이유     : {body['window']['reason']}",
                f"  디바이스 : {len(body['devices'])}대",
                f"  제외     : {len(body['excluded'])}대",
            ]
        ),
    )
    assert body["build_id"] == "build-2026-05-24"
    assert body["window"]["reason"]
    report.note(
        "**구간을 왜 그렇게 잡았는지**까지 남는다. "
        "드리프트 시작 이후만 쓴 것인지, 전부 쓴 것인지가 결과를 가른다."
    )


def test_학습에_쓸_수_있는_디바이스가_없으면_막는다(fleet_bare) -> None:
    from application.shared.errors import ConflictingRequest

    fs.create(fleet_bare)
    for device in fs.devices():
        fs.mark(fleet_bare, device.device_id, DeviceStatus.QUARANTINED)

    with pytest.raises(ConflictingRequest) as caught:
        fs.build_dataset(fleet_bare)
    report.note(str(caught.value))
    report.note(
        "전부 격리된 상태에서 재학습하려는 것은 "
        "**이상한 데이터로 이상을 고치려는 것**이다."
    )
