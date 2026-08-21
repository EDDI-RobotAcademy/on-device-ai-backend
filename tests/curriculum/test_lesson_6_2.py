"""실습 6-2 — Edge의 데이터를 S3에 모아라.

    pytest -m lesson_6_2 -s

**어디에 두느냐가 나중에 무엇을 꺼낼 수 있는지를 정한다.**

그리고 이 결정은 되돌리기 어렵다.
객체 100만 개를 다시 배치하려면 100만 번 복사해야 한다.
"""

from __future__ import annotations

import pytest

from domain.fleet.object_key import (
    KeyLayout,
    KeyLayoutPolicy,
    ObjectKey,
    ObjectStats,
)
from domain.shared.errors import InvariantViolation
from tests.support import fleet_scenario as fs
from tests.support import report

pytestmark = pytest.mark.lesson_6_2


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def test_키_설계가_나중의_조회_비용을_정한다(fleet_env) -> None:
    report.section("실습 6-2 · Edge의 데이터를 S3에 모아라")

    view = fleet_env.lake
    report.block("객체 저장소", view.render())

    assert view.object_count > 0
    assert view.narrowed_count < view.object_count
    report.note(
        f"'{view.narrowed_prefix}' 로 좁히면 "
        f"{view.object_count}개 중 {view.narrowed_count}개만 훑는다."
    )
    report.note("**접두어가 곧 비용이다.** 좁히지 못하면 버킷 전체를 페이지 넘겨 훑는다.")


def test_파티션_순서대로여야_좁혀진다() -> None:
    layout = KeyLayout()
    report.block(
        "같은 조건, 다른 순서",
        "\n".join(
            [
                "  kind + device + date  → "
                + layout.prefix_for(
                    kind="inference_log", device="DEV-02", date="2026-05-23"
                ),
                "  device 만            → " + layout.prefix_for(device="DEV-02"),
            ]
        ),
    )
    assert layout.can_narrow(kind="inference_log")
    assert not layout.can_narrow(device="DEV-02")
    report.note(
        "device 만 줘서는 아무것도 못 좁힌다. **앞이 안 맞으면 뒤는 못 좁힌다.**"
    )
    report.note(
        "그래서 **자주 걸러내는 것을 앞에 둔다.** 이 순서가 곧 쿼리 패턴이다."
    )


def test_시각을_파일_이름에_넣으면_전부_훑어야_한다() -> None:
    """가장 흔한 실수."""
    flat = ObjectKey(
        prefix="uplinks", partitions=(), filename="2026-05-23T09-14-22-DEV-02.jsonl"
    )
    partitioned = KeyLayout().key_for(
        kind="inference_log",
        device_id="DEV-02",
        date="2026-05-23",
        hour="09",
        part=1,
    )
    report.block(
        "두 가지 키",
        "\n".join([f"  평평한 키   : {flat}", f"  파티션 키   : {partitioned}"]),
    )
    assert flat.partition_names == ()
    assert partitioned.partition_names == ("kind", "device", "date", "hour")
    report.note(
        "평평한 키로 '지난주 DEV-02' 를 뽑으려면 **전부 훑어야 한다.** "
        "객체가 100만 개면 100만 번이다."
    )


def test_파티션이_없으면_판정이_막는다() -> None:
    findings = KeyLayoutPolicy().inspect(
        KeyLayout(order=("kind",)),
        ObjectStats(object_count=1000, total_bytes=100_000_000),
    )
    assert "LAKE_MISSING_PARTITION" in codes(findings)
    report.note(
        "device 와 date 가 파티션에 없으면 그 저장소는 시간이 갈수록 못 쓰게 된다."
    )


def test_작은_파일_수만_개는_큰_파일_하나보다_비싸다() -> None:
    small = KeyLayoutPolicy().inspect(
        KeyLayout(),
        ObjectStats(object_count=50_000, total_bytes=50_000 * 4_000),
    )
    big = KeyLayoutPolicy().inspect(
        KeyLayout(),
        ObjectStats(object_count=500, total_bytes=500 * 4_000_000),
    )
    report.block(
        "같은 총량, 다른 파일 수",
        "\n".join(
            [
                "  4KiB × 50,000개  → " + str(sorted(codes(small))),
                "  4MiB × 500개     → " + str(sorted(codes(big))),
            ]
        ),
    )
    assert "LAKE_SMALL_FILES" in codes(small)
    assert "LAKE_SMALL_FILES" not in codes(big)
    report.note(
        "총량은 200GB 로 같다. 그런데 앞쪽은 요청이 100배 나가고, "
        "쿼리 엔진이 파일을 5만 번 연다."
    )
    report.note(
        "그래서 현장에서는 **작은 객체를 주기적으로 합친다**(compaction). "
        "그 작업을 나중에 하려면 파티션이 잘 짜여 있어야 한다."
    )


def test_한_접두어_아래_객체가_너무_많아도_문제다() -> None:
    findings = KeyLayoutPolicy().inspect(
        KeyLayout(),
        ObjectStats(
            object_count=200_000, total_bytes=200_000 * 200_000, distinct_prefixes=5
        ),
    )
    assert "LAKE_PREFIX_TOO_WIDE" in codes(findings)
    report.note(
        "접두어 하나에 4만 개가 들어 있으면 목록 조회가 페이지를 40번 넘긴다. "
        "**파티션을 더 잘게 나눠야 한다** — 시간까지 넣는 이유다."
    )


def test_키에_넣으면_안_되는_것이_있다() -> None:
    for bad in ("a//b", "a/../b", "a b"):
        with pytest.raises(InvariantViolation):
            ObjectKey(prefix="uplinks", partitions=(), filename=bad)
    report.note(
        "`//` 와 `..` 와 공백은 객체 저장소마다 다르게 해석한다. "
        "**여기서 막지 않으면 나중에 못 찾는 객체가 생긴다.**"
    )


def test_빈_파티션은_찾을_수_없는_자리를_만든다() -> None:
    with pytest.raises(InvariantViolation):
        ObjectKey(
            prefix="uplinks", partitions=(("device", "  "),), filename="part.jsonl"
        )
    report.note("device= 로 끝나는 접두어는 어떤 조회에도 안 걸린다.")


def test_실제로_S3에_올라가_있다(fleet_env) -> None:
    """moto 안이지만 **진짜 boto3 호출이다.**"""
    store = fleet_env.fleet.store
    layout = KeyLayout()
    keys = store.list_prefix(
        layout.prefix_for(kind="inference_log", device="DEV-02", date="2026-05-23")
    )
    report.block("실제 키", "\n".join(f"  {k}" for k in keys))

    assert keys
    assert all("device=DEV-02" in key for key in keys)
    body = store.get(
        layout.key_for(
            kind="inference_log",
            device_id="DEV-02",
            date="2026-05-23",
            hour="09",
            part=0,
        )
    )
    assert body
    report.note(
        "가짜 클라이언트를 세워 두고 '호출됐다'만 확인하면 "
        "API 이름이 틀려도 통과한다. **여기서는 틀리면 터진다.**"
    )
