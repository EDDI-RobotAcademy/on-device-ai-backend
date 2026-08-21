"""실습 5-15 — 파일로 보낼 것인가, 컨테이너로 보낼 것인가.

    pytest -m lesson_5_15 -s

모듈 6 의 OTA 는 "모델 파일을 내려보낸다"를 전제로 한다.
그런데 그 전제 자체가 선택이고, 다른 선택지의 성질이 완전히 다르다.

    모델 파일만    9 KiB. 회선이 좁아도 된다.
                   **대신 디바이스의 전처리·라벨 순서가 이미 맞아야 한다.**
                   어긋나도 아무 에러가 안 난다 (실습 5-12).

    컨테이너       모델 + 런타임 + 전처리를 한 덩어리로.
                   "내 노트북에서는 됐는데"가 사라진다.
                   **대신 수백 MiB 이고, 컨테이너가 도는 보드여야 한다.**

정답이 하나가 아니다. 그래서 고르지 않고 **비교표를 만든다.**
"""

from __future__ import annotations

import pytest

from domain.operations.packaging import (
    DeviceCapability,
    PackagingComparison,
    PackagingKind,
    PackagingOption,
    PackagingPolicy,
)
from domain.shared.errors import InvariantViolation
from tests.support import report

pytestmark = pytest.mark.lesson_5_15

OPTIONS = (
    PackagingOption(
        kind=PackagingKind.MODEL_FILE, payload_bytes=9_500, rollback_seconds=20.0
    ),
    PackagingOption(
        kind=PackagingKind.BUNDLE, payload_bytes=48_000, rollback_seconds=30.0
    ),
    PackagingOption(
        kind=PackagingKind.CONTAINER,
        payload_bytes=180 * 1024 * 1024,
        rollback_seconds=240.0,
    ),
)


def _comparison(**overrides):  # noqa: ANN003, ANN202
    base = dict(
        supports_container=True,
        free_storage_bytes=2 * 1024 * 1024 * 1024,
        uplink_bytes_per_second=256 * 1024,
        device_count=3_000,
    )
    base.update(overrides)
    return PackagingComparison(device=DeviceCapability(**base), options=OPTIONS)


def test_같은_모델_다른_배포_방식(  ) -> None:
    report.section("실습 5-15 · 파일로 보낼 것인가, 컨테이너로 보낼 것인가")

    comparison = _comparison()
    report.block("배포 방식 비교", comparison.render())

    model_file = comparison.option_of(PackagingKind.MODEL_FILE)
    container = comparison.option_of(PackagingKind.CONTAINER)
    assert container.payload_bytes > model_file.payload_bytes * 1000
    report.note(
        "같은 모델을 보내는데 전송량이 2만 배 차이난다. "
        "**3,000대를 곱하면 9 KiB 는 27 MiB 이고 180 MiB 는 527 GiB 다** (모듈 6)."
    )


def test_MCU_에서는_선택지_자체가_없다() -> None:
    comparison = _comparison(supports_container=False, device_count=1)
    policy = PackagingPolicy()
    findings = policy.inspect(
        comparison, comparison.option_of(PackagingKind.CONTAINER)
    )

    report.block("소견", "\n".join(f"  {f.describe()}" for f in findings))
    assert any(f.code == "PKG_RUNTIME_UNSUPPORTED" and f.is_blocking for f in findings)
    report.note(
        "**MCU/RTOS 에는 컨테이너 런타임이 없다.** "
        "여기서는 파일 배포만 가능하고, 그래서 전처리 일치를 사람이 지켜야 한다 — "
        "그 부담이 곧 실습 5-12 의 계약 대조다."
    )


def test_모델만_보내면_전처리는_안_간다() -> None:
    comparison = _comparison()
    findings = PackagingPolicy().inspect(
        comparison, comparison.option_of(PackagingKind.MODEL_FILE)
    )

    assert "PKG_PREPROCESSING_NOT_SHIPPED" in [f.code for f in findings]
    report.note(
        "모델만 바뀌고 전처리가 그대로면 **아무 에러 없이 틀린다.** "
        "그래서 파일 배포를 고르면 계약 대조를 배포 절차에 넣어야 한다 — "
        "선택의 대가는 나중에 절차로 돌아온다."
    )


def test_옛_것을_지우고_받으면_되돌릴_곳이_없다() -> None:
    comparison = _comparison(free_storage_bytes=200 * 1024 * 1024)
    findings = PackagingPolicy(storage_headroom=2.0).inspect(
        comparison, comparison.option_of(PackagingKind.CONTAINER)
    )

    report.block("소견", "\n".join(f"  {f.describe()}" for f in findings))
    assert any(f.code == "PKG_NO_STORAGE" and f.is_blocking for f in findings)
    report.note(
        "180 MiB 짜리를 200 MiB 여유에 받으면 **옛 것을 지워야 들어간다.** "
        "그러면 실습 6-9 의 롤백이 성립하지 않는다 — "
        "저장 공간은 배포 방식이 아니라 **롤백 가능 여부**를 정한다."
    )


def test_되돌리는_시간이_곧_피해_크기다() -> None:
    comparison = _comparison()
    slow = PackagingPolicy(max_rollback_seconds=120.0).inspect(
        comparison, comparison.option_of(PackagingKind.CONTAINER)
    )
    fast = PackagingPolicy(max_rollback_seconds=120.0).inspect(
        comparison, comparison.option_of(PackagingKind.MODEL_FILE)
    )

    report.block(
        "롤백 시간",
        f"  모델 파일  {OPTIONS[0].rollback_seconds:>5.0f}s → 소견 "
        f"{[f.code for f in fast]}\n"
        f"  컨테이너   {OPTIONS[2].rollback_seconds:>5.0f}s → 소견 "
        f"{[f.code for f in slow]}",
    )

    assert any(f.code == "PKG_ROLLBACK_SLOW" for f in slow)
    assert not any(f.code == "PKG_ROLLBACK_SLOW" for f in fast)
    report.note(
        "**사고가 난 뒤의 이 시간이 곧 피해 크기다.** "
        "빠른 롤백은 나중에 붙이는 기능이 아니라 "
        "배포 방식을 고를 때 함께 정해지는 성질이다."
    )


def test_회선이_좁으면_컨테이너는_현실적이지_않다() -> None:
    comparison = _comparison(uplink_bytes_per_second=32 * 1024)
    findings = PackagingPolicy(max_transfer_seconds=600.0).inspect(
        comparison, comparison.option_of(PackagingKind.CONTAINER)
    )

    container = comparison.option_of(PackagingKind.CONTAINER)
    report.block(
        "전송 시간",
        f"  한 대에 {container.transfer_seconds(comparison.device) / 60:.0f}분\n"
        f"  플릿 총량 {container.fleet_bytes(comparison.device) / 1024**3:.0f} GiB",
    )
    assert any(f.code == "PKG_TRANSFER_TOO_LONG" for f in findings)
    report.note(
        "**업데이트 중에도 라인은 돌고 있다.** "
        "한 대에 한 시간 반이면 3,000대를 다 돌리는 데 몇 주가 걸린다 — "
        "그 사이 절반은 옛 모델, 절반은 새 모델이다 (실습 6-8)."
    )


def test_묶음은_그_중간이다() -> None:
    comparison = _comparison()
    findings = PackagingPolicy().inspect(
        comparison, comparison.option_of(PackagingKind.BUNDLE)
    )

    report.block("묶음(BUNDLE)", "\n".join(f"  {f.describe()}" for f in findings) or "  막는 소견 없음")
    assert not any(f.is_blocking for f in findings)
    assert PackagingKind.BUNDLE.carries_preprocessing
    assert not PackagingKind.BUNDLE.carries_runtime
    report.note(
        "**모델 + 전처리 명세 + 라벨을 같이 보내고, 런타임은 디바이스 것을 쓴다.** "
        "48 KiB 로 계약 불일치를 막는다 — 현장에서 가장 자주 쓰는 절충이다."
    )


def test_비교하려면_최소_두_가지가_필요하다() -> None:
    with pytest.raises(InvariantViolation, match="최소 두 가지"):
        PackagingComparison(
            device=DeviceCapability(
                supports_container=True,
                free_storage_bytes=1024**3,
                uplink_bytes_per_second=1024,
            ),
            options=(OPTIONS[0],),
        )
