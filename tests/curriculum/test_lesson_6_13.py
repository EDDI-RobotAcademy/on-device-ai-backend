"""실습 6-13 — S3에 버전과 권한을 걸어라.

    pytest -m lesson_6_13 -s

실습 6-2 는 "어디에 둘 것인가"를 정했다. 여기서는 **"어떻게 지킬 것인가"**를 정한다.

기본값 그대로 만든 버킷에는 이런 성질이 있다.

    같은 키에 다시 쓰면 **옛 것이 사라진다** — 되돌릴 곳이 없어진다 (실습 6-9)
    보관 규칙이 없으면 **원본이 영원히 쌓인다** — 1년에 7.5 TiB
    권한이 넓으면 **되돌릴 수 없는 사고가 난다** — s3:* 에는 DeleteBucket 이 있다

그리고 이 실습에서 가장 중요한 한 줄:

    **버저닝은 사고가 난 뒤에 켤 수 없다.** 켠 시점 이후의 객체만 지킨다.
"""

from __future__ import annotations

import pytest

from application.fleet.govern_storage import (
    GovernStorageCommand,
    InspectStorageCommand,
)
from domain.fleet.governance import AccessStatement
from domain.fleet.object_key import ObjectKey
from domain.shared.errors import InvariantViolation
from tests.support import report

pytestmark = pytest.mark.lesson_6_13


def _key(path: str) -> ObjectKey:
    prefix, _, filename = path.rpartition("/")
    return ObjectKey(prefix=prefix, partitions=(), filename=filename)


def test_기본값_그대로인_버킷은_아무것도_지키지_않는다(fleet_bare) -> None:
    report.section("실습 6-13 · S3에 버전과 권한을 걸어라")

    view = fleet_bare.inspect_storage().execute(InspectStorageCommand())
    report.block("만들기만 한 버킷", view.render())

    codes = {f.code for f in view.findings}
    assert "GOV_NO_VERSIONING" in codes
    assert "GOV_NO_ENCRYPTION" in codes
    assert "GOV_PUBLIC_ACCESS_OPEN" in codes
    assert view.verdict == "BLOCKED"
    report.note(
        "**켜야 켜진다.** 만들었다고 지켜지는 것은 하나도 없다. "
        "그리고 이 셋은 데이터가 들어오기 **전에** 켜야 한다."
    )


def test_걸고_나서_다시_읽어_확인한다(fleet_bare) -> None:
    view = fleet_bare.govern_storage().execute(
        GovernStorageCommand(expiration_days=365)
    )
    report.block("걸고 나서", view.render())

    assert view.versioning_enabled
    assert view.encryption_algorithm == "AES256"
    assert view.public_access_blocked
    assert view.lifecycle_expiration_days == 365
    report.note(
        "**'설정했다'와 '설정되어 있다'는 다른 이야기다.** "
        "그래서 건 다음에 다시 읽는다 — 누군가 콘솔에서 껐을 수도 있고, "
        "코드가 다른 버킷에 걸었을 수도 있다."
    )


def test_버저닝이_켜져_있으면_덮어써도_옛_것이_남는다(fleet_bare) -> None:
    """이 실습의 본론."""
    fleet_bare.govern_storage().execute(GovernStorageCommand())

    key = _key("models/line3/current.tflite")
    fleet_bare.store.put(key, b"v1-model-bytes")
    fleet_bare.store.put(key, b"v2-model-bytes")  # 같은 자리에 덮어썼다

    view = fleet_bare.inspect_storage().execute(
        InspectStorageCommand(version_prefix="models/")
    )
    report.block(
        "덮어쓴 흔적",
        f"  {key.render()} 의 남아 있는 버전 수: "
        f"{len(view.overwritten_keys)}개 키가 2회 이상",
    )

    assert view.overwritten_keys == (key.render(),)
    assert "GOV_SILENT_OVERWRITE" not in [f.code for f in view.findings]
    report.note(
        "덮어썼는데 **옛 것이 남아 있다.** "
        "6-9 의 롤백은 이 위에서만 성립한다 — "
        "버저닝이 꺼져 있었다면 v1 은 이미 없다."
    )


def test_버저닝이_꺼진_채_덮어쓰면_이미_늦었다(fleet_bare) -> None:
    key = _key("models/line3/no-versioning.tflite")
    fleet_bare.store.put(key, b"v1")
    fleet_bare.store.put(key, b"v2")

    view = fleet_bare.inspect_storage().execute(
        InspectStorageCommand(version_prefix="models/")
    )
    report.block("소견", "\n".join(f"  {f.describe()}" for f in view.findings))

    assert not view.versioning_enabled
    report.note(
        "지금 켜도 **v1 은 돌아오지 않는다.** "
        "버저닝은 켠 시점 이후의 객체만 지킨다 — "
        "그래서 이건 사고가 난 뒤에 대응할 수 있는 종류의 설정이 아니다."
    )


def test_누구에게나_허용하는_한_줄이_버킷을_연다(fleet_bare) -> None:
    view = fleet_bare.govern_storage().execute(
        GovernStorageCommand(
            statements=(
                AccessStatement(
                    sid="PublicRead",
                    effect="Allow",
                    principal="*",
                    actions=("s3:GetObject",),
                    resources=("arn:aws:s3:::ondevice-ai-lake/*",),
                ),
            )
        )
    )
    report.block("권한", view.render())

    assert "GOV_PUBLIC_STATEMENT" in [f.code for f in view.findings]
    report.note(
        "'모델 파일만 공개하려고' 넣은 한 줄이다. "
        "**현장 신호에는 생산량과 가동 패턴이 들어 있다** — "
        "경쟁사가 읽으면 원가 구조가 보인다."
    )


def test_s3_스타는_DeleteBucket_을_포함한다(fleet_bare) -> None:
    view = fleet_bare.govern_storage().execute(
        GovernStorageCommand(
            statements=(
                AccessStatement(
                    sid="TrainingJob",
                    effect="Allow",
                    principal="arn:aws:iam::123456789012:role/ondevice-ai-training",
                    actions=("s3:*",),
                    resources=("arn:aws:s3:::ondevice-ai-lake/*",),
                ),
            )
        )
    )

    assert "GOV_OVERBROAD_ACTION" in [f.code for f in view.findings]
    report.note(
        "학습 잡은 읽고 쓰기만 하면 된다. "
        "**s3:* 를 준 순간 그 역할이 버킷을 지울 수 있다** — "
        "그리고 그런 사고는 대개 자정에 도는 배치가 낸다."
    )


def test_보관_규칙이_없으면_청구서가_대신_정한다(fleet_bare) -> None:
    view = fleet_bare.govern_storage().execute(
        GovernStorageCommand(expiration_days=None)
    )

    assert "GOV_NO_LIFECYCLE" in [f.code for f in view.findings]
    report.note(
        "3,000대가 하루 7 MiB 씩 올리면 1년에 7.5 TiB 다 (실습 6-1). "
        "**지울 것을 정하지 않으면 아무도 안 지운다.**"
    )


def test_권한에는_행위와_대상이_있어야_한다() -> None:
    with pytest.raises(InvariantViolation, match="행위와 대상"):
        AccessStatement(
            sid="Empty", effect="Allow", principal="*", actions=(), resources=()
        )
    with pytest.raises(InvariantViolation, match="Allow 또는 Deny"):
        AccessStatement(
            sid="X", effect="Maybe", principal="*", actions=("s3:Get",), resources=("*",)
        )
