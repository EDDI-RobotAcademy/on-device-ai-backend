"""실습 1-4 — 센서와 이미지는 거짓말을 어떻게 하는가?

    pytest -m lesson_1_4 -s

센서는 값을 비우면서 거짓말하지 않는다. 그럴듯한 값을 계속 뱉으면서 거짓말한다.
이미지는 또 다른 방식으로 거짓말한다.
"""

from __future__ import annotations

import pytest

from application.data.inspect_signal_plausibility import (
    InspectSignalPlausibilityCommand,
)
from domain.data.signal import SignalPlausibilityPolicy
from tests.support import report
from tests.support.scenario import (
    declare_schema,
    profile,
    register,
    register_images,
)

pytestmark = pytest.mark.lesson_1_4


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def test_센서는_그럴듯한_값을_뱉으며_거짓말한다(container, power) -> None:
    report.section("실습 1-4 · 센서와 이미지는 거짓말을 어떻게 하는가?")

    register(container, "raw", power.raw)
    profile(container, "raw")
    declare_schema(container, "raw")

    view = container.inspect_signal_plausibility().execute(
        InspectSignalPlausibilityCommand(dataset_id="raw")
    )
    report.block("센서 신호 검증", view.render())

    found = codes(view)
    assert "SIGNAL_STUCK" in found        # 전압이 100분 동안 변하지 않았다
    assert "SIGNAL_OUT_OF_RANGE" in found  # 전류가 음수다
    assert "SIGNAL_SATURATED" in found     # 유효전력이 계측 한계에서 잘렸다
    assert view.verdict == "FAILED"

    stuck = next(f for f in view.findings if f.code == "SIGNAL_STUCK")
    assert stuck.subject == "voltage_v"
    report.note(
        "결측(NaN)은 눈에 보여서 오히려 덜 위험하다. "
        "여기서 잡은 것은 전부 '정상적으로 보이는 값'이다."
    )


def test_포화는_경고고_고착과_범위이탈은_차단이다(container, power) -> None:
    """세 가지 거짓말의 무게가 다르다."""
    register(container, "raw", power.raw)
    profile(container, "raw")
    declare_schema(container, "raw")
    view = container.inspect_signal_plausibility().execute(
        InspectSignalPlausibilityCommand(dataset_id="raw")
    )

    severity = {f.code: f.severity for f in view.findings}
    assert severity["SIGNAL_STUCK"] == "CRITICAL"
    assert severity["SIGNAL_OUT_OF_RANGE"] == "CRITICAL"
    assert severity["SIGNAL_SATURATED"] == "WARNING"
    report.note(
        "포화는 값이 남아 있되 잘린 것이다 — 알고 쓰면 된다. "
        "고착은 그 구간 전체가 거짓이다 — 쓰면 안 된다."
    )


def test_정리본_센서는_통과한다(container, power) -> None:
    register(container, "curated", power.curated)
    profile(container, "curated")
    declare_schema(container, "curated")
    view = container.inspect_signal_plausibility().execute(
        InspectSignalPlausibilityCommand(dataset_id="curated")
    )
    assert view.verdict == "PASSED"


def test_이미지는_다른_방식으로_거짓말한다(container, castings) -> None:
    register_images(container, "castings", castings.root)
    # 이미지 데이터셋에도 스키마는 필요하다. 파일 경로와 판정이 계약이다.
    from domain.data.profile import FieldType
    from domain.data.schema import DataSchema, FieldRole, FieldSpec

    from application.data.declare_data_schema import DeclareDataSchemaCommand
    from application.data.profile_dataset import ProfileDatasetCommand

    # 이미지 디렉터리는 표로 읽을 수 없으므로 프로파일링 경로가 다르다.
    from infrastructure.errors import UnsupportedSourceFormat

    with pytest.raises(UnsupportedSourceFormat):
        container.profile_dataset().execute(ProfileDatasetCommand(dataset_id="castings"))

    # 이미지 스키마는 파일에서 추론하는 것이 아니라 수집 규약에서 온다.
    from application.data.support import load_dataset

    dataset = load_dataset(container.repository, "castings")
    dataset.attach_profile(_image_profile())
    container.repository.save(dataset)

    container.declare_data_schema().execute(
        DeclareDataSchemaCommand(
            dataset_id="castings",
            schema=DataSchema(
                fields=(
                    FieldSpec("image_path", FieldType.IMAGE_REF, FieldRole.FEATURE),
                    FieldSpec("verdict", FieldType.CATEGORY, FieldRole.LABEL),
                )
            ),
        )
    )

    view = container.inspect_signal_plausibility().execute(
        InspectSignalPlausibilityCommand(
            dataset_id="castings",
            policy=SignalPlausibilityPolicy(min_focus_score=50.0),
        )
    )
    report.block("이미지 신호 검증", view.render())

    found = codes(view)
    assert "SIGNAL_UNREADABLE" in found          # 열리지 않는 파일
    assert "SIGNAL_DEFOCUSED" in found           # 초점이 나감
    assert "SIGNAL_EXPOSURE_SHIFT" in found      # 조명이 도중에 바뀜
    assert "SIGNAL_VISUAL_DUPLICATE" in found    # 같은 사진이 여러 장
    assert "SIGNAL_RESOLUTION_MIXED" in found    # 카메라가 섞임
    assert view.verdict == "FAILED"

    report.note(
        "중복 이미지는 분할 때 학습/평가에 나뉘어 들어가면 성능을 부풀린다. "
        "실습 1-8 에서 다시 만난다."
    )


def _image_profile():  # noqa: ANN202
    """이미지 데이터셋의 프로파일은 수집 규약에서 온다."""
    from domain.data.profile import ColumnProfile, DatasetProfile, FieldType

    return DatasetProfile(
        row_count=51,
        columns=(
            ColumnProfile("image_path", FieldType.IMAGE_REF, 51, 0, 51),
            ColumnProfile("verdict", FieldType.CATEGORY, 51, 0, 2),
        ),
    )
