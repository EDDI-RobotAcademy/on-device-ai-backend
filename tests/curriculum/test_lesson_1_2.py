"""실습 1-2 — CSV 하나에도 현장의 문제가 숨어 있다.

    pytest -m lesson_1_2 -s

스키마를 선언하는 순간, 파일이 그 계약을 어떻게 배신하는지가 드러난다.
계약이 없으면 배신도 없다. 그래서 문제를 찾으려면 먼저 계약을 세워야 한다.
"""

from __future__ import annotations

import pytest

from domain.data.inspection import InspectionKind
from tests.support import report
from tests.support.scenario import declare_schema, profile, register

pytestmark = pytest.mark.lesson_1_2


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def test_스키마를_선언하자_파일이_배신하는_지점이_드러난다(container, power) -> None:
    report.section("실습 1-2 · CSV 하나에도 현장의 문제가 숨어 있다")

    register(container, "raw", power.raw)
    profile(container, "raw")
    view = declare_schema(container, "raw")

    report.block("스키마 대조 결과", view.render())

    assert view.kind == InspectionKind.SCHEMA.value
    assert view.verdict == "FAILED"

    found = codes(view)
    assert "BELOW_PHYSICAL_RANGE" in found   # 전류 음수
    assert "UNDECLARED_FIELD" in found       # 아무도 모르는 열

    report.note("설비 사양서가 없으면 'BELOW_PHYSICAL_RANGE' 는 영원히 발견되지 않는다.")


def test_물리_범위는_데이터가_아니라_현장이_준다(container, power) -> None:
    """value_range 를 빼면 같은 파일이 '문제 없음'으로 보인다."""
    from domain.data.schema import DataSchema

    from tests.support.scenario import power_schema

    register(container, "raw", power.raw)
    profile(container, "raw")

    # 물리 범위를 지운 스키마 — 문법적으로는 아무 문제 없다.
    without_range = DataSchema(
        fields=tuple(
            f if f.value_range is None else type(f)(
                name=f.name,
                type=f.type,
                role=f.role,
                unit=f.unit,
                required=f.required,
                value_range=None,
            )
            for f in power_schema().fields
        )
    )
    view = declare_schema(container, "raw", without_range)

    report.block("물리 범위를 선언하지 않은 경우", view.render())
    assert "BELOW_PHYSICAL_RANGE" not in codes(view)
    report.note(
        "전류 -3A 가 그대로 통과한다. 데이터는 아무 말도 하지 않는다. "
        "말해 주는 것은 설비 사양서다."
    )


def test_모르는_열은_경고이지_차단이_아니다(container, power) -> None:
    """operator_note 는 학습을 막을 이유가 아니다. 다만 의미를 모른 채 쓰면 안 된다."""
    register(container, "raw", power.raw)
    profile(container, "raw")
    view = declare_schema(container, "raw")

    undeclared = [f for f in view.findings if f.code == "UNDECLARED_FIELD"]
    assert [f.subject for f in undeclared] == ["operator_note"]
    assert all(f.severity == "WARNING" for f in undeclared)


def test_깨끗한_파일은_같은_스키마를_통과한다(container, power) -> None:
    """같은 계약, 다른 파일. 문제는 스키마가 아니라 데이터에 있었다."""
    register(container, "curated", power.curated)
    profile(container, "curated")
    view = declare_schema(container, "curated")

    report.block("정리본에 같은 스키마를 적용", view.render())
    assert view.verdict == "PASSED"
