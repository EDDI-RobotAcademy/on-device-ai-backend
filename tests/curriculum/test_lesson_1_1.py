"""실습 1-1 — 데이터를 열어보는 순간 현실이 보인다.

    pytest -m lesson_1_1 -s

현장에서 받은 CSV 를 처음 열어 본다.
아직 아무 판단도 하지 않는다. 무엇이 들어 있는지 사실만 확인한다.
그런데 그 사실만으로도 이미 문제가 보인다.
"""

from __future__ import annotations

import pytest

from tests.support import report
from tests.support.scenario import profile, register

pytestmark = pytest.mark.lesson_1_1


def test_현장_CSV_를_열면_보이는_것들(container, power) -> None:
    report.section("실습 1-1 · 데이터를 열어보는 순간 현실이 보인다")

    register(container, "plant-power-raw", power.raw)
    view = profile(container, "plant-power-raw")

    report.block("프로파일 (판단 없이 사실만)", view.render())

    # 1) 파일에 적힌 행 수와 우리가 기대한 행 수는 다르다.
    #    24시간 × 10초 주기라면 8,641개여야 한다. 실제로는 그보다 적다.
    assert view.row_count == 8460
    report.note(
        f"기대 8,641행(24h/10s) → 실제 {view.row_count:,}행. "
        f"{8641 - view.row_count}행이 어딘가에서 사라졌다."
    )

    # 2) 값이 하나뿐인 열 — 정보량이 0이다.
    constant = [c.name for c in view.columns if c.distinct_count <= 1]
    assert "meter_id" in constant
    report.note(f"값이 하나뿐인 열: {constant} — 학습에 넣어도 아무것도 배우지 못한다.")

    # 3) 비어 있는 값
    missing = {c.name: c.missing_ratio for c in view.columns if c.missing_count}
    assert missing["temperature_c"] == pytest.approx(0.03, abs=0.005)
    report.note(
        "결측: " + ", ".join(f"{k} {v:.2%}" for k, v in sorted(missing.items()))
    )

    # 4) 라벨 열에도 빈 값이 있다. 정답이 없는 행이 존재한다는 뜻이다.
    condition = next(c for c in view.columns if c.name == "condition")
    assert condition.missing_count > 0
    report.note(f"라벨(condition)이 비어 있는 행 {condition.missing_count}개")

    # 5) 물리적으로 불가능한 값이 이미 눈에 보인다.
    current = next(c for c in view.columns if c.name == "current_a")
    assert current.minimum is not None and current.minimum < 0
    report.note(
        f"전류 최소값 {current.minimum}A — 전류가 음수로 흐를 수는 없다."
    )


def test_등록만_하고_열어보지_않으면_아무것도_모른다(container, power) -> None:
    """데이터를 '가지고 있다'와 '알고 있다'는 다르다."""
    register(container, "plant-power-raw", power.raw)

    from application.data.get_dataset import GetDatasetQuery

    view = container.get_dataset().execute(
        GetDatasetQuery(dataset_id="plant-power-raw")
    )
    assert view.status == "REGISTERED"
    assert view.row_count is None  # 행이 몇 개인지조차 모른다

    profile(container, "plant-power-raw")
    view = container.get_dataset().execute(
        GetDatasetQuery(dataset_id="plant-power-raw")
    )
    assert view.status == "PROFILED"
    assert view.row_count == 8460


def test_프로파일링은_사실만_남기고_판단하지_않는다(container, power) -> None:
    """이 단계에서는 어떤 검사 결과도 만들어지지 않는다."""
    register(container, "plant-power-raw", power.raw)
    profile(container, "plant-power-raw")

    from application.data.get_dataset import GetDatasetQuery

    view = container.get_dataset().execute(
        GetDatasetQuery(dataset_id="plant-power-raw")
    )
    assert view.inspections == ()
    assert view.verdict is None
