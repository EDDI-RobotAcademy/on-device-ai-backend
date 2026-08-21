"""실습 2-7 — 중복 데이터는 AI에게 같은 현실을 반복해서 가르친다.

    pytest -m lesson_2_7 -s

**행 전체가 같아야 중복이 아니다. 모델이 보는 것이 같으면 중복이다.**
타임스탬프는 모델의 입력이 아니다(실습 1-7).
그래서 모듈 1의 시간축 검사는 이 중복을 절대 잡지 못한다.
"""

from __future__ import annotations

import pytest

from application.data_quality.measure_uniqueness import MeasureUniquenessCommand
from domain.data_quality.uniqueness import DuplicateMeasurement, UniquenessPolicy
from tests.support import report
from tests.support import quality_scenario as qs

pytestmark = pytest.mark.lesson_2_7


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def measure(container, quality_container, name, path):  # noqa: ANN001
    qs.prepare_dataset(container, name, path)
    qs.start(quality_container, f"qa-{name}", name)
    return quality_container.measure_uniqueness().execute(
        MeasureUniquenessCommand(assessment_id=f"qa-{name}")
    )


def test_모듈1이_통과시킨_중복을_여기서_잡는다(
    container, quality_container, quality
) -> None:
    report.section("실습 2-7 · 중복 데이터는 AI에게 같은 현실을 반복해서 가르친다")

    view = measure(container, quality_container, "dirty", quality.dirty)
    report.block("중복 검사", view.render())

    found = codes(view)
    assert "UNIQUENESS_EXACT_DUPLICATE" in found
    assert view.verdict == "FAILED"

    duplicate = next(
        f for f in view.findings if f.code == "UNIQUENESS_EXACT_DUPLICATE"
    )
    report.note(
        f"입력이 완전히 같은 행 {duplicate.measured:.2%}. "
        "타임스탬프는 전부 다르다 — 그래서 모듈 1의 시간축 검사는 조용했다."
    )


def test_시간축_검사는_이_중복을_보지_못한다(container, quality) -> None:
    """같은 파일에 모듈 1의 시간축 검사를 돌려 확인한다."""
    from infrastructure.analysis.pandas_time_axis_measurer import (
        PandasTimeAxisMeasurer,
    )
    from tests.support.scenario import power_source

    measurement = PandasTimeAxisMeasurer().measure(
        power_source(quality.dirty), "timestamp"
    )
    assert measurement.duplicate_timestamp_count == 0
    assert measurement.out_of_order_count == 0

    report.block(
        "같은 파일, 두 관점",
        "  [모듈 1] 시간축 중복 :   0건  ← 타임스탬프는 전부 유일하다\n"
        "  [모듈 2] 입력 중복   : 276건  ← 모델이 보는 값은 같다",
    )
    report.note("무엇을 중복이라 부를지는 '모델이 무엇을 보는가'가 정한다.")


def test_중복은_표본_수를_부풀린다() -> None:
    measurement = DuplicateMeasurement(
        total_rows=10_000, exact_duplicate_count=2_500, duplicate_group_count=1_200
    )
    result = UniquenessPolicy().evaluate(measurement)

    report.block(
        "8,640행처럼 보이는 데이터",
        f"  전체 행       : {measurement.total_rows:,}\n"
        f"  서로 다른 표본 : {measurement.distinct_row_count:,}\n"
        f"  부풀림        : {measurement.inflation_ratio:.2f}배",
    )
    assert measurement.distinct_row_count == 7_500
    assert measurement.inflation_ratio == pytest.approx(10_000 / 7_500)
    assert "UNIQUENESS_SAMPLE_INFLATION" in codes(result)
    report.note("데이터가 25% 더 있다고 보고했다면, 그 보고가 틀렸다.")


def test_중복이_라벨_오류를_드러낸다(container, quality_container, quality) -> None:
    """같은 입력에 다른 라벨 — 실습 2-4 와 같은 문제를 다른 각도에서 만난다."""
    view = measure(container, quality_container, "dirty", quality.dirty)

    assert "UNIQUENESS_LABEL_CONFLICT" in codes(view)
    conflict = next(
        f for f in view.findings if f.code == "UNIQUENESS_LABEL_CONFLICT"
    )
    assert conflict.severity == "CRITICAL"
    report.note(
        "중복을 찾다가 라벨 모순이 나왔다. "
        "품질의 여섯 축은 독립적이지 않다 — 같은 사고가 여러 축에 그림자를 남긴다."
    )


def test_분할_누수와의_연결(container, quality) -> None:
    """중복이 왜 CRITICAL 인가 — 실습 1-8 과 이어지는 지점."""
    import pandas as pd

    frame = pd.read_csv(quality.dirty)
    features = [
        "active_power_kw",
        "reactive_power_kvar",
        "current_a",
        "voltage_v",
        "temperature_c",
        "spindle_rpm",
    ]
    duplicated = frame[frame.duplicated(subset=features, keep=False)]
    positions = duplicated.index.to_numpy()
    split_point = int(len(frame) * 0.85)

    crossing = sum(1 for p in positions if p >= split_point)
    report.note(
        f"중복 묶음에 속한 {len(positions)}행 중 {crossing}행이 test 구간에 있다. "
        "원본은 train 에, 사본은 test 에 — 시험 문제를 미리 본 것이다."
    )
    assert crossing > 0


def test_기준선_데이터에는_중복이_없다(container, quality_container, quality) -> None:
    view = measure(container, quality_container, "clean", quality.clean)
    assert view.verdict == "PASSED"
    assert view.score == 100.0
