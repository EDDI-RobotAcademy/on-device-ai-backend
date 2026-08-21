"""실습 1-5 — 시간축을 놓치면 현장을 잃는다.

    pytest -m lesson_1_5 -s

pandas 는 정렬해 버리고 조용히 넘어간다. 그 순간 사고 흔적이 사라진다.
"""

from __future__ import annotations

import pytest

from application.data.inspect_time_axis import InspectTimeAxisCommand
from domain.data.time_axis import SamplingInterval, TimeAxisPolicy
from tests.support import report
from tests.support.scenario import (
    SAMPLE_INTERVAL_SECONDS,
    declare_schema,
    profile,
    register,
)

pytestmark = pytest.mark.lesson_1_5


def policy(**overrides: object) -> TimeAxisPolicy:
    base: dict[str, object] = dict(
        expected_interval=SamplingInterval(SAMPLE_INTERVAL_SECONDS)
    )
    base.update(overrides)
    return TimeAxisPolicy(**base)  # type: ignore[arg-type]


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def prepare(container, path, dataset_id) -> None:  # noqa: ANN001
    register(container, dataset_id, path)
    profile(container, dataset_id)
    declare_schema(container, dataset_id)


def test_시간축이_무너진_지점을_전부_찾는다(container, power) -> None:
    report.section("실습 1-5 · 시간축을 놓치면 현장을 잃는다")

    prepare(container, power.raw, "raw")
    view = container.inspect_time_axis().execute(
        InspectTimeAxisCommand(dataset_id="raw", policy=policy())
    )
    report.block("시간축 검증", view.render())

    found = codes(view)
    assert "TIME_OUT_OF_ORDER" in found  # 수집 큐가 순서를 보장하지 못했다
    assert "TIME_DUPLICATED" in found    # 같은 순간에 값이 두 개
    assert "TIME_GAP" in found           # 끊긴 구간
    assert view.verdict == "FAILED"

    gap = next(f for f in view.findings if f.code == "TIME_GAP")
    assert gap.measured is not None and gap.measured > 600
    report.note(
        f"가장 긴 공백 {gap.measured:.0f}초. "
        "그 시간에 현장에서 무슨 일이 있었는지 데이터에는 없다."
    )


def test_정렬로_덮으면_사라지는_사실(container, power) -> None:
    """측정기는 파일에 적힌 순서 그대로 센다. 정렬은 문제를 지우는 행위다."""
    import pandas as pd

    from infrastructure.analysis.pandas_time_axis_measurer import (
        PandasTimeAxisMeasurer,
    )
    from tests.support.scenario import power_source

    measurement = PandasTimeAxisMeasurer().measure(
        power_source(power.raw), "timestamp"
    )
    # 30쌍을 뒤집어 심었지만 그중 일부는 중복 시각과 겹쳐 역전이 되지 않는다.
    # 결함은 서로 겹쳐서 나타난다 — 현장 데이터가 원래 그렇다.
    assert 25 <= measurement.out_of_order_count <= 30

    frame = pd.read_csv(power.raw).sort_values("timestamp")
    sorted_path = power.raw.parent / "sorted.csv"
    frame.to_csv(sorted_path, index=False)

    after_sort = PandasTimeAxisMeasurer().measure(
        power_source(sorted_path), "timestamp"
    )
    assert after_sort.out_of_order_count == 0
    # 중복은 정렬해도 남는다 — 정렬이 지우는 것은 '순서가 틀렸다는 사실'뿐이다.
    assert after_sort.duplicate_timestamp_count == measurement.duplicate_timestamp_count

    report.block(
        "정렬 전후",
        f"  역순 행 수 : {measurement.out_of_order_count} → {after_sort.out_of_order_count}\n"
        f"  중복 시각  : {measurement.duplicate_timestamp_count} → "
        f"{after_sort.duplicate_timestamp_count}",
    )
    report.note("정렬은 문제를 해결하지 않는다. 문제를 보이지 않게 할 뿐이다.")


def test_수집_주기를_잘못_알고_있으면_경고한다(container, power) -> None:
    prepare(container, power.raw, "raw")
    view = container.inspect_time_axis().execute(
        InspectTimeAxisCommand(
            dataset_id="raw", policy=policy(expected_interval=SamplingInterval(1.0))
        )
    )
    assert "TIME_INTERVAL_MISMATCH" in codes(view)
    report.note("1초 주기로 알고 있었는데 실제로는 10초였다 — 이런 착오는 흔하다.")


def test_정리본의_시간축은_깨끗하다(container, power) -> None:
    prepare(container, power.curated, "curated")
    view = container.inspect_time_axis().execute(
        InspectTimeAxisCommand(dataset_id="curated", policy=policy())
    )
    report.block("정리본 시간축", view.render())
    assert view.verdict == "PASSED"


def test_시간축이_없는_스키마에는_이_검사를_요구할_수_없다(container, power) -> None:
    from application.shared.errors import UnsupportedOperation
    from domain.data.profile import FieldType
    from domain.data.schema import DataSchema, FieldRole, FieldSpec

    register(container, "raw", power.raw)
    profile(container, "raw")
    declare_schema(
        container,
        "raw",
        DataSchema(
            fields=(
                FieldSpec("active_power_kw", FieldType.REAL, FieldRole.FEATURE),
                FieldSpec("condition", FieldType.CATEGORY, FieldRole.LABEL),
            )
        ),
    )
    with pytest.raises(UnsupportedOperation, match="TIME_INDEX"):
        container.inspect_time_axis().execute(
            InspectTimeAxisCommand(dataset_id="raw", policy=policy())
        )
