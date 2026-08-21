"""실습 1-9 — Training Data와 Reality는 왜 다른가?

    pytest -m lesson_1_9 -s

학습 데이터는 언제나 과거이고, 대체로 잘 정리된 과거다.
여기서 재는 것은 "데이터가 깨졌는가"가 아니라 "현실을 대표하는가"다.
"""

from __future__ import annotations

import pytest

from application.data.analyze_representativeness import (
    AnalyzeRepresentativenessCommand,
)
from tests.support import report
from tests.support.scenario import (
    declare_schema,
    power_source,
    profile,
    register,
)

pytestmark = pytest.mark.lesson_1_9


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.inspection.findings}


def prepare(container, power) -> None:  # noqa: ANN001
    register(container, "curated", power.curated, collected_from="LINE-3 / 3월")
    profile(container, "curated")
    declare_schema(container, "curated")


def test_여름_현장은_3월_학습_데이터가_본_적_없는_곳이다(container, power) -> None:
    report.section("실습 1-9 · Training Data와 Reality는 왜 다른가?")

    prepare(container, power)
    view = container.analyze_representativeness().execute(
        AnalyzeRepresentativenessCommand(
            dataset_id="curated",
            observed=power_source(power.recent_shifted, "LINE-3 / 7월 현장 표본"),
        )
    )
    report.block("3월 학습 데이터 vs 7월 현장", view.render())

    found = codes(view)
    assert "REPR_DISTRIBUTION_SHIFTED" in found  # 분포가 통째로 이동
    assert "REPR_COVERAGE_GAP" in found          # 학습 범위 밖으로 나감
    assert "REPR_UNSEEN_CATEGORY" in found       # 학습에 없던 제품 코드
    assert view.inspection.verdict == "FAILED"

    assert view.worst_field == "temperature_c"
    report.note(
        f"가장 크게 이동한 축: {view.worst_field} (PSI {view.worst_psi:.2f}). "
        "여름이 되었을 뿐인데 모델이 본 적 없는 세상이 되었다."
    )

    coverage = dict((name, cov) for name, _, cov in view.field_psi)
    assert coverage["temperature_c"] < 0.5
    report.note(
        f"온도 커버리지 {coverage['temperature_c']:.1%} — "
        "현장 온도의 대부분이 학습 데이터 범위 밖이다. 모델은 외삽으로 찍는다."
    )


def test_같은_계절_현장은_통과한다(container, power) -> None:
    """비교 도구가 아무거나 잡아내는 것이 아님을 확인한다."""
    prepare(container, power)
    view = container.analyze_representativeness().execute(
        AnalyzeRepresentativenessCommand(
            dataset_id="curated",
            observed=power_source(power.recent_stable, "LINE-3 / 3월 2주차"),
        )
    )
    report.block("3월 학습 데이터 vs 3월 2주차 현장", view.render())

    assert view.inspection.verdict == "PASSED"
    assert view.worst_psi < 0.1
    report.note("PSI 관행: 0.1 미만 안정 / 0.1~0.25 이동 중 / 0.25 이상 심각.")


def test_LOT_번호가_새로_생긴_것은_사건이_아니다(power) -> None:
    """매번 새로 생기는 식별자를 '처음 보는 범주'로 세면 경보가 무의미해진다."""
    from infrastructure.analysis.numpy_distribution_comparer import (
        NumpyDistributionComparer,
    )
    from tests.support.scenario import power_schema

    measurement = NumpyDistributionComparer().compare(
        power_source(power.curated), power_source(power.recent_stable), power_schema()
    )
    # batch_id 는 날짜가 바뀌면 전부 새 값이다. 그래도 0 이어야 한다.
    assert measurement.unseen_category_ratio == 0.0

    shifted = NumpyDistributionComparer().compare(
        power_source(power.curated), power_source(power.recent_shifted), power_schema()
    )
    # product_code 에 학습에 없던 제품이 들어왔다. 이건 사건이다.
    assert shifted.unseen_category_ratio > 0.3

    report.block(
        "처음 보는 범주 비율",
        f"  3월 2주차 : {measurement.unseen_category_ratio:.1%}  (LOT 번호만 새로 생김)\n"
        f"  7월       : {shifted.unseen_category_ratio:.1%}  (신규 제품 코드 투입)",
    )


def test_표본이_적으면_이상없음을_믿지_않는다(container, power) -> None:
    import pandas as pd

    from domain.data.representativeness import RepresentativenessPolicy

    prepare(container, power)
    small = pd.read_csv(power.recent_stable).head(20)
    path = power.recent_stable.parent / "tiny.csv"
    small.to_csv(path, index=False)

    view = container.analyze_representativeness().execute(
        AnalyzeRepresentativenessCommand(
            dataset_id="curated",
            observed=power_source(path, "표본 20건"),
            policy=RepresentativenessPolicy(min_observed_sample_count=100),
        )
    )
    assert "REPR_SAMPLE_TOO_SMALL" in codes(view)
    report.note("20건으로 '드리프트 없음'을 주장할 수는 없다.")
