"""실습 2-5 — 불균형 데이터는 AI의 판단을 왜곡한다.

    pytest -m lesson_2_5 -s

불량률 1.7% 인 라인에서 "정확도 98%" 는 아무 의미가 없다.
전부 정상이라고 찍어도 나오는 숫자다.
"""

from __future__ import annotations

import pytest

from application.data_quality.measure_balance import MeasureBalanceCommand
from domain.data_quality.balance import BalancePolicy, ClassBalanceMeasurement
from tests.support import report
from tests.support import quality_scenario as qs

pytestmark = pytest.mark.lesson_2_5


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def measure(container, quality_container, name, path, policy=None):  # noqa: ANN001
    qs.prepare_dataset(container, name, path)
    qs.start(quality_container, f"qa-{name}", name)
    command = MeasureBalanceCommand(assessment_id=f"qa-{name}")
    if policy is not None:
        command = MeasureBalanceCommand(assessment_id=f"qa-{name}", policy=policy)
    return quality_container.measure_balance().execute(command)


def test_현장_데이터는_원래_기울어져_있다(
    container, quality_container, quality
) -> None:
    report.section("실습 2-5 · 불균형 데이터는 AI의 판단을 왜곡한다")

    view = measure(container, quality_container, "clean", quality.clean)
    report.block("클래스 균형 검사", view.render())

    assert "BALANCE_IMBALANCED" in codes(view)
    assert view.verdict == "PASSED_WITH_WARNINGS"
    report.note(
        "불균형 자체는 데이터의 결함이 아니다. 현장이 원래 그렇다. "
        "그래서 CRITICAL 이 아니라 WARNING 이다."
    )


def test_정확도라는_지표가_먼저_무너진다() -> None:
    """아무것도 배우지 않은 모델이 몇 점을 받는지부터 계산한다."""
    measurement = ClassBalanceMeasurement(
        class_counts={"NORMAL": 9_970, "FAULT": 30}, test_split_ratio=0.15
    )
    result = BalancePolicy(max_baseline_accuracy=0.95).evaluate(measurement)

    report.block(
        "불량률 0.3% 라인",
        f"  baseline 정확도  : {measurement.baseline_accuracy:.2%}"
        "   ← 전부 NORMAL 이라고 찍었을 때\n"
        f"  불균형 비율      : {measurement.imbalance_ratio:.0f}배\n"
        f"  소수 클래스 표본 : {measurement.minority_count}개\n"
        f"  평가 집합 기대치 : {measurement.expected_minority_in_test:.1f}개",
    )

    assert measurement.baseline_accuracy == pytest.approx(0.997)
    assert "BALANCE_BASELINE_TOO_HIGH" in codes(result)
    report.note(
        "'정확도 99.7%' 짜리 모델을 만들었다는 보고서는 "
        "'아무것도 만들지 않았다'와 구별되지 않는다."
    )
    report.note("여기서 필요한 것은 재현율과 PR-AUC 다. 정확도가 아니다.")


def test_비율이_아니라_개수가_문제다() -> None:
    """재표집으로 해결되는 것과 해결되지 않는 것을 가른다."""
    적은데_비율만_나쁨 = ClassBalanceMeasurement(
        class_counts={"NORMAL": 100_000, "FAULT": 2_000}
    )
    개수_자체가_부족 = ClassBalanceMeasurement(
        class_counts={"NORMAL": 1_000, "FAULT": 20}
    )
    policy = BalancePolicy(min_minority_count=100)

    첫째 = policy.evaluate(적은데_비율만_나쁨)
    둘째 = policy.evaluate(개수_자체가_부족)

    report.block(
        "같은 50배 불균형, 다른 처방",
        f"  FAULT 2,000개 : {첫째.score.value:5.1f}점  {첫째.verdict.value}\n"
        f"  FAULT    20개 : {둘째.score.value:5.1f}점  {둘째.verdict.value}",
    )
    assert "BALANCE_MINORITY_TOO_FEW" not in codes(첫째)
    assert "BALANCE_MINORITY_TOO_FEW" in codes(둘째)
    report.note(
        "2,000개는 가중치·재표집으로 다룰 수 있다. "
        "20개는 데이터를 더 모으는 것 외에 방법이 없다."
    )


def test_분할하고_나면_평가할_표본이_남지_않는다() -> None:
    measurement = ClassBalanceMeasurement(
        class_counts={"NORMAL": 9_900, "FAULT": 100}, test_split_ratio=0.15
    )
    result = BalancePolicy(min_expected_minority_in_test=20.0).evaluate(measurement)

    assert measurement.expected_minority_in_test == pytest.approx(15.0)
    assert "BALANCE_TEST_TOO_THIN" in codes(result)
    report.note(
        "평가 집합에 FAULT 가 15개 남는다. "
        "그중 3개를 놓치면 재현율이 80%, 4개면 73% 다. 이 숫자로는 아무 결정도 못 한다."
    )


def test_클래스가_하나뿐이면_분류_문제가_아니다() -> None:
    result = BalancePolicy().evaluate(
        ClassBalanceMeasurement(class_counts={"NORMAL": 10_000, "FAULT": 0})
    )
    assert result.score.value == 0.0
    assert "BALANCE_SINGLE_CLASS" in codes(result)
