"""실습 2-2 — 결측치를 숨기면 AI가 대신 대가를 치른다.

    pytest -m lesson_2_2 -s

결측률이 같아도 **패턴이 다르면 대응이 다르다.**
그리고 가장 위험한 결측은 결측률 0%로 보이는 결측이다.
"""

from __future__ import annotations

import pytest

from application.data_quality.measure_completeness import MeasureCompletenessCommand
from domain.data_quality.completeness import (
    CompletenessPolicy,
    FieldMissingness,
    MissingValueMeasurement,
)
from tests.support import report
from tests.support import quality_scenario as qs

pytestmark = pytest.mark.lesson_2_2


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def measure(container, quality_container, quality, name: str, path):  # noqa: ANN001
    qs.prepare_dataset(container, name, path)
    qs.start(quality_container, f"qa-{name}", name)
    return quality_container.measure_completeness().execute(
        MeasureCompletenessCommand(assessment_id=f"qa-{name}")
    )


def test_결측을_0으로_채우면_결측률은_0이_되고_문제는_남는다(
    container, quality_container, quality
) -> None:
    report.section("실습 2-2 · 결측치를 숨기면 AI가 대신 대가를 치른다")

    view = measure(container, quality_container, quality, "dirty", quality.dirty)
    report.block("결측 검사", view.render())

    found = codes(view)
    assert "MISSING_HIDDEN" in found
    assert "MISSING_CONCENTRATED" in found
    assert view.verdict == "FAILED"

    hidden = next(f for f in view.findings if f.code == "MISSING_HIDDEN")
    assert hidden.subject == "temperature_c"
    report.note(
        "26℃ 짜리 공장에 0℃ 가 425개 있다. "
        "그 0℃ 는 물리 범위(-20~120) 안이라 모듈 1의 범위 검사를 그대로 통과했다."
    )


def test_같은_결측률도_패턴이_다르면_다르게_판정한다() -> None:
    """파일도 pandas 도 필요 없다. 측정값을 손으로 만들어 Policy 만 시험한다."""
    policy = CompletenessPolicy(max_missing_ratio=0.02)

    흩어짐 = policy.evaluate(
        MissingValueMeasurement(
            fields=(
                FieldMissingness(
                    field_name="temperature_c",
                    total_count=10_000,
                    missing_count=500,  # 5%
                    longest_missing_run=2,
                    concentration_ratio=0.11,
                ),
            )
        )
    )
    뭉침 = policy.evaluate(
        MissingValueMeasurement(
            fields=(
                FieldMissingness(
                    field_name="temperature_c",
                    total_count=10_000,
                    missing_count=500,  # 같은 5%
                    longest_missing_run=500,
                    concentration_ratio=0.95,
                ),
            )
        )
    )

    report.block(
        "결측률은 같고 패턴만 다른 두 경우",
        f"  흩어짐 : {흩어짐.score.value:5.1f}점  {흩어짐.verdict.value}\n"
        f"  뭉침   : {뭉침.score.value:5.1f}점  {뭉침.verdict.value}",
    )

    assert "MISSING_RUN_LONG" not in codes(흩어짐)
    assert "MISSING_RUN_LONG" in codes(뭉침)
    assert "MISSING_CONCENTRATED" in codes(뭉침)
    assert 뭉침.score.value < 흩어짐.score.value

    report.note("흩어진 5% 는 보간할 수 있다. 뭉친 5% 는 센서가 죽어 있던 구간이다.")
    report.note("같은 숫자를 보고 다른 결정을 하려면, 숫자를 하나만 봐서는 안 된다.")


def test_설비_정지_중의_0은_은폐_결측이_아니다(
    container, quality_container, quality
) -> None:
    """spindle_rpm=0 은 실제 물리 상태다. 이것까지 잡으면 검사는 못 쓰는 도구가 된다."""
    view = measure(container, quality_container, quality, "clean", quality.clean)

    hidden = [f for f in view.findings if f.code == "MISSING_HIDDEN"]
    assert hidden == []
    assert view.verdict == "PASSED"

    report.block("오염 없는 데이터의 결측 검사", view.render())
    report.note(
        "clean 파일에도 rpm=0 이 150개 있다. 설비가 실제로 멈춰 있었기 때문이다."
    )
    report.note(
        "가르는 기준은 **연속성**이다. "
        "채워 넣은 값은 흩어지고, 진짜 물리 상태는 뭉친다."
    )


def test_흩어짐과_뭉침의_경계는_Policy_가_정한다() -> None:
    scattered = FieldMissingness(
        field_name="temperature_c",
        total_count=1000,
        repeated_value=0.0,
        repeated_value_count=50,
        repeated_value_mean_run=1.05,
    )
    clustered = FieldMissingness(
        field_name="spindle_rpm",
        total_count=1000,
        repeated_value=0.0,
        repeated_value_count=50,
        repeated_value_mean_run=10.0,
    )
    assert scattered.hidden_missing_suspected(max_scattered_run=2.0) is True
    assert clustered.hidden_missing_suspected(max_scattered_run=2.0) is False


def test_결측이_특정_LOT_에_몰리면_평균으로_채울_수_없다(
    container, quality_container, quality
) -> None:
    view = measure(container, quality_container, quality, "dirty", quality.dirty)
    finding = next(f for f in view.findings if f.code == "MISSING_CONCENTRATED")

    assert finding.measured is not None and finding.measured > 0.6
    report.note(
        f"상위 10% LOT 이 전체 결측의 {finding.measured:.0%} 를 차지한다. "
        "그 LOT 의 평균은 그 LOT 이 아니라 다른 LOT 의 값으로 채워지게 된다."
    )
