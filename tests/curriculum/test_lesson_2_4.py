"""실습 2-4 — 잘못된 Label 하나가 모델 전체를 망친다.

    pytest -m lesson_2_4 -s

모듈 1(실습 1-6)에서는 "라벨의 **정의**가 있는가"를 물었다.
여기서는 "붙여 놓은 라벨이 실제 데이터와 **모순되지 않는가**"를 묻는다.
"""

from __future__ import annotations

import pytest

from application.data_quality.measure_label_quality import MeasureLabelQualityCommand
from domain.data_quality.label_quality import (
    LabelConsistencyRule,
    LabelErrorMeasurement,
    LabelQualityPolicy,
)
from domain.shared.errors import InvariantViolation
from tests.support import report
from tests.support import quality_scenario as qs

pytestmark = pytest.mark.lesson_2_4


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def measure(container, quality_container, name, path, rules=None):  # noqa: ANN001
    qs.prepare_dataset(container, name, path)
    qs.start(quality_container, f"qa-{name}", name)
    return quality_container.measure_label_quality().execute(
        MeasureLabelQualityCommand(
            assessment_id=f"qa-{name}",
            rules=qs.label_rules() if rules is None else rules,
        )
    )


def test_규칙이_없으면_라벨_오류를_찾을_수_없다(
    container, quality_container, quality
) -> None:
    report.section("실습 2-4 · 잘못된 Label 하나가 모델 전체를 망친다")

    view = measure(container, quality_container, "dirty", quality.dirty, rules=())
    report.block("규칙 없이 라벨을 검사하면", view.render())

    assert "LABEL_NO_CONSISTENCY_RULE" in codes(view)
    assert view.verdict == "FAILED"
    report.note(
        "'라벨 오류 0건'이 나온다. 찾을 방법이 없었으니 당연하다. "
        "그 0건은 근거가 아니다."
    )


def test_현장_규칙을_넣으면_모순이_드러난다(
    container, quality_container, quality
) -> None:
    view = measure(container, quality_container, "dirty", quality.dirty)
    report.block("현장 규칙 3개를 적용", view.render())

    found = codes(view)
    assert "LABEL_RULE_VIOLATION" in found
    assert "LABEL_CONFLICT" in found
    assert view.verdict == "FAILED"

    violation = next(f for f in view.findings if f.code == "LABEL_RULE_VIOLATION")
    report.note(
        f"규칙과 모순되는 라벨 {violation.measured:.2%}. "
        "FAULT 인데 부하가 정상이거나, NORMAL 인데 390kW 를 쓰고 있다."
    )
    report.note(
        "이 세 줄의 규칙은 데이터에서 나오지 않는다. 설비 담당자에게서 나온다."
    )


def test_규칙에는_근거가_있어야_한다() -> None:
    """숫자만 있고 이유가 없는 규칙은 나중에 아무도 못 고친다."""
    with pytest.raises(InvariantViolation, match="근거 설명이 없다"):
        LabelConsistencyRule(
            label="FAULT", field_name="active_power_kw", expected_max=30.0
        )

    with pytest.raises(InvariantViolation, match="조건이 없다"):
        LabelConsistencyRule(
            label="FAULT", field_name="active_power_kw", description="트립"
        )

    report.note("규칙도 데이터다. 출처와 근거가 없으면 신뢰할 수 없다.")


def test_같은_입력에_다른_라벨은_모델이_배울_수_없다(
    container, quality_container, quality
) -> None:
    view = measure(container, quality_container, "dirty", quality.dirty)
    conflict = next(f for f in view.findings if f.code == "LABEL_CONFLICT")

    assert conflict.severity == "CRITICAL"
    report.note(
        f"입력이 같은데 라벨이 다른 행 {conflict.measured:.2%}. "
        "모델 입장에서는 같은 문제에 두 개의 정답이 있는 것이다."
    )
    report.note("이 모순은 데이터를 아무리 늘려도 사라지지 않는다.")


def test_라벨_오류는_정확도_상한을_그_자리에서_깎는다() -> None:
    """모델을 아무리 키워도 넘을 수 없는 선이 생긴다."""
    measurement = LabelErrorMeasurement(
        total_labeled=10_000,
        rule_violations={"FAULT → power ≤ 30": 200},
        conflicting_duplicate_count=100,
    )
    result = LabelQualityPolicy().evaluate(
        measurement,
        rules=(
            LabelConsistencyRule(
                label="FAULT",
                field_name="active_power_kw",
                expected_max=30.0,
                description="보호 계전기 동작 시 부하 차단",
            ),
        ),
    )

    report.block(
        "라벨 오류 3% 가 만드는 천장",
        f"  규칙 위반   : {measurement.violation_ratio:.2%}\n"
        f"  라벨 모순   : {measurement.conflict_ratio:.2%}\n"
        f"  정확도 상한 : {measurement.accuracy_ceiling():.2%}",
    )
    assert measurement.accuracy_ceiling() == pytest.approx(0.97)
    assert "LABEL_ACCURACY_CEILING" in {f.code for f in result.findings}
    report.note(
        "'정확도 99% 목표'를 세우기 전에, 라벨이 그 숫자를 허용하는지부터 봐야 한다."
    )


def test_기준선_데이터는_규칙과_모순되지_않는다(
    container, quality_container, quality
) -> None:
    view = measure(container, quality_container, "clean", quality.clean)
    report.block("오염 없는 데이터", view.render())
    assert view.verdict == "PASSED"
    assert view.score == 100.0
