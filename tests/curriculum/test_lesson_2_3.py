"""실습 2-3 — 이상치는 데이터에 숨어 있는 사고다.

    pytest -m lesson_2_3 -s

이상치를 지우기 전에 물어야 할 것: **이게 오류인가, 사고인가?**
"""

from __future__ import annotations

import pytest

from application.data_quality.measure_validity import MeasureValidityCommand
from domain.data_quality.validity import (
    FieldOutliers,
    OutlierMeasurement,
    ValidityPolicy,
)
from tests.support import report
from tests.support import quality_scenario as qs

pytestmark = pytest.mark.lesson_2_3


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def measure(container, quality_container, name, path):  # noqa: ANN001
    qs.prepare_dataset(container, name, path)
    qs.start(quality_container, f"qa-{name}", name)
    return quality_container.measure_validity().execute(
        MeasureValidityCommand(assessment_id=f"qa-{name}")
    )


def test_이상치가_정상_라벨_구간에_몰려_있다(
    container, quality_container, quality
) -> None:
    report.section("실습 2-3 · 이상치는 데이터에 숨어 있는 사고다")

    view = measure(container, quality_container, "dirty", quality.dirty)
    report.block("이상치 검사", view.render())

    found = codes(view)
    assert "VALIDITY_OUTLIER_IN_NORMAL" in found
    assert view.verdict == "FAILED"

    finding = next(f for f in view.findings if f.code == "VALIDITY_OUTLIER_IN_NORMAL")
    assert finding.measured is not None and finding.measured > 0.9
    report.note(
        f"유효전력 이상치의 {finding.measured:.0%} 가 NORMAL 구간에 있다. "
        "설비가 340~390kW 를 쓰는 동안 아무도 이상하다고 기록하지 않았다."
    )
    report.note("여기서 할 일은 삭제가 아니라 조사다.")


def test_변화율_위반은_값_하나만_봐서는_잡히지_않는다(
    container, quality_container, quality
) -> None:
    view = measure(container, quality_container, "dirty", quality.dirty)

    assert "VALIDITY_RATE_VIOLATION" in codes(view)
    assert "VALIDITY_OUT_OF_RANGE" not in codes(view)

    report.note(
        "전류 540A 는 물리 범위(0~600A) 안이다. 값 하나만 보면 정상이다. "
        "직전 표본과 비교해야 '10초 만에 300A 가 뛰었다'가 보인다."
    )


def test_z_score_는_이상치가_많아지면_눈이_먼다() -> None:
    """masking — 평균과 표준편차가 이상치에 오염되어 이상치를 못 찾는 현상."""
    policy = ValidityPolicy(max_masking_gap_ratio=0.005)
    result = policy.evaluate(
        OutlierMeasurement(
            fields=(
                FieldOutliers(
                    field_name="active_power_kw",
                    total_count=10_000,
                    z_outlier_count=12,     # 평균·표준편차가 이미 끌려갔다
                    mad_outlier_count=340,  # 중앙값 기준으로는 340건
                ),
            )
        )
    )
    codes_found = {f.code for f in result.findings}
    assert "VALIDITY_ZSCORE_MASKED" in codes_found

    report.block(
        "같은 데이터, 다른 척도",
        "  z-score(평균 기반) :  12건\n"
        "  MAD(중앙값 기반)   : 340건\n"
        "  → 이상치가 평균과 표준편차를 자기 쪽으로 끌어당겼다.",
    )
    report.note("이상치를 찾는 도구가 이상치에 오염되면, 많을수록 못 찾게 된다.")


def test_이상치는_그_자체로_나쁜_데이터가_아니다(
    container, quality_container, quality
) -> None:
    """오염 없는 데이터에도 이상치는 있다. 설비가 실제로 정지했기 때문이다."""
    view = measure(container, quality_container, "clean", quality.clean)
    report.block("오염 없는 데이터의 이상치 검사", view.render())

    assert view.verdict == "PASSED_WITH_WARNINGS"
    assert "VALIDITY_OUTLIER_IN_NORMAL" not in codes(view)
    assert "VALIDITY_OUTLIER_RATIO" in codes(view)

    report.note(
        "spindle_rpm 이상치 150건 — 전부 FAULT 구간이다. "
        "이건 잡음이 아니라 우리가 찾으려는 바로 그 사건이다."
    )
    report.note("지웠다면 학습할 것이 사라진다. 그래서 WARNING 이고 CRITICAL 이 아니다.")


def test_무엇이_정상_라벨인지는_Policy_가_안다() -> None:
    """측정기는 라벨별 개수만 센다. '정상'이 무엇인지는 현장이 정한다."""
    measurement = OutlierMeasurement(
        fields=(
            FieldOutliers(
                field_name="active_power_kw",
                total_count=1000,
                mad_outlier_count=100,
                outliers_by_label={"NORMAL": 95, "FAULT": 5},
            ),
        )
    )

    모름 = ValidityPolicy(normal_label=None).evaluate(measurement)
    assert "VALIDITY_OUTLIER_IN_NORMAL" not in {f.code for f in 모름.findings}

    앎 = ValidityPolicy(normal_label="NORMAL").evaluate(measurement)
    assert "VALIDITY_OUTLIER_IN_NORMAL" in {f.code for f in 앎.findings}

    report.note(
        "같은 측정값, 다른 판정. 라인마다 '정상' 라벨의 이름이 다르므로 "
        "이 지식은 측정기가 아니라 Policy 에 있어야 한다."
    )
