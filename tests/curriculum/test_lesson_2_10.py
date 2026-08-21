"""실습 2-10 — AI를 학습시키기 전에 Data Quality Gate를 통과시켜라.

    pytest -m lesson_2_10 -s

모듈 1의 판정과 같은 자리에 있지만 묻는 것이 다르다.

    모듈 1  이 데이터가 무엇인지 아는가   (구조와 계약)
    모듈 2  이 데이터가 쓸 만한가          (오염도)

둘 다 통과해야 모델 학습으로 넘어간다.
"""

from __future__ import annotations

import pytest

from application.data.certify_dataset_readiness import CertifyDatasetReadinessCommand
from application.data_quality.evaluate_quality_gate import ReopenAssessmentCommand
from application.data_quality.get_assessment import GetAssessmentQuery
from application.data_quality.measure_noise import MeasureNoiseCommand
from domain.data_quality.dimensions import QualityDimension
from domain.data_quality.gate import QualityGatePolicy
from domain.shared.errors import IllegalStateTransition
from tests.support import report
from tests.support import quality_scenario as qs

pytestmark = pytest.mark.lesson_2_10


def test_모듈1을_통과한_데이터가_여기서_막힌다(
    container, quality_container, quality
) -> None:
    report.section("실습 2-10 · Data Quality Gate를 통과시켜라")

    qs.full_assessment(
        container,
        quality_container,
        dataset_id="dirty",
        assessment_id="qa-dirty",
        path=quality.dirty,
    )

    structural = container.certify_dataset_readiness().execute(
        CertifyDatasetReadinessCommand(
            dataset_id="dirty", policy=qs.structural_readiness_policy()
        )
    )
    gate = qs.run_gate(quality_container, "qa-dirty")

    report.block("모듈 1 — 이 데이터가 무엇인지 아는가", structural.render())
    report.block("모듈 2 — 이 데이터가 쓸 만한가", gate.render())

    assert structural.is_ready is True
    assert gate.is_ready is False
    assert len(gate.blocking_reasons) >= 3

    report.note(
        "같은 파일, 두 게이트. 하나는 통과하고 하나는 막는다. "
        "둘은 다른 질문을 하고 있다."
    )


def test_점수와_차단축을_함께_본다(container, quality_container, quality) -> None:
    """모듈 1은 순수하게 심각도로 막았다. 품질은 그렇게 다룰 수 없다."""
    qs.full_assessment(
        container,
        quality_container,
        dataset_id="dirty",
        assessment_id="qa-dirty",
        path=quality.dirty,
    )
    gate = qs.run_gate(quality_container, "qa-dirty")

    reasons = "\n".join(gate.blocking_reasons)
    assert "COMPLETENESS" in reasons  # 차단 축
    assert "LABEL_QUALITY" in reasons  # 차단 축
    assert "NOISE" in reasons          # 점수 미달
    assert "종합 점수" in reasons       # 종합 미달

    report.block("차단 사유의 종류", "\n".join(f"  ✗ {r}" for r in gate.blocking_reasons))
    report.note(
        "차단 축(라벨·결측)은 판정이 FAILED 면 점수와 무관하게 막는다. "
        "나머지 축은 점수로 막는다. 잡음이 조금 있는 데이터를 매번 막으면 아무도 이 게이트를 쓰지 않는다."
    )


def test_정리된_데이터는_통과한다(container, quality_container, quality) -> None:
    qs.full_assessment(
        container,
        quality_container,
        dataset_id="clean",
        assessment_id="qa-clean",
        path=quality.clean,
    )
    gate = qs.run_gate(quality_container, "qa-clean")

    report.block("정리된 데이터", gate.render())

    assert gate.is_ready is True
    assert gate.overall_score > 95
    assert gate.grade == "A"
    assert gate.warnings  # 경고는 남는다
    report.note(
        "불균형 경고는 남은 채로 통과했다. "
        "현장이 원래 그렇고, 데이터를 고쳐서 없앨 문제가 아니기 때문이다."
    )


def test_측정하지_않은_축이_있으면_통과할_수_없다(
    container, quality_container, quality
) -> None:
    from application.data_quality.measure_completeness import (
        MeasureCompletenessCommand,
    )

    qs.prepare_dataset(container, "clean", quality.clean)
    qs.start(quality_container, "qa-partial", "clean")
    quality_container.measure_completeness().execute(
        MeasureCompletenessCommand(assessment_id="qa-partial")
    )
    gate = qs.run_gate(quality_container, "qa-partial")

    report.block("한 축만 측정한 경우", gate.render())

    assert gate.is_ready is False
    assert len(gate.missing_dimensions) == 5
    assert gate.overall_score == pytest.approx(100.0)
    report.note(
        "종합 점수는 100점인데 막혔다. "
        "'측정하지 않았다'는 '문제 없다'가 아니다 — 모듈 1과 같은 원칙이다."
    )


def test_라인마다_기준이_다를_수_있다(container, quality_container, quality) -> None:
    """안전 등급이 높은 라인은 더 조인다."""
    qs.full_assessment(
        container,
        quality_container,
        dataset_id="clean",
        assessment_id="qa-clean",
        path=quality.clean,
    )

    보통 = qs.run_gate(quality_container, "qa-clean")
    assert 보통.is_ready is True

    quality_container.reopen_assessment().execute(
        ReopenAssessmentCommand(assessment_id="qa-clean", reason="안전 등급 재검토")
    )
    엄격 = qs.run_gate(
        quality_container,
        "qa-clean",
        QualityGatePolicy(
            minimum_overall_score=99.5,
            minimum_dimension_score=96.0,
            blocking_dimensions=frozenset(QualityDimension),
        ),
    )

    report.block(
        "같은 데이터에 대한 두 기준",
        f"  일반 라인 : {보통.verdict} (종합 {보통.overall_score:.1f})\n"
        f"  안전 라인 : {엄격.verdict} (종합 {엄격.overall_score:.1f})\n"
        + "\n".join(f"    ✗ {r}" for r in 엄격.blocking_reasons),
    )
    assert 엄격.is_ready is False


def test_판정된_평가는_몰래_바뀌지_않는다(
    container, quality_container, quality
) -> None:
    qs.full_assessment(
        container,
        quality_container,
        dataset_id="clean",
        assessment_id="qa-clean",
        path=quality.clean,
    )
    qs.run_gate(quality_container, "qa-clean")

    with pytest.raises(IllegalStateTransition, match="reopen"):
        quality_container.measure_noise().execute(
            MeasureNoiseCommand(assessment_id="qa-clean")
        )

    quality_container.reopen_assessment().execute(
        ReopenAssessmentCommand(
            assessment_id="qa-clean", reason="전압 계측 배선 교체 후 재수집"
        )
    )
    view = quality_container.get_assessment().execute(
        GetAssessmentQuery(assessment_id="qa-clean")
    )
    assert view.status == "MEASURING"
    assert view.verdict is None


def test_전체_과정이_이벤트로_남는다(
    container, quality_container, quality, events
) -> None:
    qs.full_assessment(
        container,
        quality_container,
        dataset_id="dirty",
        assessment_id="qa-dirty",
        path=quality.dirty,
    )
    qs.run_gate(quality_container, "qa-dirty")

    names = events.names()
    assert "QualityAssessmentStarted" in names
    assert names.count("QualityDimensionMeasured") == 6
    assert "QualityGateBlocked" in names

    report.block(
        "품질 평가가 남긴 Event",
        "\n".join(
            f"  {name}" for name in names if name.startswith("Quality") or name == "RemediationRecorded"
        ),
    )
    report.note(
        "누가 언제 무엇을 측정했고 왜 막혔는지가 남는다. "
        "석 달 뒤 '그때 왜 학습을 시작했나'에 답할 수 있는 유일한 근거다."
    )
