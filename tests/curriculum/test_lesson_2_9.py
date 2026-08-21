"""실습 2-9 — 망가진 데이터와 정상 데이터를 직접 비교하라.

    pytest -m lesson_2_9 -s

"좋아졌다"는 근거가 아니다.
어느 축이 몇 점 올랐고, 학습 가능 표본이 몇 개 바뀌었는지까지 말해야 근거다.
"""

from __future__ import annotations

import pytest

from application.data_quality.compare_quality import CompareQualityCommand
from application.data_quality.record_remediation import RecordRemediationCommand
from domain.data_quality.dimensions import QualityDimension
from domain.data_quality.remediation import RemediationAction, RemediationKind
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from tests.support import report
from tests.support import quality_scenario as qs

pytestmark = pytest.mark.lesson_2_9


def assess_both(container, quality_container, quality) -> None:  # noqa: ANN001
    qs.full_assessment(
        container,
        quality_container,
        dataset_id="dirty",
        assessment_id="qa-before",
        path=quality.dirty,
    )
    qs.full_assessment(
        container,
        quality_container,
        dataset_id="clean",
        assessment_id="qa-after",
        path=quality.clean,
    )


def test_조치_전후를_축별로_비교한다(container, quality_container, quality) -> None:
    report.section("실습 2-9 · 망가진 데이터와 정상 데이터를 직접 비교하라")

    assess_both(container, quality_container, quality)
    view = quality_container.compare_quality().execute(
        CompareQualityCommand(
            before_assessment_id="qa-before",
            after_assessment_id="qa-after",
            before_label="조치 전(dirty)",
            after_label="조치 후(clean)",
        )
    )
    report.block("품질 비교", view.render())

    assert view.overall_delta > 20
    assert set(view.improved) >= {
        QualityDimension.COMPLETENESS.value,
        QualityDimension.LABEL_QUALITY.value,
        QualityDimension.NOISE.value,
        QualityDimension.UNIQUENESS.value,
        QualityDimension.VALIDITY.value,
    }
    report.note(
        f"종합 {view.before_overall:.1f} → {view.after_overall:.1f} "
        f"({view.overall_delta:+.1f})"
    )


def test_품질을_고치면_나빠_보이는_지표도_생긴다(
    container, quality_container, quality
) -> None:
    """이것이 이 실습에서 가장 중요한 장면이다."""
    assess_both(container, quality_container, quality)
    view = quality_container.compare_quality().execute(
        CompareQualityCommand(
            before_assessment_id="qa-before", after_assessment_id="qa-after"
        )
    )

    assert view.regressed == (QualityDimension.BALANCE.value,)
    before, after = view.before_impact, view.after_impact
    assert after.minority_count < before.minority_count

    report.block(
        "BALANCE 만 점수가 내려갔다",
        f"  소수 클래스(FAULT) : {before.minority_count} → {after.minority_count}\n"
        f"  BALANCE 점수       : "
        f"{dict((n, b) for n, b, _, _ in view.deltas)['BALANCE']:.1f} → "
        f"{dict((n, a) for n, _, a, _ in view.deltas)['BALANCE']:.1f}",
    )
    report.note(
        "라벨 오류가 FAULT 를 180건으로 부풀리고 있었다. "
        "정정하니 실제 값인 150건이 드러났다."
    )
    report.note(
        "**나빠진 것이 아니라, 부풀려져 있던 것이 드러난 것이다.** "
        "품질을 고치면 어떤 지표는 나빠 보인다. 그 방향을 뒤집으려 하면 안 된다."
    )


def test_학습_관점의_변화가_함께_보여야_한다(
    container, quality_container, quality
) -> None:
    assess_both(container, quality_container, quality)
    view = quality_container.compare_quality().execute(
        CompareQualityCommand(
            before_assessment_id="qa-before", after_assessment_id="qa-after"
        )
    )

    before, after = view.before_impact, view.after_impact
    assert before is not None and after is not None

    report.block(
        "표본 수로 환산한 변화",
        f"  학습 가능 표본 : {before.usable_rows:,} → {after.usable_rows:,}\n"
        f"  라벨 모순      : {before.conflicting_rows:,} → {after.conflicting_rows:,}\n"
        f"  정확도 상한    : {before.accuracy_ceiling:.2%} → {after.accuracy_ceiling:.2%}\n"
        f"  결측 포함 행   : {before.rows_with_missing:,} → {after.rows_with_missing:,}",
    )

    assert after.conflicting_rows == 0
    assert after.accuracy_ceiling > before.accuracy_ceiling
    assert after.rows_with_missing < before.rows_with_missing
    report.note("점수만 비교하면 '올랐다'로 끝난다. 무엇을 얻었는지는 여기서 나온다.")


def test_데이터를_고친_기록에는_근거가_있어야_한다() -> None:
    """근거 없는 데이터 수정은 조작과 구분되지 않는다."""
    with pytest.raises(InvariantViolation, match="근거가 없다"):
        RemediationAction(
            kind=RemediationKind.IMPUTE,
            dimension=QualityDimension.COMPLETENESS,
            target="temperature_c",
            affected_rows=347,
            rationale="",
            decided_by="데이터팀",
        )

    with pytest.raises(InvariantViolation, match="누가 결정했는지"):
        RemediationAction(
            kind=RemediationKind.IMPUTE,
            dimension=QualityDimension.COMPLETENESS,
            target="temperature_c",
            affected_rows=347,
            rationale="LOT 3개 구간은 잘라내고 나머지는 선형 보간",
            decided_by="  ",
        )

    report.note("데이터를 고치는 것은 되돌릴 수 없는 행위다. 기록이 유일한 안전장치다.")


def test_측정하지_않은_축은_고칠_수_없다(
    container, quality_container, quality
) -> None:
    """무엇을 고쳤는지 확인할 방법이 없기 때문이다."""
    from application.data_quality.measure_completeness import (
        MeasureCompletenessCommand,
    )

    qs.prepare_dataset(container, "dirty", quality.dirty)
    qs.start(quality_container, "qa", "dirty")
    quality_container.measure_completeness().execute(
        MeasureCompletenessCommand(assessment_id="qa")
    )

    with pytest.raises(IllegalStateTransition, match="측정하지 않은 채로"):
        quality_container.record_remediation().execute(
            RecordRemediationCommand(
                assessment_id="qa",
                action=RemediationAction(
                    kind=RemediationKind.DEDUPLICATE,
                    dimension=QualityDimension.UNIQUENESS,  # 측정한 적 없다
                    target="feature_vector",
                    affected_rows=276,
                    rationale="입력이 동일한 행을 제거",
                    decided_by="데이터팀",
                ),
            )
        )


def test_고쳤다는_주장은_재측정으로만_확인된다(
    container, quality_container, quality
) -> None:
    """현장에서 가장 흔한 사고 — 고쳤다고 하고 확인하지 않는 것."""
    from application.data_quality.measure_completeness import (
        MeasureCompletenessCommand,
    )

    qs.full_assessment(
        container,
        quality_container,
        dataset_id="dirty",
        assessment_id="qa",
        path=quality.dirty,
    )
    view = quality_container.record_remediation().execute(
        RecordRemediationCommand(
            assessment_id="qa",
            action=RemediationAction(
                kind=RemediationKind.EXCLUDE_SEGMENT,
                dimension=QualityDimension.COMPLETENESS,
                target="temperature_c / LOT 3개 구간",
                affected_rows=259,
                rationale=(
                    "결측이 3개 LOT 에 몰려 있어 보간하면 그 구간의 현실이 조작된다. "
                    "해당 LOT 을 학습 대상에서 제외한다."
                ),
                decided_by="데이터팀 · 설비운영팀 합의 (2026-04-10)",
            ),
        )
    )

    report.block(
        "조치 기록",
        "\n".join(f"  {line}" for line in view.remediations)
        + f"\n  상태: {view.status}"
        + f"\n  미검증 축: {', '.join(view.unverified_dimensions)}",
    )

    assert view.status == "REMEDIATING"
    assert view.unverified_dimensions == ("COMPLETENESS",)

    gate = qs.run_gate(quality_container, "qa")
    assert gate.is_ready is False
    assert any("재측정하지 않았다" in reason for reason in gate.blocking_reasons)
    report.note("조치를 기록한 것만으로는 게이트를 통과할 수 없다.")

    # 재측정하면 미검증 표시가 사라진다.
    quality_container.reopen_assessment().execute(
        __import__(
            "application.data_quality.evaluate_quality_gate", fromlist=["x"]
        ).ReopenAssessmentCommand(assessment_id="qa", reason="재측정 진행")
    )
    quality_container.measure_completeness().execute(
        MeasureCompletenessCommand(assessment_id="qa")
    )
    from application.data_quality.get_assessment import GetAssessmentQuery

    after = quality_container.get_assessment().execute(
        GetAssessmentQuery(assessment_id="qa")
    )
    assert after.unverified_dimensions == ()
    report.note("재측정하면 '미검증' 표시가 사라진다. 그때부터 판정할 수 있다.")
