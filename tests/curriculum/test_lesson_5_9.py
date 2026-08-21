"""실습 5-9 — 새 모델과 기존 모델을 실제 데이터로 비교하라.

    pytest -m lesson_5_9 -s

오프라인 평가에서 새 모델이 더 좋았다. 그래서 배포한다?

**아니다.** 오프라인 평가는 학습 때 모아 둔 데이터로 한 것이다.
현장은 그 데이터가 아니다 — 모듈 5 전체가 그 사실 위에 서 있었다.

그래서 같은 현장 입력을 둘 다에게 넣는다. **새 모델의 답은 쓰지 않는다.**
"""

from __future__ import annotations

import pytest

from domain.operations.shadow import PromotionPolicy, ShadowRun
from domain.operations.window import ObservationWindow
from tests.support import operations_scenario as os5
from tests.support import report

pytestmark = pytest.mark.lesson_5_9


@pytest.fixture
def shadowed(operations_container, deployed):  # noqa: ANN001, ANN201
    artifact_id = os5.attach_shadow(operations_container, deployed)
    return operations_container, artifact_id


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def window(**overrides) -> ObservationWindow:  # noqa: ANN003
    base: dict[str, object] = dict(
        label="비교 창",
        started_at="2026-05-23 00:00:00",
        ended_at="2026-05-23 23:59:59",
        sample_count=0,
    )
    base.update(overrides)
    return ObservationWindow(**base)  # type: ignore[arg-type]


def test_같은_입력에_두_모델을_나란히_돌린다(shadowed) -> None:
    report.section("실습 5-9 · 새 모델과 기존 모델을 실제 데이터로 비교하라")

    operations, artifact_id = shadowed
    view = os5.compare_shadow(operations, window(), artifact_id)
    report.block("그림자 비교 (4일차 하루)", view.render())

    assert view.sample_count > 500
    assert 0.0 <= view.agreement_ratio <= 1.0
    report.note(
        "설비는 여전히 기존 모델의 답으로 움직였다. "
        "새 모델은 옆에서 같은 입력을 받아 **답만 남겼다.**"
    )


def test_다르다와_낫다는_다른_말이다(shadowed) -> None:
    """**이 실습에서 정직해야 하는 지점.**"""
    operations, artifact_id = shadowed
    view = os5.compare_shadow(operations, window(), artifact_id)

    report.block(
        "이 비교가 답한 것과 답하지 못한 것",
        "\n".join(
            [
                f"  답이 갈린 비율   : {1 - view.agreement_ratio:.1%}  ← 정답 없이도 안다",
                f"  정답 붙은 표본   : {view.labeled_count:,}건",
                f"  기존 정확도      : {view.incumbent_accuracy:.4f}",
                f"  새 모델 정확도   : {view.candidate_accuracy:.4f}",
                f"  차이             : {view.accuracy_gain:+.4f}  ← 정답이 있어야 안다",
            ]
        ),
    )
    assert view.labeled_count > 0
    report.note(
        "두 모델이 **다르다**는 것은 정답 없이 말할 수 있다. "
        "어느 쪽이 **낫다**는 것은 정답이 있어야 말할 수 있다."
    )


def test_정답이_없으면_승격시키지_않는다() -> None:
    unlabeled = ShadowRun(
        window=window(sample_count=5_000),
        incumbent_label="TFLITE/FP16",
        candidate_label="TFLITE/INT8",
        sample_count=5_000,
        agreement_count=4_800,
        incumbent_p95_ms=0.075,
        candidate_p95_ms=0.070,
        labeled_count=0,
    )
    verdict = PromotionPolicy().evaluate(unlabeled)
    report.block("정답이 하나도 없는 비교", verdict.render())

    assert "SHADOW_NO_LABELS" in {f.code for f in verdict.findings}
    assert not verdict.promote
    report.note(
        "4% 에서 답이 갈렸다. 그런데 **어느 쪽이 맞았는지 모른다.** "
        "이걸 '더 낫다'로 읽는 순간 근거 없는 배포가 된다."
    )


def test_한나절은_돌려_봐야_한다() -> None:
    """현장 조건은 시간대마다 다르다."""
    brief = ShadowRun(
        window=window(sample_count=40),
        incumbent_label="A",
        candidate_label="B",
        sample_count=40,
        agreement_count=38,
        incumbent_p95_ms=0.075,
        candidate_p95_ms=0.070,
        labeled_count=40,
        incumbent_correct=30,
        candidate_correct=34,
    )
    verdict = PromotionPolicy().evaluate(brief)
    assert "SHADOW_TOO_FEW_SAMPLES" in {f.code for f in verdict.findings}
    assert not verdict.promote
    report.note(
        "40건에서 새 모델이 4건 더 맞았다. 그 4건은 우연일 수 있다. "
        "야간 교대, 제품 전환, 온도 변화 — 하루가 다 다르다."
    )


def test_정확도를_얻고_사이클을_잃는_거래() -> None:
    slower = ShadowRun(
        window=window(sample_count=5_000),
        incumbent_label="TFLITE/FP16",
        candidate_label="ONNX/FP32",
        sample_count=5_000,
        agreement_count=4_700,
        incumbent_p95_ms=0.075,
        candidate_p95_ms=0.210,
        labeled_count=600,
        incumbent_correct=500,
        candidate_correct=520,
    )
    verdict = PromotionPolicy().evaluate(slower)
    report.block("더 정확한데 2.8배 느린 새 모델", verdict.render())

    assert "SHADOW_SLOWER" in {f.code for f in verdict.findings}
    assert not verdict.promote
    report.note(
        f"정확도는 {slower.accuracy_gain:+.4f} 올랐다. 그런데 2.8배 느리다. "
        "모듈 4 의 예산을 다시 확인해야 하는 상황이다."
    )


def test_오프라인에서_좋았다고_현장에서_좋은_것은_아니다() -> None:
    """이 실습의 결론."""
    worse = ShadowRun(
        window=window(sample_count=5_000),
        incumbent_label="TFLITE/FP16",
        candidate_label="v2-tflite-fp16",
        sample_count=5_000,
        agreement_count=4_100,
        incumbent_p95_ms=0.075,
        candidate_p95_ms=0.074,
        labeled_count=600,
        incumbent_correct=498,
        candidate_correct=474,
    )
    verdict = PromotionPolicy().evaluate(worse)
    report.block("오프라인 평가에서는 더 좋았던 모델", verdict.render())

    assert "SHADOW_NOT_BETTER" in {f.code for f in verdict.findings}
    report.note(
        "오프라인 평가에서 좋았던 것은 **그 데이터에서** 좋았다는 뜻이다. "
        "현장 데이터는 그 데이터가 아니다."
    )
    report.note(
        "그림자 비교를 건너뛰면 이 사실을 배포한 뒤에 알게 된다."
    )


def test_답이_갈리는_것_자체는_나쁜_게_아니다() -> None:
    diverging = ShadowRun(
        window=window(sample_count=5_000),
        incumbent_label="A",
        candidate_label="B",
        sample_count=5_000,
        agreement_count=3_900,
        incumbent_p95_ms=0.075,
        candidate_p95_ms=0.074,
        labeled_count=600,
        incumbent_correct=480,
        candidate_correct=540,
    )
    verdict = PromotionPolicy(min_agreement_ratio=0.9).evaluate(diverging)
    found = {f.code for f in verdict.findings}

    assert "SHADOW_HIGH_DISAGREEMENT" in found
    assert verdict.promote  # WARNING 이지 CRITICAL 이 아니다
    report.note(
        "22% 에서 답이 갈렸는데 새 모델이 60건 더 맞았다. "
        "**갈렸다는 것은 나아졌다는 뜻일 수도 있다.**"
    )
    report.note(
        "그래서 이건 WARNING 이다 — 막지는 않되, 갈린 건들을 사람이 봐야 한다."
    )


def test_그림자_실행기가_없으면_비교할_수_없다(operations_container) -> None:
    from application.shared.errors import UnsupportedOperation

    operations_container.shadow = None
    with pytest.raises(UnsupportedOperation) as caught:
        os5.compare_shadow(operations_container, window(), "무엇이든")
    report.note(str(caught.value))
    report.note(
        "새 모델을 **실제로 돌릴 수 있어야** 비교가 성립한다. "
        "숫자만 비교하는 것은 오프라인 평가와 같다."
    )
