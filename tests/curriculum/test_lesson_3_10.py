"""실습 3-10 — 현장 데이터를 통과하는 모델만 살아남는다.

    pytest -m lesson_3_10 -s

학습이 끝났다고 모델이 완성된 것이 아니다.
"검증 정확도가 제일 높은 모델"과 "현장에 내보낼 수 있는 모델"은 다르다.

    모듈 1  이 데이터가 무엇인지 아는가
    모듈 2  이 데이터가 쓸 만한가
    모듈 3  이 **모델**이 쓸 만한가
"""

from __future__ import annotations

import pytest

from application.model.accept_model import AcceptModelCommand, ReopenTrainingRunCommand
from domain.model.acceptance import LatencyBudget, ModelAcceptancePolicy
from domain.model.evaluation import EvaluationPolicy
from domain.shared.errors import IllegalStateTransition
from tests.support import model_scenario as ms
from tests.support import report

pytestmark = pytest.mark.lesson_3_10


@pytest.fixture(scope="module")
def overlapping(model_data):  # noqa: ANN201
    """창이 67% 겹치는 학습. 표본은 많고 분할은 샌다."""
    return ms.build_pipeline(
        dataset_id="acc-overlap",
        assessment_id="qa-acc-overlap",
        run_id="run-overlap",
        train_path=model_data.train,
        stride=10,
    )


@pytest.fixture(scope="module")
def disjoint(model_data):  # noqa: ANN201
    """창이 겹치지 않는 학습. 표본은 적고 분할은 깨끗하다."""
    return ms.build_pipeline(
        dataset_id="acc-disjoint",
        assessment_id="qa-acc-disjoint",
        run_id="run-clean",
        train_path=model_data.train,
        stride=30,
    )


def field_policy() -> ModelAcceptancePolicy:
    """현장이 정하는 승인 기준.

    FAULT 를 놓치는 것과 다른 클래스를 놓치는 것은 무게가 다르다.
    그리고 30ms 는 설비의 사이클 타임이 정한 숫자다.
    """
    return ModelAcceptancePolicy(
        evaluation=EvaluationPolicy(
            critical_labels=frozenset({"FAULT"}), min_critical_recall=0.9
        ),
        latency=LatencyBudget(p95_ms=30.0),
    )


def test_정확도_97퍼센트_모델이_막힌다(overlapping) -> None:
    report.section("실습 3-10 · 현장 데이터를 통과하는 모델만 살아남는다")

    certificate = ms.accept(overlapping.model, overlapping.run_id, policy=field_policy())
    report.block("겹치는 창으로 학습한 모델", certificate.render())

    assert certificate.is_deployable is False
    reasons = {f.code for f in certificate.blocking}
    assert "PROTOCOL_SPLIT_OVERLAP" in reasons

    report.note(
        f"정확도 {certificate.accuracy:.1%}. 나쁘지 않은 숫자다. "
        "그런데 분할 경계에서 40 표본이 새고 있었다(실습 3-8)."
    )
    report.note(
        "**그 정확도는 시험 문제를 일부 보고 낸 점수다.** 숫자 자체를 믿을 수 없다."
    )


def test_겹침을_없앤_모델은_통과한다(disjoint) -> None:
    certificate = ms.accept(disjoint.model, disjoint.run_id, policy=field_policy())
    report.block("겹치지 않는 창으로 학습한 모델", certificate.render())

    assert certificate.is_deployable is True
    report.note(f"macro recall {certificate.macro_recall:.3f} / FAULT 도 놓치지 않았다.")

    test_count = disjoint.preparation.summaries[2].sample_count
    report.note(
        f"대신 평가 표본이 {test_count}개다 — 겹치는 쪽은 195개였다. "
        f"{test_count}개로 낸 정확도 {certificate.accuracy:.1%} 를 얼마나 믿을 것인가. "
        "이것이 겹침을 없앤 대가다."
    )


def test_다른_날_데이터로_다시_평가한다(disjoint, model_data) -> None:
    """test 분할은 같은 36시간에서 나온다. 같은 날, 같은 조건이다."""
    from application.model.evaluate_model import EvaluateOnFieldCommand

    run_id = disjoint.run_id
    disjoint.model.reopen_training_run().execute(
        ReopenTrainingRunCommand(run_id=run_id, reason="현장 홀드아웃 평가 추가")
    )
    view = disjoint.model.evaluate_model_on_field().execute(
        EvaluateOnFieldCommand(
            run_id=run_id,
            field_uri=str(model_data.field),
            policy=EvaluationPolicy(
                critical_labels=frozenset({"FAULT"}), min_critical_recall=0.9
            ),
        )
    )
    report.block("현장 홀드아웃 (이틀 뒤 12시간)", view.render())

    assert view.split == "field"
    report.note(
        f"test 정확도 {disjoint.evaluation.accuracy:.1%} → "
        f"현장 정확도 {view.accuracy:.1%}"
    )
    report.note(
        "정규화 통계는 학습 때 쓰던 값을 그대로 썼다(실습 1-7). "
        "현장 데이터로 다시 뽑으면 그건 배포된 모델과 다른 전처리다."
    )

    certificate = ms.accept(disjoint.model, run_id, split="field", policy=field_policy())
    report.block("현장 데이터 기준 승인 판정", certificate.render())
    assert certificate.model_version_id


def test_지연시간_예산을_넘으면_정확도와_무관하게_막힌다(disjoint) -> None:
    """현장이 30ms 를 요구하면, 31ms 짜리 모델은 못 쓴다."""
    disjoint.model.reopen_training_run().execute(
        ReopenTrainingRunCommand(run_id=disjoint.run_id, reason="지연시간 기준 재검토")
    )
    tight = ModelAcceptancePolicy(
        evaluation=EvaluationPolicy(critical_labels=frozenset({"FAULT"})),
        latency=LatencyBudget(p95_ms=0.01),  # 10마이크로초
    )
    certificate = disjoint.model.accept_model().execute(
        AcceptModelCommand(run_id=disjoint.run_id, split="test", policy=tight)
    )

    assert certificate.is_deployable is False
    assert any(
        f.code == "ACCEPT_LATENCY_OVER_BUDGET" for f in certificate.blocking
    )
    report.block("지연시간 예산 0.01ms", certificate.render())
    report.note(
        "정확도는 그대로다. 그런데 못 쓴다. "
        "**정확도와 지연시간은 같은 저울에 올릴 수 없다** — 둘 다 통과해야 한다."
    )
    report.note("이 예산을 맞추는 일이 모듈 4(최적화) 전체다.")


def test_판정된_모델은_몰래_바뀌지_않는다(disjoint) -> None:
    from application.model.evaluate_model import EvaluateModelCommand

    disjoint.model.reopen_training_run().execute(
        ReopenTrainingRunCommand(run_id=disjoint.run_id, reason="상태 확인")
    )
    ms.accept(disjoint.model, disjoint.run_id, policy=field_policy())

    with pytest.raises(IllegalStateTransition, match="reopen"):
        disjoint.model.evaluate_model().execute(
            EvaluateModelCommand(run_id=disjoint.run_id, split="validation")
        )

    status = disjoint.model.reopen_training_run().execute(
        ReopenTrainingRunCommand(
            run_id=disjoint.run_id, reason="현장 재수집분으로 재평가"
        )
    )
    assert status == "COMPLETED"
    report.note(
        "승인된 모델에 평가를 덧붙이는 경로가 없다. "
        "되돌리려면 이유를 남겨야 하고, 그 기록이 감사 근거가 된다."
    )


def test_모듈_3의_판정도_이벤트로_남는다(disjoint) -> None:
    names = disjoint.events.names()
    assert "ModelEvaluated" in names
    assert "ModelAccepted" in names or "ModelRejected" in names

    report.block(
        "모델이 남긴 Event",
        "\n".join(
            f"  {name}"
            for name in dict.fromkeys(n for n in names if n.startswith("Model"))
        ),
    )
    report.note(
        "어느 데이터로 학습했고, 어떤 기준으로 통과시켰는지가 남는다. "
        "모듈 5(운영)에서 '이 모델이 왜 배포되었나'를 되짚을 근거다."
    )
