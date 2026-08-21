"""실습 3-8 — Training과 Validation을 분리하라.

    pytest -m lesson_3_8 -s

분할은 실습 1-8 에서 이미 했다. 그런데 분할해 놓고도 새는 경로가 남아 있다.
그리고 그중 하나는 이 프로젝트의 기본 설정에서 실제로 새고 있다.
"""

from __future__ import annotations

import pytest

from domain.model.protocol import EvaluationProtocol, SplitUsage
from domain.shared.inspection import Severity
from tests.support import model_scenario as ms
from tests.support import report

pytestmark = pytest.mark.lesson_3_8


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def test_창이_겹치면_나눠도_겹친다(trained) -> None:
    report.section("실습 3-8 · Training과 Validation을 분리하라")

    from application.model.support import load_run

    run = load_run(trained.model.runs, trained.run_id)
    usage = run.usage

    report.block(
        "분할 사용 기록",
        f"  train      : {usage.train_sample_count:>5,}개 ({usage.ratio_of('train'):.0%})\n"
        f"  validation : {usage.validation_sample_count:>5,}개\n"
        f"  test       : {usage.test_sample_count:>5,}개\n"
        f"  경계 겹침  : {usage.overlapping_samples} 표본",
    )

    assert usage.overlapping_samples == 40
    findings = EvaluationProtocol().inspect(usage)
    assert "PROTOCOL_SPLIT_OVERLAP" in codes(findings)

    report.note(
        "창 길이 30, stride 10 이므로 경계마다 20 표본이 양쪽에 걸친다. "
        "경계가 둘이니 40 표본. 시험 문제의 일부를 이미 본 것이다."
    )
    report.note(
        "실습 1-8 에서 '시간순으로 나눴으니 안전하다'고 했던 그 분할이다. "
        "**창으로 자르는 순간 다시 새기 시작한다.**"
    )


def test_겹침을_없애면_누수가_사라진다(trained_disjoint) -> None:
    from application.model.support import load_run

    run = load_run(trained_disjoint.model.runs, trained_disjoint.run_id)
    usage = run.usage

    assert usage.overlapping_samples == 0
    assert EvaluationProtocol().inspect(usage) == ()

    report.block(
        "stride 를 창 길이와 같게 하면",
        f"  경계 겹침  : {usage.overlapping_samples} 표본\n"
        f"  train      : {usage.train_sample_count:>5,}개\n"
        f"  validation : {usage.validation_sample_count:>5,}개\n"
        f"  test       : {usage.test_sample_count:>5,}개",
    )
    report.note(
        "대신 표본이 3분의 1로 줄었다. 겹침을 없앤 대가다. "
        "공짜로 얻는 것은 없다."
    )


def test_검증_집합을_스무_번_보면_그_점수에는_우연이_섞인다() -> None:
    usage = SplitUsage(
        train_sample_count=900,
        validation_sample_count=200,
        test_sample_count=200,
        validation_evaluations=50,
    )
    findings = EvaluationProtocol(max_validation_evaluations=20).inspect(usage)

    assert "PROTOCOL_VALIDATION_OVERUSED" in codes(findings)
    report.note(
        "50번 실험하고 가장 좋은 것을 보고하면, 그 점수는 실력이 아니라 "
        "50번 중 가장 운 좋았던 한 번이다."
    )


def test_test_집합은_마지막에_딱_한_번() -> None:
    usage = SplitUsage(
        train_sample_count=900,
        validation_sample_count=200,
        test_sample_count=200,
        test_evaluations=3,
    )
    findings = EvaluationProtocol().inspect(usage)
    finding = next(f for f in findings if f.code == "PROTOCOL_TEST_REUSED")

    assert finding.severity is Severity.CRITICAL
    report.block("test 를 세 번 봤다면", f"  {finding.message}")
    report.note(
        "두 번째부터 test 는 두 번째 validation 이다. "
        "그리고 이건 되돌릴 수 없다 — 새 데이터를 구하는 것 외에 방법이 없다."
    )


def test_평가_집합이_너무_작으면_숫자를_믿을_수_없다() -> None:
    thin = SplitUsage(
        train_sample_count=900, validation_sample_count=20, test_sample_count=15
    )
    findings = EvaluationProtocol().inspect(thin)

    assert "PROTOCOL_VALIDATION_TOO_SMALL" in codes(findings)
    assert "PROTOCOL_TEST_TOO_SMALL" in codes(findings)
    report.note(
        "15개로 낸 정확도는 한 건만 틀려도 6.7%p 가 움직인다. "
        "그 숫자로 모델 A 와 B 를 비교할 수는 없다."
    )


def test_학습기가_검증을_몇_번_봤는지_스스로_센다(trained) -> None:
    from application.model.support import load_run

    usage = load_run(trained.model.runs, trained.run_id).usage
    assert usage.validation_evaluations == 10  # epoch 마다 한 번
    assert usage.test_evaluations == 0

    report.note(
        "epoch 마다 검증했으니 10번이다. 이건 정상이다 — 조기 종료의 근거다. "
        "문제가 되는 것은 **그 점수를 보고 설정을 바꿔 다시 돌리는** 횟수다."
    )
