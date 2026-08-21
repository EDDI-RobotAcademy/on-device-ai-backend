"""실습 3-12 — 데이터 구조를 바꾸면 모델이 달라진다.

    pytest -m lesson_3_12 -s

실습 1-7 에서 "AI가 먹을 수 있는 데이터로 다시 설계하라"를 했다.
그때 정한 창 길이 30(5분)은 **가정**이었다. 여기서 그 가정을 검증한다.

같은 CSV, 같은 모델, 같은 학습 설정. 창 길이만 15 / 30 / 60 / 120 으로 바꾼다.
그러면 세 가지가 한꺼번에 움직인다.

    표본 수     864개 → 108개 (창을 8배로 하면 표본은 1/8)
    라벨 구성   30% 규칙에 걸리는 창이 달라진다
    지표 신뢰도 17개로 잰 1.000 은 1.000 이 아니다

**"정확도가 올랐다"가 아니라 "무엇으로 잰 정확도인가"를 먼저 봐야 한다.**
"""

from __future__ import annotations

import pytest

from application.model.compare_experiments import (
    CompareExperimentsCommand,
    TrialRequest,
)
from domain.model.experiment import ExperimentPolicy
from tests.support import report

pytestmark = pytest.mark.lesson_3_12


def _board(structure_lab, **kwargs):  # noqa: ANN001, ANN003
    return structure_lab.model.compare_experiments().execute(
        CompareExperimentsCommand(
            name="창 길이를 바꿔 보았다",
            trials=tuple(
                TrialRequest(
                    run_id=f"run-win{length}",
                    label=f"창 {length}",
                    knobs={"window": str(length)},
                )
                for length in structure_lab.windows
            ),
            **kwargs,
        )
    )


def test_창_길이를_바꾸면_결과가_달라진다(structure_lab) -> None:
    report.section("실습 3-12 · 데이터 구조를 바꾸면 모델이 달라진다")

    view = _board(structure_lab)
    report.block("비교표", view.render())

    assert len(view.trials) == 4
    assert view.spread > 0.0
    report.note(
        "구조를 바꿨을 뿐인데 macro F1 이 움직인다. "
        "**모델을 손대지 않고도 결과는 달라진다** — "
        "그래서 데이터 설계는 모델 설계보다 먼저 온다."
    )


def test_창을_길게_잡을수록_표본이_줄어든다(structure_lab) -> None:
    view = _board(structure_lab)
    counts = {t.label: t.evaluated_samples for t in view.trials}

    report.block(
        "창 길이 → 평가 표본 수",
        "\n".join(f"  {label:<10} {count:>5}개" for label, count in counts.items()),
    )

    assert counts["창 15"] > counts["창 30"] > counts["창 60"] > counts["창 120"]
    report.note(
        "원본 12,960행은 그대로다. 창을 8배로 늘리면 표본은 1/8 이 된다. "
        "**데이터를 더 모은 것이 아니라 잘게 쪼갠 것이었다.**"
    )


def test_표본이_적으면_1점0은_1점0이_아니다(structure_lab) -> None:
    """이 실습의 본론."""
    view = _board(structure_lab)
    codes = [f.code for f in view.findings]

    report.block(
        "비교를 믿어도 되는가",
        "\n".join(f"  {f.describe()}" for f in view.findings),
    )

    assert "EXP_TOO_FEW_EVALUATED" in codes
    worst = min(view.trials, key=lambda t: t.evaluated_samples)
    assert worst.accuracy >= 0.99
    report.note(
        f"'{worst.label}' 은 정확도 {worst.accuracy:.3f} 인데 "
        f"표본이 {worst.evaluated_samples}개다. "
        "**한 개만 틀렸어도 숫자가 크게 흔들린다.** "
        "비교표에서 이 칸을 지우면 잘못된 창 길이를 고르게 된다."
    )


def test_가장_긴_창이_이겼다고_고르면_안_된다(structure_lab) -> None:
    view = _board(structure_lab)
    best = view.trial_of(view.best_label)
    blocking = [f for f in view.findings if f.severity == "CRITICAL"]

    report.block(
        "1등의 근거",
        f"  1등        : {view.best_label} (macro F1 {best.macro_f1:.3f})\n"
        f"  2등과 차이 : {view.gap_to_runner_up:.4f}\n"
        f"  막는 소견  : {len(blocking)}건",
    )

    assert blocking, "표본이 모자란 시행이 있으면 그대로 고르면 안 된다"
    report.note(
        "1등이 있다고 고를 수 있는 것이 아니다. "
        "**막는 소견이 남아 있으면 그 표는 아직 결론이 아니다.** "
        "창을 60·120 으로 쓰려면 데이터를 더 모아야 한다."
    )


def test_흔들림_폭_안의_차이는_차이가_아니다(structure_lab) -> None:
    view = _board(structure_lab, policy=ExperimentPolicy(noise_band=0.02))
    codes = [f.code for f in view.findings]

    assert "EXP_WITHIN_NOISE" in codes
    report.note(
        f"1등과 2등의 차이가 {view.gap_to_runner_up:.4f} 다. "
        "**시드만 바꿔도 이 정도는 흔들린다.** "
        "이 숫자로 '더 낫다'고 쓰면 다음 사람이 그 결론을 믿고 두 달을 쓴다."
    )
