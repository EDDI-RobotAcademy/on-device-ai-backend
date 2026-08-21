"""실습 5-11 — 재학습이 필요한 순간을 직접 정의하라.

    pytest -m lesson_5_11 -s

가장 흔한 답: **"정확도가 떨어지면 재학습한다."**

이 문장은 실행할 수 없다. 현장에 정답이 없어서 정확도를 못 재기 때문이다.
모듈 5 전체가 그 사실 위에 서 있었다.

실행 가능한 기준은 이렇게 생겼다.

    입력 분포가 변했고         (실습 5-7)
    그 상태가 지속됐고         (실습 5-4 — 한 번 튄 게 아니다)
    새 라벨이 충분히 모였다    (없으면 재학습해도 같은 것을 배운다)
"""

from __future__ import annotations

import pytest

from domain.operations.retraining import (
    LabelSupply,
    RetrainingPolicy,
    TriggerReason,
    Urgency,
)
from tests.support import operations_scenario as os5
from tests.support import report

pytestmark = pytest.mark.lesson_5_11


def test_지속된_입력_드리프트가_재학습_사유다(deployed) -> None:
    report.section("실습 5-11 · 재학습이 필요한 순간을 직접 정의하라")

    view = os5.decide_retraining(deployed.operations)
    report.block("재학습 판정", view.render())

    assert view.needed
    assert "INPUT_DRIFT" in view.reasons
    report.note(
        "근거는 '**지속된** 입력 드리프트' 다. 한 번 튄 것으로는 재학습하지 않는다."
    )


def test_한_번_튄_것은_재학습_사유가_아니다(deployed) -> None:
    """실습 5-4 의 구분이 여기서 실제로 쓰인다."""
    strict = os5.decide_retraining(
        deployed.operations,
        policy=RetrainingPolicy(sustained_windows=8),
    )
    report.block("연속 8창을 요구했을 때", strict.render())

    assert not strict.needed
    codes = {f.code for f in strict.findings}
    report.note(
        "4일치에 8창 연속은 없었다. 그래서 재학습 사유가 아니라고 판정한다."
    )
    report.note(
        "연속 창 수를 몇으로 둘지는 **재학습 비용이 정한다.** "
        "라벨링에 2주가 걸리면 짧게 잡아 미리 시작해야 한다."
    )


def test_재학습해야_하는데_지금은_할_수_없다() -> None:
    """**이것이 이 판정의 진짜 값어치다.**"""
    from domain.operations.health import HealthTimeline

    timeline = _drifting_timeline()
    decision = RetrainingPolicy(min_new_labels=5_000).decide(
        timeline,
        LabelSupply(
            total_records=34_000,
            labeled_records=4_118,
            labeled_since_deploy=4_118,
            minority_label_counts={"FAULT": 30, "OVERLOAD": 856, "NORMAL": 3232},
        ),
    )
    report.block("라벨이 모자랄 때", decision.render())

    assert decision.needed
    assert not decision.can_start
    assert decision.blockers
    report.note(
        "재학습이 필요하다는 것과 지금 시작할 수 있다는 것은 다르다. "
        "**막고 있는 것의 목록이 곧 할 일 목록이다.**"
    )


def test_소수_클래스_라벨이_없으면_그_클래스는_안_나아진다() -> None:
    decision = RetrainingPolicy(min_labels_per_class=100).decide(
        _drifting_timeline(),
        LabelSupply(
            total_records=34_000,
            labeled_records=4_118,
            labeled_since_deploy=4_118,
            minority_label_counts={"FAULT": 30, "OVERLOAD": 856, "NORMAL": 3232},
        ),
    )
    report.block("FAULT 라벨이 30건뿐일 때", decision.render())

    assert any("FAULT" in blocker for blocker in decision.blockers)
    report.note(
        "전체 라벨은 4,118건으로 충분하다. 그런데 FAULT 는 30건이다. "
        "**재학습해도 FAULT 는 나아지지 않는다.**"
    )
    report.note(
        "그러니 할 일은 '재학습'이 아니라 '**FAULT 구간을 찾아 라벨링**' 이다. "
        "그 구분이 몇 주를 아낀다."
    )


def test_정답이_있으면_가장_강한_근거가_된다(deployed) -> None:
    """현장 정답 12% 로 실제 정확도를 재 봤을 때."""
    view = os5.decide_retraining(
        deployed.operations,
        measured_accuracy=0.729,
        policy=RetrainingPolicy(measured_accuracy_floor=0.80),
    )
    report.block("정답 붙은 표본에서 잰 정확도가 0.729 일 때", view.render())

    assert "MEASURED_ACCURACY_DROP" in view.reasons
    assert view.urgency == Urgency.NOW.value
    report.note(
        "다른 근거들은 '**아마 나빠졌을 것**' 이다. 이것만 '**나빠졌다**' 이다."
    )
    report.note(
        "그래서 현장 라벨링에 사람을 붙이는 것이 결국 가장 싼 관측 장치다."
    )


def test_아무_일도_없으면_재학습하지_않는다() -> None:
    from domain.operations.health import HealthTimeline

    decision = RetrainingPolicy().decide(
        _calm_timeline(),
        LabelSupply(
            total_records=34_000, labeled_records=4_118, labeled_since_deploy=4_118
        ),
    )
    assert not decision.needed
    assert decision.urgency == Urgency.NONE
    report.note(
        "정기 재학습을 할 수는 있다. 그러나 그건 SCHEDULE 이라는 다른 이유다 — "
        "**드리프트가 없는데 드리프트를 이유로 대지 않는다.**"
    )


def test_관측_없이는_판단할_수_없다() -> None:
    from domain.operations.health import HealthTimeline
    from domain.shared.errors import InvariantViolation

    with pytest.raises(InvariantViolation) as caught:
        RetrainingPolicy().decide(
            HealthTimeline(), LabelSupply(total_records=0, labeled_records=0)
        )
    report.note(str(caught.value))
    report.note(
        "배포하고 아무것도 안 보고 있었다면 이 질문 자체가 성립하지 않는다."
    )


def test_순환은_여기서_닫힌다(operations_container, deployed) -> None:
    """모듈 5 의 마지막이자 모듈 1 로 돌아가는 문."""
    from infrastructure.monitoring.event_log import RecordingEventPublisher

    events = RecordingEventPublisher()
    operations_container.publisher = events

    windows = os5.windows(operations_container)
    for w in windows:
        os5.observe(operations_container, w)
    view = os5.decide_retraining(
        operations_container, watch_id=f"watch-{os5.DEPLOYMENT_ID}"
    )

    names = events.names()
    report.block(
        "재학습 판정이 남긴 사건",
        "\n".join(f"  {name}" for name in names if name == "RetrainingRequested"),
    )
    assert view.needed
    assert "RetrainingRequested" in names
    report.note(
        "이 사건을 받는 쪽이 모듈 1 의 데이터 수집이다."
    )
    report.note(
        "그리고 돌아갈 때 **무엇이 어떻게 변했는지**와 "
        "**어느 클래스의 라벨이 모자란지**를 들고 간다. "
        "그 두 가지가 다음 데이터 수집 계획이 된다."
    )
    report.block(
        "데이터 → 품질 → 모델 → 최적화 → 운영 → **다시 데이터**",
        "  다섯 모듈이 한 바퀴를 돈다.",
    )


# ---------------------------------------------------------------------------
# 손으로 만든 시간선 — 파일도 모델도 필요 없다
# ---------------------------------------------------------------------------
def _timeline(psi_values):  # noqa: ANN001, ANN202
    from domain.operations.drift import DriftReport, FeatureDrift
    from domain.operations.health import HealthReport, HealthTimeline
    from domain.operations.window import ObservationWindow

    reports = []
    for index, psi in enumerate(psi_values):
        window = ObservationWindow(
            label=f"w{index + 1:02d}",
            started_at=f"2026-05-{20 + index // 3:02d} {(index % 3) * 8:02d}:00:00",
            ended_at=f"2026-05-{20 + index // 3:02d} {(index % 3) * 8 + 7:02d}:59:59",
            sample_count=2_880,
        )
        reports.append(
            HealthReport(
                window=window,
                deployment_version=1,
                drift=DriftReport(
                    window=window,
                    features=(
                        FeatureDrift(
                            field_name="temperature_c",
                            psi=psi,
                            mean_shift_sigma=psi,
                            out_of_range_ratio=0.0,
                        ),
                    ),
                ),
            )
        )
    return HealthTimeline(reports=tuple(reports))


def _drifting_timeline():  # noqa: ANN202
    return _timeline([0.01, 0.01, 0.01, 0.01, 7.3, 0.01, 4.6, 4.6, 3.3, 9.2, 9.5, 9.4])


def _calm_timeline():  # noqa: ANN202
    return _timeline([0.01] * 12)
