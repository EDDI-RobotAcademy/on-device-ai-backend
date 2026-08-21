"""실습 5-7 — Data Drift를 직접 찾아라.

    pytest -m lesson_5_7 -s

실습 1-9 에서 이미 PSI 를 봤다. 그때는 **학습 전에** 물었다.
지금은 **학습 후에** 묻는다. 같은 계산, 다른 질문이다.

    학습 전   "이 데이터가 현실을 대표하는가?"
    학습 후   "현실이 학습 데이터에서 얼마나 멀어졌는가?"
"""

from __future__ import annotations

import pytest

from domain.operations.drift import DriftPolicy, DriftReport, FeatureDrift
from domain.operations.window import ObservationWindow
from domain.shared.inspection import Severity
from tests.support import report

pytestmark = pytest.mark.lesson_5_7


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def window(label: str = "w") -> ObservationWindow:
    return ObservationWindow(
        label=label,
        started_at="2026-05-23 00:00:00",
        ended_at="2026-05-23 07:59:59",
        sample_count=3000,
    )


def test_1일차는_학습_데이터와_같다(deployed) -> None:
    report.section("실습 5-7 · Data Drift를 직접 찾아라")

    first = deployed.reports[0]
    report.note(f"1일차 최대 PSI = {first.max_psi:.4f}")
    assert first.max_psi < 0.1
    report.note(
        "**이 숫자가 낮게 나오는 것이 먼저다.** "
        "1일차부터 PSI 가 튀면 드리프트를 재는 게 아니라 창을 잘못 자른 것이다."
    )
    report.note(
        "이 설비의 느린 부하 주기는 8시간이다. 4시간 창으로 자르면 주기의 절반만 담기고, "
        "드리프트가 없어도 PSI 1.5 가 나온다. **관측 창은 공정 주기의 배수여야 한다.**"
    )


def test_현실이_학습_데이터에서_멀어진다(deployed) -> None:
    values = [r.max_psi for r in deployed.reports]
    report.block(
        "창별 최대 PSI",
        "\n".join(
            f"  {r.window_label:<14}{r.max_psi:>9.4f}" for r in deployed.reports
        ),
    )
    assert values[0] < 0.1
    assert values[-1] > 5.0
    report.note(
        f"1일차 {values[0]:.4f} → 4일차 {values[-1]:.4f}. "
        "여름이 왔고, 설비 온도가 학습 때 본 적 없는 구간으로 들어갔다."
    )
    report.note(
        "**데이터는 고정돼 있는데 현실은 계속 움직인다.** 이것이 배포 후 가장 흔한 실패다."
    )


def test_본_적_없는_범위는_틀리는_게_아니라_정의되지_않는다(deployed) -> None:
    last = deployed.reports[-1]
    out_of_range = [f for f in last.findings if f.code == "OPS_INPUT_OUT_OF_RANGE"]
    report.block(
        "학습 때 본 적 없는 범위",
        "\n".join(f"  - {f.describe()}" for f in out_of_range),
    )

    assert out_of_range
    # DTO 를 거쳐 나온 소견이라 severity 는 Enum 이 아니라 문자열이다.
    assert all(f.severity == Severity.CRITICAL.value for f in out_of_range)
    report.note(
        "PSI 는 '분포가 다르다' 를 말하고, 범위 밖 비율은 "
        "'**아예 본 적 없다**' 를 말한다. 뒤쪽이 훨씬 심각하다."
    )
    report.note(
        "본 적 없는 구간에서 모델이 무슨 답을 할지는 아무도 모른다. "
        "틀리는 것이 아니라 정의되지 않는다."
    )


def test_여러_채널이_동시에_움직이면_공정이_바뀐_것이다(deployed) -> None:
    last = deployed.reports[-1]
    assert "OPS_MULTI_FEATURE_DRIFT" in {f.code for f in last.findings}
    finding = next(
        f for f in last.findings if f.code == "OPS_MULTI_FEATURE_DRIFT"
    )
    report.note(finding.describe())
    report.note(
        "센서 하나가 고장 나면 채널 하나만 움직인다. "
        "네 개가 같이 움직이면 **공정 자체가 바뀐 것**에 가깝다."
    )


def test_세_가지_드리프트_중_하나는_잴_수_없다() -> None:
    """**이 사실을 아는 것이 이 모듈의 핵심이다.**"""
    report.block(
        "드리프트 세 가지",
        "\n".join(
            [
                "  입력 드리프트   들어오는 값이 변했다          → 잰다 (실습 5-7)",
                "  예측 드리프트   나가는 답이 변했다            → 잰다 (실습 5-6)",
                "  개념 드리프트   입력과 정답의 관계가 변했다   → **못 잰다**",
            ]
        ),
    )
    report.note(
        "개념 드리프트를 재려면 정답이 있어야 한다. 현장에는 없다. "
        "그래서 앞의 둘로 **대신** 본다 — 대신 본다는 사실을 잊으면 안 된다."
    )


def test_PSI_기준은_통계가_아니라_재학습_비용이_정한다() -> None:
    report_ = DriftReport(
        window=window(),
        features=(
            FeatureDrift(
                field_name="temperature_c",
                psi=0.14,
                mean_shift_sigma=0.6,
                out_of_range_ratio=0.0,
            ),
        ),
    )
    strict = DriftPolicy(max_psi=0.1).inspect(report_)
    loose = DriftPolicy(max_psi=0.2).inspect(report_)

    assert "OPS_INPUT_DRIFT" in codes(strict)
    assert "OPS_INPUT_DRIFT_WATCH" in codes(loose)
    report.note(
        "PSI 0.14 는 기준 0.1 에서는 '조치', 0.2 에서는 '지켜보기' 다. "
        "**같은 숫자, 다른 판정.**"
    )
    report.note(
        "라벨링에 2주가 걸리는 현장이면 기준을 낮게 잡아 미리 시작해야 한다. "
        "이 숫자는 통계가 아니라 재학습 비용이 정한다."
    )


def test_학습_분포를_안_남겨_두면_잴_수_없다() -> None:
    findings = DriftPolicy().inspect(DriftReport(window=window(), features=()))
    assert "OPS_NO_DRIFT_BASELINE" in codes(findings)
    report.note(
        "드리프트는 **두 분포의 비교**다. 학습 때 분포를 남겨 두지 않으면 "
        "현장 데이터를 아무리 봐도 달라졌는지 알 수 없다."
    )
    report.note("그래서 배포할 때 그것을 함께 내보내야 한다 (실습 5-1).")
