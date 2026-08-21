"""실습 2-1 — 깨끗한 데이터부터 의심하라.

    pytest -m lesson_2_1 -s

모듈 1을 통과한 데이터를 가져온다.
스키마도 맞고, 시간축도 깨끗하고, 라벨 정의도 서 있다. **READY 판정까지 받았다.**

그리고 품질을 재 본다.
"""

from __future__ import annotations

import pytest

from application.data.certify_dataset_readiness import CertifyDatasetReadinessCommand
from application.data.get_dataset import GetDatasetQuery
from tests.support import report
from tests.support import quality_scenario as qs

pytestmark = pytest.mark.lesson_2_1


def test_모듈1을_통과한_데이터로_시작한다(container, quality) -> None:
    report.section("실습 2-1 · 깨끗한 데이터부터 의심하라")

    qs.prepare_dataset(container, "dirty", quality.dirty)
    certificate = container.certify_dataset_readiness().execute(
        CertifyDatasetReadinessCommand(
            dataset_id="dirty", policy=qs.structural_readiness_policy()
        )
    )

    report.block("모듈 1 구조 검증", certificate.render())

    assert certificate.is_ready is True
    assert certificate.blocking == ()
    state = container.get_dataset().execute(GetDatasetQuery(dataset_id="dirty"))
    assert state.status == "READY"

    report.note(
        "스키마 ✓ 시간축 ✓ 신호 ✓ 라벨 정의 ✓ 분할 ✓ 학습 설계 ✓ — 전부 통과했다."
    )


def test_그런데_품질을_재면_이야기가_달라진다(
    container, quality_container, quality
) -> None:
    qs.prepare_dataset(container, "dirty", quality.dirty)
    qs.start(quality_container, "qa-dirty", "dirty")
    views = qs.measure_all(quality_container, "qa-dirty")

    report.block(
        "품질 여섯 축",
        "\n".join(
            f"  {name:<15} {view.score:6.1f} ({view.grade})  {view.verdict}"
            for name, view in views.items()
        ),
    )

    failed = [name for name, view in views.items() if not view.passed]
    assert len(failed) >= 4
    report.note(f"여섯 축 중 {len(failed)}개가 FAILED: {', '.join(failed)}")
    report.note(
        "'구조가 맞다'와 '내용이 깨끗하다'는 다른 질문이다. "
        "모듈 1은 앞의 질문만 물었다."
    )


def test_기준선_데이터는_품질도_통과한다(container, quality_container, quality) -> None:
    """검사 도구가 아무거나 잡아내는 것이 아님을 확인한다."""
    qs.prepare_dataset(container, "clean", quality.clean)
    qs.start(quality_container, "qa-clean", "clean")
    views = qs.measure_all(quality_container, "qa-clean")

    report.block(
        "같은 설비, 오염 없는 데이터",
        "\n".join(
            f"  {name:<15} {view.score:6.1f} ({view.grade})  {view.verdict}"
            for name, view in views.items()
        ),
    )
    assert all(view.passed for view in views.values())


def test_오염은_다른_검사의_결과까지_오염시킨다(container, quality) -> None:
    """대표성 검사가 엉뚱한 신호를 낸다.

    드리프트가 일어난 것이 아니다. 온도를 0.0 으로 채워 넣은 것이
    분포를 통째로 바꿔 놓았을 뿐이다.
    그래서 품질을 먼저 정리해야 다른 판단도 신뢰할 수 있다.
    """
    from application.data.analyze_representativeness import (
        AnalyzeRepresentativenessCommand,
    )
    from tests.support.scenario import power_source

    qs.prepare_dataset(container, "dirty", quality.dirty)
    view = container.analyze_representativeness().execute(
        AnalyzeRepresentativenessCommand(
            dataset_id="dirty",
            observed=power_source(quality.clean, "같은 설비 · 오염 없는 표본"),
        )
    )

    report.block("오염된 데이터를 기준으로 한 대표성 비교", view.render())

    assert view.inspection.verdict == "FAILED"
    assert view.worst_field == "voltage_v"
    report.note(
        "설비는 그대로다. 바뀐 것은 데이터를 다룬 방식뿐인데 "
        f"전압 PSI 가 {view.worst_psi:.2f} 로 튀었다. "
        "품질 문제를 드리프트로 오진하면 엉뚱한 대응을 하게 된다."
    )

    # 반대 방향의 함정: 온도에는 0.0 이 5% 나 채워져 있는데 PSI 는 조용하다.
    psi = {name: value for name, value, _ in view.field_psi}
    assert psi["temperature_c"] < 0.1
    report.note(
        f"반대로 온도 PSI 는 {psi['temperature_c']:.3f} 로 조용하다. "
        "분위수 구간 나누기는 한 점에 몰린 값(0.0)을 옆 구간에 흡수해 버린다. "
        "드리프트 지표로는 품질 문제를 찾을 수 없다 — 품질은 품질 도구로 봐야 한다."
    )
