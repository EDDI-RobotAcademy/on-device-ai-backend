"""실습 4-11 — 구조를 줄이는 것과 숫자를 줄이는 것은 다르다.

    pytest -m lesson_4_11 -s

실습 4-5 ~ 4-7 은 **숫자를 좁혔다.** FP32 → FP16 → INT8.
구조는 손대지 않았다. 층도 채널도 그대로고, **곱셈 횟수도 그대로**다.

경량화에는 축이 하나 더 있다. 구조 자체를 줄이는 것.
그리고 두 축의 성질이 완전히 다르다.

    양자화     재학습 없음. 곱셈 횟수 그대로. 파일만 작아진다.
    구조 축소  **재학습 필수.** 곱셈 횟수가 실제로 준다.

여기서 확인할 함정이 두 개 있다.

    1. 비구조적 가지치기는 **파일도 속도도 안 줄인다** — 0도 저장되고 0도 곱해진다.
    2. 가지치기 뒤에 미세조정하면 **0이 다시 채워진다** — 가지치기가 없던 일이 된다.
"""

from __future__ import annotations

import pytest

from domain.optimization.structural import (
    ReductionKind,
    StructuralOutcome,
    StructuralPolicy,
    StructuralReduction,
)
from domain.shared.errors import InvariantViolation
from tests.support import report

pytestmark = pytest.mark.lesson_4_11


def test_구조를_줄이면_곱셈_횟수가_실제로_준다(reduced) -> None:
    report.section("실습 4-11 · 구조를 줄이는 것과 숫자를 줄이는 것은 다르다")

    report.block("구조 축소 비교", reduced.render())

    width = reduced.outcome_of("폭 절반 · 재학습")
    assert width.mac_reduction > 0.5
    assert width.size_reduction > 0.4
    report.note(
        f"채널을 절반으로 줄이니 곱셈 횟수가 {width.mac_reduction:.0%} 줄었다. "
        "**양자화로는 이 칸이 절대 안 움직인다** — "
        "INT8 로 바꿔도 곱셈 횟수는 그대로다."
    )


def test_구조를_줄이고_재학습을_안_하면_모델이_무너진다(reduced) -> None:
    """이 실습의 본론 (1)."""
    naive = reduced.outcome_of("폭 절반 · 재학습 없음")
    tuned = reduced.outcome_of("폭 절반 · 재학습")

    report.block(
        "같은 축소, 재학습 여부만 다르다",
        f"  재학습 없음 : 정확도 {naive.accuracy_after:.3f}  ({naive.verdict})\n"
        f"  재학습      : 정확도 {tuned.accuracy_after:.3f}  ({tuned.verdict})\n"
        f"  둘 다 MAC {naive.mac_reduction:.0%}↓ / 크기 {naive.size_reduction:.0%}↓",
    )

    assert naive.accuracy_after < 0.5
    assert tuned.accuracy_after > naive.accuracy_after + 0.4
    assert naive.verdict == "BLOCKED"
    report.note(
        "모양이 달라졌으니 **배운 가중치를 물려받을 수 없다.** "
        "새 구조는 무작위 초기값에서 시작한다 — 그래서 미세조정이 절차의 일부다. "
        "양자화와 여기서 갈린다."
    )


def test_비구조적_가지치기는_장부상의_경량화다(reduced) -> None:
    """이 실습의 본론 (2)."""
    pruned = reduced.outcome_of("가지치기 50% · 그대로")

    report.block(
        "가중치의 절반을 0으로 만들었다",
        f"  파라미터 수   {pruned.parameter_count_after:,} (그대로)\n"
        f"  0 아닌 것     {pruned.nonzero_parameter_count:,}\n"
        f"  희소도        {pruned.sparsity:.0%}\n"
        f"  파일 크기     {pruned.size_reduction:+.1%}\n"
        f"  곱셈 횟수     {pruned.mac_reduction:+.1%}",
    )

    assert pruned.sparsity > 0.4
    assert pruned.size_reduction < 0.05
    assert pruned.mac_reduction < 0.01
    assert "STRUCT_SPARSITY_NOT_REALIZED" in [f.code for f in reduced.findings]
    report.note(
        "가중치의 49%가 0이 되었다. **그런데 파일도 속도도 그대로다.** "
        "0도 4바이트로 저장되고, 0도 곱셈 한 번을 쓴다. "
        "희소 연산을 지원하는 런타임이 있어야 비로소 이득이 생긴다 — "
        "이 사실을 모르면 '50% 가지치기 했는데 왜 그대로죠?'에서 하루를 쓴다."
    )


def test_미세조정이_가지친_자리를_다시_채운다(reduced) -> None:
    """이 실습에서 가장 놓치기 쉬운 한 줄."""
    plain = reduced.outcome_of("가지치기 50% · 그대로")
    tuned = reduced.outcome_of("가지치기 50% · 미세조정")

    report.block(
        "가지치기 뒤에 다시 학습하면",
        f"  그대로     : 희소도 {plain.sparsity:.0%}  정확도 {plain.accuracy_after:.3f}\n"
        f"  미세조정   : 희소도 {tuned.sparsity:.0%}  정확도 {tuned.accuracy_after:.3f}",
    )

    assert tuned.accuracy_after > plain.accuracy_after
    assert tuned.sparsity < plain.sparsity / 2
    assert "STRUCT_PRUNE_UNDONE" in [f.code for f in reduced.findings]
    report.note(
        "정확도는 회복됐다. **그런데 0이 6%만 남았다.** "
        "옵티마이저가 0인 자리에도 기울기를 흘려서 다시 채운 것이다. "
        "마스크로 붙잡아 두지 않으면 **가지치기가 없던 일이 된다.**"
    )


def test_채널을_0으로_만드는_것과_떼어내는_것은_다르다(reduced) -> None:
    channel = reduced.outcome_of("채널 가지치기 50%")

    report.block(
        "채널 단위 가지치기",
        f"  희소도      {channel.sparsity:.0%}\n"
        f"  곱셈 횟수   {channel.mac_reduction:+.1%}\n"
        f"  파일 크기   {channel.size_reduction:+.1%}",
    )

    assert channel.mac_reduction < 0.01
    report.note(
        "채널을 통째로 0으로 만들어도 **그 채널은 여전히 거기 있다.** "
        "실제로 떼어 내서 더 작은 층으로 다시 조립해야 곱셈이 준다 — "
        "그건 결국 '폭 줄이기'와 같은 일이고, 그래서 재학습이 따라온다."
    )


def test_쓸_수_있는_것은_하나뿐이다(reduced) -> None:
    report.block(
        "판정",
        "\n".join(
            f"  {o.label:<24}{o.verdict:<9}"
            f"MAC {o.mac_reduction:>6.1%}↓  정확도 {o.accuracy_after:.3f}"
            for o in reduced.outcomes
        )
        + f"\n\n  쓸 수 있는 것: {reduced.usable}",
    )

    assert "폭 절반 · 재학습" in reduced.usable
    report.note(
        "다섯 가지를 해 봤고 하나가 남았다. "
        "**'경량화 기법'을 아는 것과 '이 모델에 무엇이 통하는지' 아는 것은 다르다.** "
        "표를 만들지 않으면 통하지 않는 것을 계속 시도하게 된다."
    )


def test_양자화와_구조_축소는_같이_쓸_수_있다(reduced, optimized) -> None:
    """두 축은 곱해진다 — 그것이 이 둘을 나눠 본 이유다."""
    width = reduced.outcome_of("폭 절반 · 재학습")
    report.block(
        "두 축",
        f"  구조 축소  : 파라미터 {width.parameter_count_before:,} → "
        f"{width.parameter_count_after:,}\n"
        f"  양자화     : 남은 파라미터를 4바이트 → 1바이트로\n"
        f"  둘 다 하면 : 대략 {width.size_reduction:.0%} × 추가 75% 감소",
    )
    assert width.parameter_count_after < width.parameter_count_before
    report.note(
        "구조 축소로 **개수**를 줄이고, 양자화로 **하나의 크기**를 줄인다. "
        "순서는 구조 축소가 먼저다 — 재학습이 필요하기 때문이다."
    )


def test_줄이는_비율은_0과_1_사이여야_한다() -> None:
    with pytest.raises(InvariantViolation, match="0 초과 1 미만"):
        StructuralReduction(ReductionKind.WIDTH, 1.0)


def test_연산량이_안_줄면_경량화했다고_말할_수_없다() -> None:
    outcome = StructuralOutcome(
        reduction=StructuralReduction(ReductionKind.WIDTH, 0.1, fine_tuned=True),
        label="폭 10%",
        parameter_count_before=1000,
        parameter_count_after=980,
        nonzero_parameter_count=980,
        mac_count_before=100_000,
        mac_count_after=98_000,
        size_bytes_before=4000,
        size_bytes_after=3920,
        accuracy_before=0.95,
        accuracy_after=0.95,
    )
    findings = StructuralPolicy().inspect(outcome)

    report.block("소견", "\n".join(f"  {f.describe()}" for f in findings))
    assert any(f.code == "STRUCT_MAC_UNCHANGED" for f in findings)
    report.note(
        "2% 줄이려고 재학습 절차를 하나 더 만드는 것은 남는 장사가 아니다. "
        "**줄인 값보다 유지 비용이 크면 안 하는 것이 맞다.**"
    )
