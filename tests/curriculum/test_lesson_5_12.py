"""실습 5-12 — 디바이스에서 도는 추론 파이프라인을 완성하라.

    pytest -m lesson_5_12 -s

지금까지 모듈 5 는 **로그를 읽는 쪽**만 봤다.
그 로그를 만드는 쪽, 디바이스 안에서 실제로 도는 순서는 다섯 단계다.

    입력 수집 → 전처리 → 추론 → 후처리 → 알람/저장

그리고 이 실습의 결론은 하나다.

    **추론 성공률이 100%여도 끝까지 간 것은 82%일 수 있다.**

빠지는 곳은 대개 추론이 아니다. 그런데 사람들은 추론만 본다.
"""

from __future__ import annotations

import pytest

from domain.operations.pipeline import (
    PipelineContract,
    PipelinePolicy,
    PipelineRun,
    PipelineStage,
    StageOutcome,
)
from domain.shared.errors import InvariantViolation
from tests.support import report

pytestmark = pytest.mark.lesson_5_12


def test_다섯_단계를_순서대로_돌린다(device_pipeline) -> None:
    report.section("실습 5-12 · 디바이스에서 도는 추론 파이프라인을 완성하라")

    run = device_pipeline.run
    report.block("파이프라인", run.render())

    assert [s.stage for s in run.stages] == list(PipelineStage)
    report.note(
        "모듈 4 가 고른 그 결과물을, 모듈 5 의 4일치 현장 신호에 **실제로** 돌렸다. "
        "각 단계의 숫자는 세어서 나온 것이다."
    )


def test_추론은_다_성공했는데_답은_그만큼_안_나온다(device_pipeline) -> None:
    """이 실습의 본론."""
    run = device_pipeline.run
    infer = run.stage_of(PipelineStage.INFER)

    report.block(
        "어디서 빠지는가",
        f"  추론 성공률   {1 - infer.drop_ratio:.1%}\n"
        f"  끝까지 간 비율 {run.end_to_end_ratio:.1%}\n"
        f"  차이의 출처   전처리 {run.stage_of(PipelineStage.PREPROCESS).dropped:,}건, "
        f"후처리 {run.stage_of(PipelineStage.POSTPROCESS).dropped:,}건",
    )

    assert infer.drop_ratio == 0.0
    assert run.end_to_end_ratio < 0.9
    report.note(
        "**'모델은 잘 돕니다'가 맞는 말이면서 동시에 쓸모없는 말인 이유다.** "
        "현장에서 사라진 판단은 추론이 아니라 그 앞뒤에서 사라졌다."
    )


def test_창이_배치_경계를_넘으면_판단하지_않는다(device_pipeline) -> None:
    stage = device_pipeline.run.stage_of(PipelineStage.PREPROCESS)

    report.block(
        "전처리에서 빠지는 것",
        f"  판단 기회 {stage.attempted:,}건 → {stage.succeeded:,}건\n"
        + "\n".join(f"  {k} {v:,}건" for k, v in stage.reason_counts.items()),
    )

    assert stage.dropped > 0
    assert "배치 경계를 넘는 창" in stage.reason_counts
    report.note(
        "배치가 바뀌면 제품이 바뀐다. 한 창에 두 제품이 섞이면 "
        "모델은 학습 때 본 적 없는 조합을 본다. "
        "**에러는 안 난다** — 그래서 여기서 세어 두지 않으면 아무도 모른다."
    )


def test_이유를_안_적으면_고칠_수_없다(device_pipeline) -> None:
    run = device_pipeline.run
    with_reasons = [s for s in run.stages if s.dropped and s.reason_counts]

    report.block(
        "빠진 것마다 이유가 붙어 있다",
        "\n".join(
            f"  {s.stage.value:<12}"
            + ", ".join(f"{k} {v:,}" for k, v in s.reason_counts.items())
            for s in with_reasons
        ),
    )

    assert with_reasons
    report.note(
        "'1,719건 실패'로는 아무것도 못 한다. "
        "**'배치 경계를 넘는 창 1,719건'이어야 고칠 수 있다.**"
    )


def test_계약이_어긋나면_아무_에러_없이_틀린다(device_pipeline) -> None:
    """실습 5-1 에서 만난 그 사고를 배포 전에 잡는 장치."""
    trained = device_pipeline.contract
    field = PipelineContract(
        input_shape=trained.input_shape,
        sample_interval_seconds=30.0,  # 학습은 10초였다
        feature_fields=trained.feature_fields,
        normalization=dict(trained.normalization),
        class_labels=trained.class_labels,
    )

    gaps = field.differences_from(trained)
    report.block("계약 대조", "\n".join(f"  {g}" for g in gaps))

    run = PipelineRun(
        device_id="DEV-99",
        contract=field,
        stages=device_pipeline.run.stages,
    )
    findings = PipelinePolicy().inspect(run, trained_contract=trained)

    assert any(f.code == "PIPE_CONTRACT_MISMATCH" and f.is_blocking for f in findings)
    report.note(
        "표본 간격이 10초에서 30초로 바뀌었다. **모양은 그대로다.** "
        "그래서 학습도 배포도 에러 없이 돌아간다 — "
        "그리고 같은 30표본이 5분이 아니라 15분을 덮는다 (실습 5-1)."
    )


def test_추론_뒤에_전처리를_하는_파이프라인은_없다() -> None:
    contract = PipelineContract(input_shape=(30, 6), sample_interval_seconds=10.0)
    with pytest.raises(InvariantViolation, match="순서대로가 아니다"):
        PipelineRun(
            device_id="DEV-01",
            contract=contract,
            stages=(
                StageOutcome(stage=PipelineStage.INFER, attempted=10, succeeded=10),
                StageOutcome(
                    stage=PipelineStage.PREPROCESS, attempted=10, succeeded=10
                ),
            ),
        )


def test_성공이_시도보다_많을_수_없다() -> None:
    with pytest.raises(InvariantViolation, match="성공이 시도보다"):
        StageOutcome(stage=PipelineStage.INFER, attempted=5, succeeded=6)


def test_표본_간격은_0보다_커야_한다() -> None:
    with pytest.raises(InvariantViolation, match="0보다 커야"):
        PipelineContract(input_shape=(30, 6), sample_interval_seconds=0.0)
