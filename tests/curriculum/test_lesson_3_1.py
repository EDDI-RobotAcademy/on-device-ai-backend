"""실습 3-1 — 데이터를 처음으로 AI에게 먹여보자.

    pytest -m lesson_3_1 -s

모델은 CSV 를 먹지 않는다. **고정된 모양을 가진 숫자 덩어리**를 먹는다.
그리고 그 덩어리를 만들 수 있는지는, 학습을 시작하기 전에 확인해야 한다.
"""

from __future__ import annotations

import pytest

from domain.model.tensor_spec import BatchSpec, TensorLayout, TensorSpec
from domain.shared.errors import IllegalStateTransition
from tests.support import model_scenario as ms
from tests.support import report

pytestmark = pytest.mark.lesson_3_1


def test_CSV_가_텐서가_되는_순간(trained) -> None:
    report.section("실습 3-1 · 데이터를 처음으로 AI에게 먹여보자")

    view = trained.preparation
    report.block("학습 준비", view.render())

    assert view.input_shape == (30, 6)
    assert view.batch_shape == (32, 30, 6)

    train = next(s for s in view.summaries if s.split == "train")
    assert train.sample_shape == (30, 6)
    assert train.sample_count > 800
    report.note(
        "12,960행짜리 CSV 가 (30, 6) 짜리 표본 1,294개가 되었다. "
        "이 모양이 앞으로 모든 것을 결정한다."
    )


def test_정규화가_실제로_적용되었는지_숫자로_확인한다(trained) -> None:
    """실습 1-7 에서 train 분할로 뽑아 둔 통계를 그대로 쓴다."""
    train = next(s for s in trained.preparation.summaries if s.split == "train")

    assert train.feature_mean == pytest.approx(0.0, abs=0.05)
    assert train.feature_std == pytest.approx(1.0, abs=0.05)
    report.note(
        f"train 평균 {train.feature_mean:.3f} / 표준편차 {train.feature_std:.3f} — "
        "정규화가 걸렸다는 증거다."
    )

    test = next(s for s in trained.preparation.summaries if s.split == "test")
    assert abs(test.feature_mean) > 0.01
    report.note(
        f"test 평균은 {test.feature_mean:.3f} 로 0 이 아니다. "
        "**당연하다** — 통계를 train 에서만 뽑았기 때문이다. "
        "test 평균이 정확히 0 이면 그게 누수다."
    )


def test_NaN_이_하나라도_있으면_첫_배치에서_무너진다(trained) -> None:
    for summary in trained.preparation.summaries:
        assert summary.nan_count == 0
    report.note("NaN 개수 0. 이걸 확인하지 않으면 학습 3분 뒤에 loss=nan 을 보게 된다.")


def test_배치_크기는_메모리를_결정한다() -> None:
    """모듈 4(최적화)에서 이 계산이 그대로 다시 쓰인다."""
    spec = TensorSpec(shape=(30, 6), layout=TensorLayout.TIME_FIRST)
    small = BatchSpec(sample=spec, batch_size=32)
    large = BatchSpec(sample=spec, batch_size=256)

    report.block(
        "배치 크기와 메모리",
        f"  표본 하나  : {spec.element_count}개 원소 × 4바이트 = {spec.bytes_per_sample}바이트\n"
        f"  배치 32    : {small.shape}  {small.bytes_per_batch / 1024:.1f} KiB\n"
        f"  배치 256   : {large.shape}  {large.bytes_per_batch / 1024:.1f} KiB",
    )
    assert large.bytes_per_batch == small.bytes_per_batch * 8
    assert small.batch_count(1000) == 32  # 마지막 배치는 24개짜리
    assert BatchSpec(sample=spec, batch_size=32, drop_last=True).batch_count(1000) == 31


def test_모양이_안_맞으면_준비_단계에서_막힌다(model_container, container, quality_container, model_data) -> None:
    """학습을 3분 돌린 뒤가 아니라, 시작 전에 안다."""
    from domain.model.errors import ShapeMismatch

    ms.pass_both_gates(
        container,
        quality_container,
        dataset_id="mt",
        assessment_id="qa-mt",
        path=model_data.train,
    )
    wrong = ms.cnn_architecture()
    wrong = type(wrong)(
        kind=wrong.kind,
        input_spec=TensorSpec(shape=(60, 6), layout=TensorLayout.TIME_FIRST),
        class_count=3,
        hidden_channels=wrong.hidden_channels,
    )
    with pytest.raises(ShapeMismatch, match="창 길이"):
        ms.prepare(
            model_container,
            run_id="bad",
            dataset_id="mt",
            assessment_id="qa-mt",
            architecture=wrong,
        )
    report.note("창은 30인데 모델 입력은 60. 학습은 돌아가고 배포할 때 터진다 — 그걸 여기서 막는다.")


def test_게이트를_통과하지_않은_데이터로는_시작조차_못_한다(
    model_container, container, quality_container, model_data
) -> None:
    """모듈 1과 2가 존재한 이유가 이 한 줄이다."""
    from tests.support import quality_scenario as qs

    # 모듈 1의 검사는 다 돌렸지만 **판정을 받지 않았고**, 품질 평가도 하지 않았다.
    qs.prepare_dataset(container, "ungated", model_data.train)

    with pytest.raises(IllegalStateTransition, match="게이트를 통과하지 않은") as excinfo:
        ms.prepare(
            model_container,
            run_id="ungated-run",
            dataset_id="ungated",
            assessment_id=None,
        )
    assert "모듈 1" in str(excinfo.value)
    assert "모듈 2" in str(excinfo.value)
    report.note(
        "데이터가 준비된 것과 **승인된 것**은 다르다. "
        "학습은 승인된 데이터로만 시작한다."
    )
