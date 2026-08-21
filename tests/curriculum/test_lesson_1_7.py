"""실습 1-7 — AI가 먹을 수 있는 데이터로 다시 설계하라.

    pytest -m lesson_1_7 -s

CSV 한 줄은 모델의 입력이 아니다. 모델의 입력은 고정된 모양을 가진 텐서다.
여기서 정하는 것은 값이 아니라 계약이다. 이 계약은 학습·최적화·배포에서 계속 쓰인다.
"""

from __future__ import annotations

import pytest

from application.data.design_training_data import DesignTrainingDataCommand
from application.data.partition_dataset import PartitionDatasetCommand
from application.data.support import load_dataset
from application.shared.errors import UnsupportedOperation
from domain.data.partition import PartitionPlan, SplitRatio, SplitStrategy
from domain.data.time_axis import SamplingInterval
from domain.data.training_spec import (
    NormalizationMethod,
    NormalizationSpec,
    TrainingDataSpec,
    WindowSpec,
)
from domain.shared.errors import InvariantViolation
from tests.support import report
from tests.support.scenario import (
    FEATURE_FIELDS,
    LABEL_FIELD,
    SAMPLE_INTERVAL_SECONDS,
    TIME_FIELD,
    declare_schema,
    define_labels,
    profile,
    register,
)

pytestmark = pytest.mark.lesson_1_7


def prepare(container, power) -> None:  # noqa: ANN001
    register(container, "curated", power.curated)
    profile(container, "curated")
    declare_schema(container, "curated")
    define_labels(container, "curated")


def spec_for(container, **overrides):  # noqa: ANN001, ANN201
    dataset = load_dataset(container.repository, "curated")
    base: dict[str, object] = dict(
        schema=dataset.schema,
        feature_fields=FEATURE_FIELDS,
        label_field=LABEL_FIELD,
        window=WindowSpec(
            length=30, stride=30, interval=SamplingInterval(SAMPLE_INTERVAL_SECONDS)
        ),
    )
    base.update(overrides)
    return TrainingDataSpec(**base)  # type: ignore[arg-type]


def test_입력_모양은_창_설계에서_결정된다(container, power) -> None:
    report.section("실습 1-7 · AI가 먹을 수 있는 데이터로 다시 설계하라")

    prepare(container, power)
    view = container.design_training_data().execute(
        DesignTrainingDataCommand(dataset_id="curated", spec=spec_for(container))
    )
    report.block("설계 결과", view.render())

    assert view.input_shape == (30, 6)  # 30 표본 × 6 채널
    assert view.window_seconds == pytest.approx(300.0)
    report.note(
        "한 표본이 현장 시간 300초를 본다. "
        "이 숫자는 '이상 징후가 몇 초 만에 드러나는가'에서 나와야 한다."
    )


def test_넣으면_안_되는_것을_설계_단계에서_막는다(container, power) -> None:
    prepare(container, power)

    with pytest.raises(InvariantViolation, match="묶음을 외운다"):
        spec_for(container, feature_fields=("batch_id",) + FEATURE_FIELDS)

    with pytest.raises(InvariantViolation, match="정확도 100%"):
        spec_for(container, feature_fields=(LABEL_FIELD,) + FEATURE_FIELDS)

    with pytest.raises(InvariantViolation, match="배포 즉시 분포가 벗어난다"):
        spec_for(container, feature_fields=(TIME_FIELD,) + FEATURE_FIELDS)

    report.note(
        "이 셋은 전부 검증 점수를 올린다. 그리고 전부 현장에서 무너진다."
    )


def test_정규화_없이_설계하면_경고가_남는다(container, power) -> None:
    prepare(container, power)
    view = container.design_training_data().execute(
        DesignTrainingDataCommand(dataset_id="curated", spec=spec_for(container))
    )
    codes = {f.code for f in view.inspection.findings}
    assert "SPEC_NO_NORMALIZATION" in codes
    report.note("전압 380, 전류 400, 온도 26 — 단위가 다른 축을 그대로 넣으면 큰 값이 학습을 지배한다.")


def test_정규화_통계는_분할_없이_뽑을_수_없다(container, power) -> None:
    """전체 데이터로 평균을 내는 선택지를 아예 주지 않는다."""
    prepare(container, power)

    with pytest.raises(UnsupportedOperation, match="분할이 없다"):
        container.design_training_data().execute(
            DesignTrainingDataCommand(
                dataset_id="curated",
                spec=spec_for(
                    container,
                    normalization=NormalizationSpec(
                        method=NormalizationMethod.ZSCORE
                    ),
                ),
                fit_normalization=True,
            )
        )
    report.note("정규화 통계는 train 에서만 나온다. 그러려면 분할이 먼저다.")


def test_분할_뒤에_train_에서만_통계를_뽑는다(container, power) -> None:
    prepare(container, power)
    container.partition_dataset().execute(
        PartitionDatasetCommand(
            dataset_id="curated",
            plan=PartitionPlan(
                strategy=SplitStrategy.TIME_ORDERED,
                ratio=SplitRatio.of(0.7, 0.15, 0.15),
                time_field=TIME_FIELD,
            ),
        )
    )
    view = container.design_training_data().execute(
        DesignTrainingDataCommand(
            dataset_id="curated",
            spec=spec_for(
                container,
                normalization=NormalizationSpec(method=NormalizationMethod.ZSCORE),
            ),
            fit_normalization=True,
        )
    )
    report.block("정규화 통계를 채운 설계", view.render())

    assert view.normalization_method == "ZSCORE"
    assert view.normalization_fitted_on == "train"
    assert view.inspection.verdict == "PASSED"

    dataset = load_dataset(container.repository, "curated")
    statistics = dataset.training_spec.normalization.statistics
    assert set(statistics) == set(FEATURE_FIELDS)

    report.block(
        "train 구간에서 계산한 (평균, 표준편차)",
        "\n".join(
            f"  {name:<22} mean={mean:10.3f}  std={std:9.3f}"
            for name, (mean, std) in sorted(statistics.items())
        ),
    )


def test_train_이_아닌_출처를_적으면_객체가_거부한다() -> None:
    """이 규칙은 Use Case 가 아니라 Value Object 가 지킨다."""
    with pytest.raises(InvariantViolation, match="데이터 누수"):
        NormalizationSpec(
            method=NormalizationMethod.ZSCORE,
            fitted_on="all",
            statistics={"voltage_v": (380.0, 2.2)},
        )
