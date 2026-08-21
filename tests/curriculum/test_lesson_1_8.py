"""실습 1-8 — 데이터를 쪼개야 진짜 문제가 보인다.

    pytest -m lesson_1_8 -s

분할은 나누기가 아니라 시험 문제를 격리하는 일이다.
여기서 학생은 두 가지를 본다.
    1. 무작위 분할이 시계열에서 실제로 몇 초를 새는지 (숫자로)
    2. 그래서 Domain 이 왜 그 계획 자체를 거부하는지
"""

from __future__ import annotations

import pytest

from application.data.partition_dataset import PartitionDatasetCommand
from domain.data.partition import PartitionPlan, SplitRatio, SplitStrategy
from domain.shared.errors import InvariantViolation
from infrastructure.analysis.pandas_partition_engine import PandasPartitionEngine
from tests.support import report
from tests.support.scenario import (
    GROUP_FIELD,
    LABEL_FIELD,
    TIME_FIELD,
    declare_schema,
    define_labels,
    power_schema,
    power_source,
    profile,
    register,
)

pytestmark = pytest.mark.lesson_1_8

RATIO = SplitRatio.of(0.7, 0.15, 0.15)


def prepare(container, power) -> None:  # noqa: ANN001
    register(container, "curated", power.curated)
    profile(container, "curated")
    declare_schema(container, "curated")
    define_labels(container, "curated")


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.inspection.findings}


def test_무작위_분할이_실제로_얼마나_새는지_직접_재본다(power) -> None:
    """Domain 을 거치지 않고 측정기만 직접 부른다. 숫자를 보기 위해서다."""
    report.section("실습 1-8 · 데이터를 쪼개야 진짜 문제가 보인다")

    engine = PandasPartitionEngine()
    schema = power_schema()
    source = power_source(power.curated)

    random_plan = PartitionPlan(
        strategy=SplitStrategy.RANDOM,
        ratio=RATIO,
        time_field=TIME_FIELD,
        group_field=GROUP_FIELD,
    )
    time_plan = PartitionPlan(
        strategy=SplitStrategy.TIME_ORDERED, ratio=RATIO, time_field=TIME_FIELD
    )

    bad = engine.apply(source, schema, random_plan, LABEL_FIELD)
    good = engine.apply(source, schema, time_plan, LABEL_FIELD)

    report.block(
        "같은 데이터, 다른 분할",
        f"{'':<14}{'train':>8}{'val':>8}{'test':>8}{'그룹누수':>10}{'시간누수(초)':>14}\n"
        f"{'RANDOM':<14}{bad.train_count:>8,}{bad.validation_count:>8,}"
        f"{bad.test_count:>8,}{bad.overlapping_group_count:>10}"
        f"{bad.time_overlap_seconds:>14,.0f}\n"
        f"{'TIME_ORDERED':<14}{good.train_count:>8,}{good.validation_count:>8,}"
        f"{good.test_count:>8,}{good.overlapping_group_count:>10}"
        f"{good.time_overlap_seconds:>14,.0f}",
    )

    assert bad.time_overlap_seconds > 80_000  # train 이 test 를 하루치 침범한다
    assert bad.overlapping_group_count > 40   # 거의 모든 LOT 이 양쪽에 걸친다
    assert good.time_overlap_seconds == 0.0
    assert good.overlapping_group_count == 0

    report.note(
        f"무작위 분할은 train 이 test 구간을 {bad.time_overlap_seconds / 3600:.1f}시간 침범한다. "
        "10:00:00 으로 배우고 10:00:01 을 맞히는 시험이다."
    )


def test_시계열을_무작위로_나누려는_계획은_거부된다(container, power) -> None:
    prepare(container, power)

    plan = PartitionPlan(
        strategy=SplitStrategy.RANDOM, ratio=RATIO, time_field=TIME_FIELD
    )
    with pytest.raises(InvariantViolation, match="미래가 학습에 섞인다"):
        container.partition_dataset().execute(
            PartitionDatasetCommand(dataset_id="curated", plan=plan)
        )

    report.note(
        "숫자를 본 뒤에야 이 규칙이 납득된다. "
        "규칙을 먼저 외우면 다음 프로젝트에서 다시 어긴다."
    )


def test_시간_기준_분할은_통과한다(container, power) -> None:
    prepare(container, power)
    view = container.partition_dataset().execute(
        PartitionDatasetCommand(
            dataset_id="curated",
            plan=PartitionPlan(
                strategy=SplitStrategy.TIME_ORDERED,
                ratio=RATIO,
                time_field=TIME_FIELD,
            ),
        )
    )
    report.block("시간 기준 분할", view.render())

    assert view.inspection.verdict in ("PASSED", "PASSED_WITH_WARNINGS")
    assert view.time_overlap_seconds == 0.0
    assert "PARTITION_TIME_LEAKAGE" not in codes(view)


def test_그룹_분할은_LOT_을_통째로_한쪽에만_넣는다(container, power) -> None:
    """이미지 데이터셋에서 같은 부품이 train 과 test 에 흩어지는 사고를 막는 방식."""
    prepare(container, power)
    view = container.partition_dataset().execute(
        PartitionDatasetCommand(
            dataset_id="curated",
            plan=PartitionPlan(
                strategy=SplitStrategy.GROUP_HOLDOUT,
                ratio=RATIO,
                group_field=GROUP_FIELD,
            ),
        )
    )
    report.block("LOT 단위 분할", view.render())

    assert view.overlapping_group_count == 0
    assert "PARTITION_GROUP_LEAKAGE" not in codes(view)
    report.note(
        "그룹 분할은 비율이 정확히 7:1.5:1.5 가 되지 않는다. "
        "LOT 단위로 끊기 때문이다. 그 대가로 누수가 사라진다."
    )


def test_스키마에_없는_필드로는_나눌_수_없다(container, power) -> None:
    from domain.data.errors import UnknownField

    prepare(container, power)
    with pytest.raises(UnknownField):
        container.partition_dataset().execute(
            PartitionDatasetCommand(
                dataset_id="curated",
                plan=PartitionPlan(
                    strategy=SplitStrategy.GROUP_HOLDOUT,
                    ratio=RATIO,
                    group_field="없는열",
                ),
            )
        )
