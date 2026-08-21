"""Infrastructure 어댑터 — 심어 놓은 결함을 실제로 세는가.

샘플 데이터에는 어떤 결함이 몇 개 들어 있는지가 정확히 알려져 있다.
(infrastructure/sample_data/plant_power.py 의 D01~D13, casting_images.py 의 I01~I05)
정답을 아는 데이터로 측정기를 검증하는 것이 이 테스트의 목적이다.
"""

from __future__ import annotations

import pytest

from domain.data.profile import FieldType
from infrastructure.analysis.numpy_distribution_comparer import (
    NumpyDistributionComparer,
    population_stability_index,
)
from infrastructure.analysis.pandas_dataset_profiler import (
    HeuristicSchemaInferrer,
    PandasDatasetProfiler,
)
from infrastructure.analysis.pandas_label_measurer import PandasLabelMeasurer
from infrastructure.analysis.pandas_normalization_fitter import (
    PandasNormalizationFitter,
)
from infrastructure.analysis.pandas_partition_engine import PandasPartitionEngine
from infrastructure.analysis.pandas_sensor_signal_measurer import (
    PandasSensorSignalMeasurer,
)
from infrastructure.analysis.pandas_time_axis_measurer import PandasTimeAxisMeasurer
from infrastructure.analysis.pillow_image_signal_measurer import (
    PillowImageSignalMeasurer,
)
from infrastructure.errors import SourceUnreadable, UnsupportedSourceFormat
from tests.support.scenario import (
    GROUP_FIELD,
    LABEL_FIELD,
    TIME_FIELD,
    image_source,
    power_schema,
    power_source,
)


class TestPandasDatasetProfiler:
    def test_D01_결측과_D10_상수열을_센다(self, power) -> None:
        profile = PandasDatasetProfiler().profile(power_source(power.raw))

        assert profile.row_count == 8460
        assert profile.column("temperature_c").missing_ratio == pytest.approx(
            0.03, abs=0.005
        )
        assert profile.column("meter_id").is_constant is True
        assert "meter_id" in [c.name for c in profile.constant_columns]

    def test_타입을_추론하되_주장일_뿐이다(self, power) -> None:
        profile = PandasDatasetProfiler().profile(power_source(power.curated))

        assert profile.column(TIME_FIELD).inferred_type is FieldType.TIMESTAMP
        assert profile.column("active_power_kw").inferred_type is FieldType.REAL
        assert profile.column(LABEL_FIELD).inferred_type is FieldType.CATEGORY

    def test_이미지_디렉터리는_표로_읽을_수_없다(self, castings) -> None:
        with pytest.raises(UnsupportedSourceFormat):
            PandasDatasetProfiler().profile(image_source(castings.root))

    def test_없는_파일은_원본_오류다(self) -> None:
        from pathlib import Path

        with pytest.raises(SourceUnreadable):
            PandasDatasetProfiler().profile(power_source(Path("없는파일.csv")))


class TestHeuristicSchemaInferrer:
    def test_초안은_사람의_확정과_다르다(self, power) -> None:
        profile = PandasDatasetProfiler().profile(power_source(power.raw))
        draft = HeuristicSchemaInferrer().infer(profile)

        assert draft.time_index is not None
        assert draft.time_index.name == TIME_FIELD
        # 이름이 id 로 끝난다는 이유로 상수 열을 식별자로 본다 — 틀렸다.
        assert draft.field_of("meter_id").role.value == "IDENTIFIER"


class TestPandasTimeAxisMeasurer:
    def test_D05_중복과_D06_역순과_D07_공백을_센다(self, power) -> None:
        measurement = PandasTimeAxisMeasurer().measure(
            power_source(power.raw), TIME_FIELD
        )

        assert measurement.duplicate_timestamp_count == 40
        assert 25 <= measurement.out_of_order_count <= 30
        assert measurement.gap_count == 2
        assert measurement.longest_gap_seconds == pytest.approx(910.0)
        assert measurement.median_interval_seconds == pytest.approx(10.0)

    def test_정리본에는_결함이_없다(self, power) -> None:
        measurement = PandasTimeAxisMeasurer().measure(
            power_source(power.curated), TIME_FIELD
        )
        assert measurement.duplicate_timestamp_count == 0
        assert measurement.out_of_order_count == 0
        assert measurement.gap_count == 0

    def test_없는_열을_지목하면_거부한다(self, power) -> None:
        with pytest.raises(SourceUnreadable, match="없는열"):
            PandasTimeAxisMeasurer().measure(power_source(power.raw), "없는열")


class TestPandasSensorSignalMeasurer:
    def test_D02_고착과_D03_범위이탈과_D04_포화를_센다(self, power) -> None:
        measurements = {
            m.field_name: m
            for m in PandasSensorSignalMeasurer().measure(
                power_source(power.raw), power_schema()
            )
        }

        assert measurements["voltage_v"].longest_constant_run == 600
        assert measurements["current_a"].out_of_range_count == 20
        assert measurements["active_power_kw"].saturated_count > 200

    def test_FEATURE_가_아닌_열은_재지_않는다(self, power) -> None:
        names = {
            m.field_name
            for m in PandasSensorSignalMeasurer().measure(
                power_source(power.raw), power_schema()
            )
        }
        assert TIME_FIELD not in names
        assert GROUP_FIELD not in names
        assert LABEL_FIELD not in names


class TestPillowImageSignalMeasurer:
    def test_I01부터_I05까지_전부_센다(self, castings) -> None:
        measurement = PillowImageSignalMeasurer().measure(image_source(castings.root))

        assert measurement.total_images == 51
        assert measurement.unreadable_count == 3          # I01
        assert measurement.defocused_count(50.0) == 6     # I02
        assert measurement.brightness_stddev > 40.0       # I03
        assert measurement.visual_duplicate_count == 4    # I04
        assert measurement.distinct_resolution_count == 2  # I05

    def test_지문_해상도를_낮추면_서로_다른_부품이_중복으로_뭉친다(self, castings) -> None:
        """중복 판정은 임계 설정에 민감하다 — 그 사실 자체를 알고 있어야 한다."""
        coarse = PillowImageSignalMeasurer(hash_size=4).measure(
            image_source(castings.root)
        )
        fine = PillowImageSignalMeasurer(hash_size=16).measure(
            image_source(castings.root)
        )
        assert coarse.visual_duplicate_count > fine.visual_duplicate_count

    def test_이미지가_없는_디렉터리는_거부한다(self, tmp_path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SourceUnreadable, match="한 장도 없다"):
            PillowImageSignalMeasurer().measure(image_source(empty))


class TestPandasLabelMeasurer:
    def test_D08_D09_D12_D13_을_센다(self, power) -> None:
        measurement = PandasLabelMeasurer().measure(
            power_source(power.raw), LABEL_FIELD
        )

        assert "UNKNOWN" in measurement.class_counts       # D08
        assert measurement.unlabeled_count == 12           # D09
        assert measurement.agreement_ratio < 0.9           # D12
        assert measurement.imbalance_ratio > 100           # D13
        assert measurement.annotator_count == 2

    def test_교차검토_열이_없으면_일치율을_주장하지_않는다(self, power, tmp_path) -> None:
        import pandas as pd

        frame = pd.read_csv(power.curated).drop(columns=["condition_review"])
        path = tmp_path / "no_review.csv"
        frame.to_csv(path, index=False)

        measurement = PandasLabelMeasurer().measure(power_source(path), LABEL_FIELD)
        assert measurement.reviewed_sample_count == 0
        assert measurement.agreement_ratio == 0.0
        assert measurement.annotator_count == 1


class TestPandasPartitionEngine:
    def test_시간_분할은_새지_않고_무작위_분할은_샌다(self, power) -> None:
        from domain.data.partition import PartitionPlan, SplitRatio, SplitStrategy

        engine = PandasPartitionEngine()
        ratio = SplitRatio.of(0.7, 0.15, 0.15)

        random_result = engine.apply(
            power_source(power.curated),
            power_schema(),
            PartitionPlan(SplitStrategy.RANDOM, ratio, TIME_FIELD, GROUP_FIELD),
            LABEL_FIELD,
        )
        time_result = engine.apply(
            power_source(power.curated),
            power_schema(),
            PartitionPlan(SplitStrategy.TIME_ORDERED, ratio, TIME_FIELD),
            LABEL_FIELD,
        )

        assert random_result.time_overlap_seconds > 80_000
        assert time_result.time_overlap_seconds == 0.0
        assert random_result.total_count == time_result.total_count == 8640

    def test_그룹_분할은_LOT_을_쪼개지_않는다(self, power) -> None:
        from domain.data.partition import PartitionPlan, SplitRatio, SplitStrategy

        result = PandasPartitionEngine().apply(
            power_source(power.curated),
            power_schema(),
            PartitionPlan(
                SplitStrategy.GROUP_HOLDOUT,
                SplitRatio.of(0.7, 0.15, 0.15),
                group_field=GROUP_FIELD,
            ),
            LABEL_FIELD,
        )
        assert result.overlapping_group_count == 0


class TestPandasNormalizationFitter:
    def test_통계는_train_구간에서만_나온다(self, power) -> None:
        import pandas as pd

        from domain.data.partition import PartitionPlan, SplitRatio, SplitStrategy
        from domain.data.training_spec import NormalizationMethod

        plan = PartitionPlan(
            SplitStrategy.TIME_ORDERED,
            SplitRatio.of(0.7, 0.15, 0.15),
            time_field=TIME_FIELD,
        )
        statistics = PandasNormalizationFitter().fit(
            power_source(power.curated),
            power_schema(),
            plan,
            ("active_power_kw",),
            NormalizationMethod.ZSCORE,
        )
        mean, stddev = statistics["active_power_kw"]

        frame = pd.read_csv(power.curated).sort_values(TIME_FIELD)
        train_only = frame["active_power_kw"].head(int(len(frame) * 0.7))

        assert mean == pytest.approx(float(train_only.mean()), rel=1e-6)
        assert mean != pytest.approx(float(frame["active_power_kw"].mean()), rel=1e-6)
        assert stddev > 0


class TestNumpyDistributionComparer:
    def test_같은_분포의_PSI_는_0에_가깝다(self) -> None:
        import numpy as np

        rng = np.random.default_rng(7)
        a = rng.normal(0, 1, 5000)
        b = rng.normal(0, 1, 5000)
        assert population_stability_index(a, b) < 0.02

    def test_평균이_이동하면_PSI_가_커진다(self) -> None:
        import numpy as np

        rng = np.random.default_rng(7)
        a = rng.normal(0, 1, 5000)
        b = rng.normal(2, 1, 5000)
        assert population_stability_index(a, b) > 1.0

    def test_계절이_바뀐_현장은_잡고_같은_계절은_잡지_않는다(self, power) -> None:
        comparer = NumpyDistributionComparer()

        stable = comparer.compare(
            power_source(power.curated),
            power_source(power.recent_stable),
            power_schema(),
        )
        shifted = comparer.compare(
            power_source(power.curated),
            power_source(power.recent_shifted),
            power_schema(),
        )

        assert stable.worst_psi < 0.1
        assert shifted.worst_psi > 1.0
        assert shifted.worst_field == "temperature_c"
