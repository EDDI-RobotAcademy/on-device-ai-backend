"""모듈 2 측정 어댑터 — 심어 놓은 오염을 실제로 세는가.

정답은 infrastructure/sample_data/plant_power_quality.py 의 Q01~Q10 에 적혀 있다.
"""

from __future__ import annotations

import pytest

from domain.data_quality.target import AssessmentTarget
from infrastructure.analysis.pandas_class_balance_measurer import (
    PandasClassBalanceMeasurer,
)
from infrastructure.analysis.pandas_duplicate_measurer import PandasDuplicateMeasurer
from infrastructure.analysis.pandas_label_error_measurer import (
    PandasLabelErrorMeasurer,
)
from infrastructure.analysis.pandas_missing_value_measurer import (
    PandasMissingValueMeasurer,
)
from infrastructure.analysis.pandas_noise_measurer import PandasNoiseMeasurer
from infrastructure.analysis.pandas_outlier_measurer import PandasOutlierMeasurer
from infrastructure.errors import SourceUnreadable
from tests.support.quality_scenario import label_rules

FEATURES = (
    "active_power_kw",
    "reactive_power_kvar",
    "current_a",
    "voltage_v",
    "temperature_c",
    "spindle_rpm",
)
RANGES = {
    "active_power_kw": (0.0, 400.0),
    "reactive_power_kvar": (0.0, 400.0),
    "current_a": (0.0, 600.0),
    "voltage_v": (300.0, 440.0),
    "temperature_c": (-20.0, 120.0),
    "spindle_rpm": (0.0, 4000.0),
}


def target(path, ref: str = "ds") -> AssessmentTarget:  # noqa: ANN001
    return AssessmentTarget(
        dataset_ref=ref,
        uri=str(path),
        feature_fields=FEATURES,
        label_field="condition",
        time_field="timestamp",
        group_field="batch_id",
        physical_ranges=RANGES,
    )


class TestMissingValueMeasurer:
    def test_Q01_결측과_집중도를_센다(self, quality) -> None:
        measurement = PandasMissingValueMeasurer().measure(target(quality.dirty))
        temperature = measurement.field_of("temperature_c")

        assert temperature is not None
        assert temperature.missing_ratio == pytest.approx(0.04, abs=0.005)
        assert temperature.concentration_ratio > 0.6

    def test_Q02_은폐된_결측을_찾는다(self, quality) -> None:
        measurement = PandasMissingValueMeasurer().measure(target(quality.dirty))
        temperature = measurement.field_of("temperature_c")

        assert temperature.repeated_value == 0.0
        assert temperature.repeated_value_count == pytest.approx(432, abs=20)
        assert temperature.repeated_value_mean_run < 1.2  # 흩어져 있다

    def test_설비_정지_중의_0은_뭉쳐_있다(self, quality) -> None:
        """같은 '반복되는 0' 이지만 성격이 다르다."""
        measurement = PandasMissingValueMeasurer().measure(target(quality.clean))
        spindle = measurement.field_of("spindle_rpm")

        assert spindle.repeated_value == 0.0
        assert spindle.repeated_value_mean_run >= 9.0  # 10표본짜리 정지 구간

    def test_기준선에는_결측이_없다(self, quality) -> None:
        measurement = PandasMissingValueMeasurer().measure(target(quality.clean))
        assert measurement.worst_missing_ratio == 0.0


class TestOutlierMeasurer:
    def test_Q03_스파이크를_MAD_로_잡는다(self, quality) -> None:
        measurement = PandasOutlierMeasurer().measure(target(quality.dirty))
        power = measurement.field_of("active_power_kw")

        assert power.mad_outlier_count >= 110
        assert power.out_of_physical_range_count == 0  # 물리 범위 안이다

    def test_Q04_변화율_위반을_센다(self, quality) -> None:
        measurement = PandasOutlierMeasurer().measure(target(quality.dirty))
        current = measurement.field_of("current_a")
        assert current.rate_violation_count > 0

    def test_라벨별_이상치_분포를_낸다(self, quality) -> None:
        measurement = PandasOutlierMeasurer().measure(target(quality.dirty))
        power = measurement.field_of("active_power_kw")

        assert power.outliers_by_label
        assert power.outlier_share_of("NORMAL") > 0.9

    def test_MAD_가_0이면_표준편차로_물러선다(self, tmp_path) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=100, freq="10s"),
                "v": [1.0] * 99 + [99.0],
                "condition": ["NORMAL"] * 100,
            }
        )
        path = tmp_path / "constant.csv"
        frame.to_csv(path, index=False)

        measurement = PandasOutlierMeasurer().measure(
            AssessmentTarget(
                dataset_ref="x",
                uri=str(path),
                feature_fields=("v",),
                label_field="condition",
                time_field="timestamp",
            )
        )
        assert measurement.field_of("v").mad_outlier_count >= 1


class TestLabelErrorMeasurer:
    def test_Q05_규칙_위반을_센다(self, quality) -> None:
        measurement = PandasLabelErrorMeasurer().measure(
            target(quality.dirty), label_rules()
        )
        assert measurement.total_violation_count >= 150
        assert "FAULT" in measurement.violations_by_label
        assert measurement.examples

    def test_Q10_라벨_모순을_센다(self, quality) -> None:
        measurement = PandasLabelErrorMeasurer().measure(
            target(quality.dirty), label_rules()
        )
        assert measurement.conflicting_duplicate_count > 0
        assert measurement.conflicting_group_count > 0

    def test_규칙이_없으면_위반도_0이다(self, quality) -> None:
        measurement = PandasLabelErrorMeasurer().measure(target(quality.dirty), ())
        assert measurement.total_violation_count == 0

    def test_기준선은_규칙을_지킨다(self, quality) -> None:
        measurement = PandasLabelErrorMeasurer().measure(
            target(quality.clean), label_rules()
        )
        assert measurement.total_violation_count == 0
        assert measurement.conflicting_duplicate_count == 0

    def test_라벨_필드가_없으면_거부한다(self, quality) -> None:
        bad = AssessmentTarget(
            dataset_ref="x", uri=str(quality.dirty), feature_fields=FEATURES
        )
        with pytest.raises(SourceUnreadable, match="라벨 필드"):
            PandasLabelErrorMeasurer().measure(bad, label_rules())


class TestClassBalanceMeasurer:
    def test_Q06_클래스_분포를_센다(self, quality) -> None:
        measurement = PandasClassBalanceMeasurer().measure(target(quality.clean))
        assert set(measurement.class_counts) == {"NORMAL", "OVERLOAD", "FAULT"}
        assert measurement.minority_count == 150
        assert measurement.total_count == 8640


class TestNoiseMeasurer:
    def test_Q07_전압_잡음을_잡는다(self, quality) -> None:
        measurement = PandasNoiseMeasurer().measure(target(quality.dirty))
        voltage = measurement.field_of("voltage_v")

        assert voltage.snr_db < 10
        assert voltage.reversal_ratio > 0.95  # 톱니

    def test_기준선의_모든_채널은_깨끗하다(self, quality) -> None:
        measurement = PandasNoiseMeasurer().measure(target(quality.clean))
        assert measurement.worst_snr_db > 20.0

    def test_설비_정지_150건이_잡음으로_계산되지_않는다(self, quality) -> None:
        """분산 대신 MAD 를 쓰는 이유."""
        measurement = PandasNoiseMeasurer().measure(target(quality.clean))
        spindle = measurement.field_of("spindle_rpm")
        assert spindle.snr_db > 20.0

    def test_창_크기는_홀수여야_한다(self) -> None:
        with pytest.raises(ValueError):
            PandasNoiseMeasurer(window=8)


class TestDuplicateMeasurer:
    def test_Q08_입력_기준_중복을_센다(self, quality) -> None:
        measurement = PandasDuplicateMeasurer().measure(target(quality.dirty))
        assert measurement.exact_duplicate_count >= 120
        assert measurement.duplicate_group_count > 0
        assert measurement.inflation_ratio > 1.0

    def test_Q09_근접_중복을_센다(self, quality) -> None:
        measurement = PandasDuplicateMeasurer().measure(target(quality.dirty))
        assert measurement.near_duplicate_count >= 100

    def test_Q10_라벨_모순을_센다(self, quality) -> None:
        measurement = PandasDuplicateMeasurer().measure(target(quality.dirty))
        assert measurement.conflicting_label_count > 0

    def test_기준선에는_중복이_없다(self, quality) -> None:
        measurement = PandasDuplicateMeasurer().measure(target(quality.clean))
        assert measurement.exact_duplicate_count == 0
        assert measurement.conflicting_label_count == 0
        assert measurement.near_duplicate_count == 0

    def test_입력_열이_없으면_거부한다(self, quality) -> None:
        bad = AssessmentTarget(
            dataset_ref="x", uri=str(quality.dirty), feature_fields=("없는열",)
        )
        with pytest.raises(SourceUnreadable, match="입력 열"):
            PandasDuplicateMeasurer().measure(bad)


class TestQualitySampleData:
    def test_같은_seed_는_같은_파일을_만든다(self, tmp_path) -> None:
        from infrastructure.sample_data import write_quality_samples

        first = write_quality_samples(tmp_path / "a", seed=77)
        second = write_quality_samples(tmp_path / "b", seed=77)
        assert first.dirty.read_bytes() == second.dirty.read_bytes()

    def test_두_파일의_구조는_동일하다(self, quality) -> None:
        """모듈 1의 검사가 둘을 구분하지 못한다는 것이 실습 2-1 의 전제다."""
        import pandas as pd

        clean = pd.read_csv(quality.clean)
        dirty = pd.read_csv(quality.dirty)

        assert list(clean.columns) == list(dirty.columns)
        assert len(clean) == len(dirty) == 8640
        assert clean["timestamp"].equals(dirty["timestamp"])
