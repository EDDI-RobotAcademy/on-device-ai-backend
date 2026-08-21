"""TrainingDataSpec — 모델 입력 계약의 불변식. (실습 1-7)"""

from __future__ import annotations

import pytest

from domain.data.profile import FieldType
from domain.data.schema import DataSchema, FieldRole, FieldSpec
from domain.data.time_axis import SamplingInterval
from domain.data.training_spec import (
    ImageInputSpec,
    NormalizationMethod,
    NormalizationSpec,
    TrainingDataSpec,
    WindowSpec,
)
from domain.shared.errors import InvariantViolation

SCHEMA = DataSchema(
    fields=(
        FieldSpec("timestamp", FieldType.TIMESTAMP, FieldRole.TIME_INDEX),
        FieldSpec("machine_id", FieldType.CATEGORY, FieldRole.IDENTIFIER),
        FieldSpec("active_power_kw", FieldType.REAL, FieldRole.FEATURE),
        FieldSpec("voltage_v", FieldType.REAL, FieldRole.FEATURE),
        FieldSpec("condition", FieldType.CATEGORY, FieldRole.LABEL),
    )
)

IMAGE_SCHEMA = DataSchema(
    fields=(
        FieldSpec("path", FieldType.IMAGE_REF, FieldRole.FEATURE),
        FieldSpec("verdict", FieldType.CATEGORY, FieldRole.LABEL),
    )
)


def spec(**overrides: object) -> TrainingDataSpec:
    base: dict[str, object] = dict(
        schema=SCHEMA,
        feature_fields=("active_power_kw", "voltage_v"),
        label_field="condition",
        window=WindowSpec(length=30, stride=30, interval=SamplingInterval(10.0)),
    )
    base.update(overrides)
    return TrainingDataSpec(**base)  # type: ignore[arg-type]


class TestWindowSpec:
    def test_stride_가_length_보다_크면_데이터가_통째로_버려진다(self) -> None:
        with pytest.raises(InvariantViolation, match="통째로 버려진다"):
            WindowSpec(length=10, stride=20, interval=SamplingInterval(10.0))

    def test_창이_보는_현장_시간과_겹침_비율을_계산한다(self) -> None:
        window = WindowSpec(length=30, stride=15, interval=SamplingInterval(10.0))
        assert window.duration_seconds == pytest.approx(300.0)
        assert window.overlap_ratio == pytest.approx(0.5)


class TestTrainingDataSpec:
    def test_식별자를_입력에_넣을_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="ID 를 외운다"):
            spec(feature_fields=("machine_id", "voltage_v"))

    def test_정답을_입력에_넣을_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="정확도 100%"):
            spec(feature_fields=("condition",))

    def test_시각_자체를_입력에_넣을_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="배포 즉시 분포가 벗어난다"):
            spec(feature_fields=("timestamp", "voltage_v"))

    def test_라벨이_아닌_필드를_정답으로_지목할_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="정답이 아니다"):
            spec(label_field="voltage_v")

    def test_시간축이_있는데_창_설계가_없으면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="한 줄만 보는 모델"):
            spec(window=None)

    def test_입력_모양을_스스로_계산한다(self) -> None:
        assert spec().input_shape == (30, 2)
        assert spec().input_element_count == 60

    def test_이미지_입력은_채널_우선_모양이다(self) -> None:
        image_spec = TrainingDataSpec(
            schema=IMAGE_SCHEMA,
            feature_fields=("path",),
            label_field="verdict",
            image=ImageInputSpec(width=224, height=224, channels=3),
        )
        assert image_spec.input_shape == (3, 224, 224)
        assert image_spec.input_element_count == 150_528


class TestNormalizationSpec:
    def test_train_이_아닌_곳에서_통계를_뽑으면_거부한다(self) -> None:
        with pytest.raises(InvariantViolation, match="데이터 누수"):
            NormalizationSpec(
                method=NormalizationMethod.ZSCORE,
                fitted_on="all",
                statistics={"voltage_v": (380.0, 2.2)},
            )

    def test_표준편차가_0인_열은_정규화할_수_없다(self) -> None:
        with pytest.raises(InvariantViolation, match="표준편차가 0"):
            NormalizationSpec(
                method=NormalizationMethod.ZSCORE,
                statistics={"voltage_v": (380.0, 0.0)},
            )

    def test_정규화가_없으면_경고한다(self) -> None:
        report = spec().inspect()
        assert "SPEC_NO_NORMALIZATION" in {f.code for f in report.findings}

    def test_통계가_빠진_입력_필드가_있으면_막는다(self) -> None:
        report = spec(
            normalization=NormalizationSpec(
                method=NormalizationMethod.ZSCORE,
                statistics={"voltage_v": (380.0, 2.2)},
            )
        ).inspect()
        codes = {f.code for f in report.findings}
        assert "SPEC_STATISTICS_MISSING" in codes
        assert report.verdict.value == "FAILED"

    def test_창이_과하게_겹치면_경고한다(self) -> None:
        report = spec(
            window=WindowSpec(length=100, stride=1, interval=SamplingInterval(10.0)),
            normalization=NormalizationSpec(
                method=NormalizationMethod.ZSCORE,
                statistics={
                    "active_power_kw": (150.0, 40.0),
                    "voltage_v": (380.0, 2.2),
                },
            ),
        ).inspect()
        assert "SPEC_WINDOW_OVERLAP_HIGH" in {f.code for f in report.findings}
