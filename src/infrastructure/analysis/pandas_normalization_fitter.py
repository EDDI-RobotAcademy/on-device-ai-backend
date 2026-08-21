"""NormalizationFitter 구현 (pandas). — 실습 1-7

정규화 통계를 **train 분할에서만** 계산한다.

전체 데이터로 평균과 표준편차를 구하면, 그 안에는 test 의 정보가 섞인다.
모델은 시험 문제의 평균을 미리 알고 시험을 보는 셈이 되고,
검증 점수는 올라가지만 현장 성능은 그대로다. 가장 잡기 어려운 종류의 누수다.
"""

from __future__ import annotations

from collections.abc import Mapping

from domain.data.partition import PartitionPlan
from domain.data.schema import DataSchema
from domain.data.source import DataSourceDescriptor
from domain.data.training_spec import NormalizationMethod
from infrastructure.analysis.pandas_partition_engine import assign_splits
from infrastructure.analysis.table_loader import load_table, numeric_view
from infrastructure.errors import SourceUnreadable


class PandasNormalizationFitter:
    """domain.data.ports.NormalizationFitter 구현."""

    def fit(
        self,
        source: DataSourceDescriptor,
        schema: DataSchema,
        plan: PartitionPlan,
        feature_fields: tuple[str, ...],
        method: NormalizationMethod,
    ) -> Mapping[str, tuple[float, float]]:
        if method is NormalizationMethod.NONE:
            return {}

        frame = load_table(source).frame
        assignment = assign_splits(frame, plan)
        train = frame.loc[assignment == "train"]
        if train.empty:
            raise SourceUnreadable("train 분할이 비어 있다.", subject=source.uri)

        statistics: dict[str, tuple[float, float]] = {}
        for name in feature_fields:
            if name not in train.columns:
                raise SourceUnreadable(f"'{name}' 열이 원본에 없다.", subject=name)
            values = numeric_view(train[name]).dropna()
            if values.empty:
                raise SourceUnreadable(
                    f"'{name}' 의 train 구간에 수치 값이 없다.", subject=name
                )

            if method is NormalizationMethod.ZSCORE:
                mean = float(values.mean())
                stddev = float(values.std(ddof=0))
                if stddev <= 0:
                    raise SourceUnreadable(
                        f"'{name}' 은 train 구간에서 값이 변하지 않는다. "
                        "정규화할 수 없다 — 애초에 입력으로 쓸 이유가 없는 열이다.",
                        subject=name,
                    )
                statistics[name] = (mean, stddev)
            else:  # MINMAX
                minimum = float(values.min())
                maximum = float(values.max())
                if maximum <= minimum:
                    raise SourceUnreadable(
                        f"'{name}' 의 train 구간 최대/최소가 같다.", subject=name
                    )
                statistics[name] = (minimum, maximum)

        return statistics
