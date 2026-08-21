"""LabelMeasurer 구현 (pandas). — 실습 1-6

교차 검토 규약:
    라벨 필드가 `condition` 이면, 2차 작업자의 판단은 `condition_review` 열에 둔다.
    두 열이 모두 채워진 행이 "교차 검토된 표본"이고, 값이 다른 행이 "불일치"다.

이 규약은 기술적 편의가 아니라 조직의 약속이다.
약속이 없으면 라벨 일치율을 측정할 방법 자체가 없다.
"""

from __future__ import annotations

import pandas as pd

from domain.data.labeling import LabelAgreementMeasurement
from domain.data.source import DataSourceDescriptor
from infrastructure.analysis.table_loader import load_table
from infrastructure.errors import SourceUnreadable

REVIEW_SUFFIX = "_review"


class PandasLabelMeasurer:
    """domain.data.ports.LabelMeasurer 구현."""

    def measure(
        self, source: DataSourceDescriptor, label_field: str
    ) -> LabelAgreementMeasurement:
        frame = load_table(source).frame
        if label_field not in frame.columns:
            raise SourceUnreadable(
                f"'{label_field}' 열이 원본에 없다.", subject=label_field
            )

        labels = frame[label_field]
        normalized = labels.astype("string").str.strip()
        blank = normalized.isna() | (normalized == "")

        counts = normalized[~blank].value_counts()
        class_counts = {str(name): int(count) for name, count in counts.items()}

        review_column = f"{label_field}{REVIEW_SUFFIX}"
        reviewed = disagreement = 0
        annotators = 1
        if review_column in frame.columns:
            annotators = 2
            second = frame[review_column].astype("string").str.strip()
            second_blank = second.isna() | (second == "")
            both = ~blank & ~second_blank
            reviewed = int(both.sum())
            disagreement = int((normalized[both] != second[both]).sum())

        return LabelAgreementMeasurement(
            class_counts=class_counts,
            annotator_count=annotators,
            reviewed_sample_count=reviewed,
            disagreement_count=disagreement,
            unlabeled_count=int(blank.sum()),
        )


def observed_label_values(frame: pd.DataFrame, label_field: str) -> tuple[str, ...]:
    """디버깅/교육용 보조: 실제로 등장한 라벨 값들."""
    values = frame[label_field].astype("string").str.strip().dropna().unique()
    return tuple(sorted(str(v) for v in values if v))
