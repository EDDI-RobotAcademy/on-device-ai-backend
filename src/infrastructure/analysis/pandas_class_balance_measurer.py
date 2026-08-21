"""ClassBalanceMeasurer 구현 (pandas). — 실습 2-5

세는 것은 클래스별 표본 수 하나뿐이다.
'불균형이 얼마나 나쁜가'는 전부 Domain(BalancePolicy)이 판단한다.
"""

from __future__ import annotations

from domain.data_quality.balance import ClassBalanceMeasurement
from domain.data_quality.target import AssessmentTarget
from infrastructure.analysis.table_loader import load_frame
from infrastructure.errors import SourceUnreadable


class PandasClassBalanceMeasurer:
    """domain.data_quality.ports.ClassBalanceMeasurer 구현."""

    def __init__(self, test_split_ratio: float = 0.15) -> None:
        self._test_split_ratio = test_split_ratio

    def measure(self, target: AssessmentTarget) -> ClassBalanceMeasurement:
        if not target.label_field:
            raise SourceUnreadable("라벨 필드가 지정되지 않았다.", subject=target.uri)

        frame = load_frame(target.uri, target.source_format).frame
        if target.label_field not in frame.columns:
            raise SourceUnreadable(
                f"'{target.label_field}' 열이 원본에 없다.", subject=target.label_field
            )

        labels = frame[target.label_field].astype("string").str.strip()
        present = labels[labels.notna() & (labels != "")]
        counts = present.value_counts()

        return ClassBalanceMeasurement(
            class_counts={str(name): int(count) for name, count in counts.items()},
            test_split_ratio=self._test_split_ratio,
        )
