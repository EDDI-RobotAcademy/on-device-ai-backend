"""LabelErrorMeasurer 구현 (pandas). — 실습 2-4

두 가지를 센다.

    1. 현장 규칙과 모순되는 라벨
       규칙은 Domain(LabelConsistencyRule)이 갖고 있고, 여기서는 적용만 한다.
       규칙이 없으면 아무것도 못 찾는다 — 그 사실 자체를 Policy 가 CRITICAL 로 잡는다.

    2. 같은 입력에 다른 라벨
       모델이 절대 학습할 수 없는 모순이다. 정확도 상한이 그 자리에서 깎인다.
"""

from __future__ import annotations

import pandas as pd

from domain.data_quality.label_quality import (
    LabelConsistencyRule,
    LabelErrorMeasurement,
)
from domain.data_quality.target import AssessmentTarget
from infrastructure.analysis.table_loader import load_frame, numeric_view
from infrastructure.errors import SourceUnreadable


class PandasLabelErrorMeasurer:
    """domain.data_quality.ports.LabelErrorMeasurer 구현."""

    def __init__(self, example_limit: int = 3, round_digits: int = 6) -> None:
        self._example_limit = example_limit
        self._round_digits = round_digits

    def measure(
        self, target: AssessmentTarget, rules: tuple[LabelConsistencyRule, ...] = ()
    ) -> LabelErrorMeasurement:
        if not target.label_field:
            raise SourceUnreadable("라벨 필드가 지정되지 않았다.", subject=target.uri)

        frame = load_frame(target.uri, target.source_format).frame
        if target.label_field not in frame.columns:
            raise SourceUnreadable(
                f"'{target.label_field}' 열이 원본에 없다.", subject=target.label_field
            )

        labels = frame[target.label_field].astype("string").str.strip()
        labeled = labels.notna() & (labels != "")

        violations: dict[str, int] = {}
        by_label: dict[str, int] = {}
        examples: list[str] = []

        for rule in rules:
            if rule.field_name not in frame.columns:
                continue
            values = numeric_view(frame[rule.field_name])
            scope = labeled & (labels == rule.label)
            broken = pd.Series(False, index=frame.index)
            if rule.expected_min is not None:
                broken |= values < rule.expected_min
            if rule.expected_max is not None:
                broken |= values > rule.expected_max
            broken &= scope & values.notna()

            count = int(broken.sum())
            if count == 0:
                continue
            violations[rule.describe()] = count
            by_label[rule.label] = by_label.get(rule.label, 0) + count
            for index in list(frame.index[broken])[: self._example_limit]:
                examples.append(
                    f"row {index}: {rule.label} 인데 "
                    f"{rule.field_name}={values.loc[index]:.4g}"
                )

        conflicting, groups = self._label_conflicts(frame, target, labels, labeled)

        return LabelErrorMeasurement(
            total_labeled=int(labeled.sum()),
            rule_violations=violations,
            violations_by_label=by_label,
            conflicting_duplicate_count=conflicting,
            conflicting_group_count=groups,
            examples=tuple(examples[: self._example_limit * 2]),
        )

    def _label_conflicts(
        self,
        frame: pd.DataFrame,
        target: AssessmentTarget,
        labels: pd.Series,
        labeled: pd.Series,
    ) -> tuple[int, int]:
        """입력이 같은데 라벨이 다른 행을 센다."""
        present = [f for f in target.feature_fields if f in frame.columns]
        if not present:
            return 0, 0

        key = frame.loc[labeled, present].round(self._round_digits)
        if key.empty:
            return 0, 0

        keyed = key.assign(_label=labels[labeled].to_numpy())
        grouped = keyed.groupby(present, dropna=False, observed=True)["_label"]
        distinct = grouped.nunique()
        sizes = grouped.size()

        conflicting_keys = distinct[distinct > 1].index
        if len(conflicting_keys) == 0:
            return 0, 0
        return int(sizes.loc[conflicting_keys].sum()), int(len(conflicting_keys))
