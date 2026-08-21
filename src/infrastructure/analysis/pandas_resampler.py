"""불균형 완화 전략을 실제로 적용해 본다. (실습 2-11)

**적용만 한다.** 무엇을 잃었는지 판정하는 것은 RebalancingPolicy 다.

여기서 만들어지는 것은 새 CSV 가 아니라 **숫자**다.
실제 파이프라인이라면 여기서 만든 인덱스를 학습이 그대로 쓴다.
"""

from __future__ import annotations

import numpy as np

from domain.data_quality.rebalancing import (
    RebalancingOutcome,
    RebalancingPlan,
    RebalancingStrategy,
)
from infrastructure.analysis.table_loader import load_frame
from infrastructure.errors import SourceUnreadable


class PandasResampler:
    """domain.data_quality.ports.Resampler 구현."""

    def resample(
        self,
        uri: str,
        source_format: str,
        *,
        label_field: str,
        plan: RebalancingPlan,
        train_ratio: float = 0.7,
        seed: int = 42,
    ) -> RebalancingOutcome:
        frame = load_frame(uri, source_format).frame
        if label_field not in frame.columns:
            raise SourceUnreadable(f"'{label_field}' 열이 없다.", subject=label_field)

        labels = frame[label_field].astype("string").str.strip()
        valid = labels.notna() & (labels != "")

        # 분할 후 적용이면 train 구간에만 손을 댄다.
        # 분할 전 적용이면 전체에 손을 댄다 — 그러면 test 에도 복제가 들어간다.
        cut = int(len(frame) * train_ratio)
        scope = np.zeros(len(frame), dtype=bool)
        if plan.applied_after_split:
            scope[:cut] = True
        else:
            scope[:] = True
        scope &= valid.to_numpy()

        values = labels.to_numpy()
        before = _counts(values[scope])
        if not before:
            raise SourceUnreadable("라벨이 하나도 없다.", subject=label_field)

        distinct_minority = min(before.values())
        rng = np.random.default_rng(seed)

        duplicated = discarded = synthesized = 0
        after = dict(before)
        largest = max(before.values())
        target = max(1, int(round(largest * plan.target_ratio)))

        if plan.strategy is RebalancingStrategy.OVERSAMPLE:
            for name, count in before.items():
                if count < target:
                    duplicated += target - count
                    after[name] = target
        elif plan.strategy is RebalancingStrategy.UNDERSAMPLE:
            smallest = min(before.values())
            keep = max(1, int(round(smallest / plan.target_ratio)))
            for name, count in before.items():
                if count > keep:
                    discarded += count - keep
                    after[name] = keep
        elif plan.strategy is RebalancingStrategy.SYNTHETIC:
            for name, count in before.items():
                if count < target:
                    synthesized += target - count
                    after[name] = target
        # NONE 과 CLASS_WEIGHT 는 데이터를 건드리지 않는다.

        _ = rng  # 시드는 재현성을 위해 받아 두지만, 여기서는 개수만 센다

        return RebalancingOutcome(
            strategy=plan.strategy,
            before=before,
            after=after,
            duplicated_rows=duplicated,
            discarded_rows=discarded,
            synthesized_rows=synthesized,
            distinct_minority_samples=distinct_minority,
        )


def _counts(values: np.ndarray) -> dict[str, int]:
    names, counts = np.unique(values.astype(str), return_counts=True)
    return {str(n): int(c) for n, c in zip(names, counts, strict=True)}
