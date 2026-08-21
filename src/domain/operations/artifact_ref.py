"""배포 대상 (Anti-Corruption Layer VO). (실습 5-1)

Operations Context 는 `OptimizationRun` 도 `ModelArtifact` 도 모른다.
알아야 하는 것은 네 가지뿐이다.

    무엇을 올리는가        artifact_id / runtime / precision
    올려도 되는가          selected — 모듈 4 의 선택 판정
    무엇을 기대하는가      expected_* — 현장과 비교할 **기준선**
    무엇을 먹는가          input_fields — 입력이 변했는지 보려면 필요하다

`expected_p95_ms` 를 굳이 들고 오는 이유가 중요하다.
현장에서 "느려졌다"고 말하려면 **원래 얼마였는지**를 알아야 한다.
그 숫자는 모듈 4 의 벤치마크에서 온다. 여기서 새로 만들지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class DeployedArtifactRef:
    """현장에 올릴 결과물."""

    artifact_id: str
    optimization_run_ref: str
    model_version_id: str
    runtime: str
    """'TFLITE' 같은 문자열이다. RuntimeTarget enum 을 import 하지 않는다."""

    precision: str
    size_bytes: int
    class_labels: tuple[str, ...]
    input_fields: tuple[str, ...] = field(default_factory=tuple)

    expected_p95_ms: float = 0.0
    """최적화 때 잰 지연시간. 현장 지연시간을 여기에 견준다. (실습 5-5)"""

    expected_accuracy: float = 0.0
    expected_class_mix: Mapping[str, float] = field(default_factory=dict)
    """평가 때 이 모델이 각 클래스를 답한 비율. 현장 분포를 여기에 견준다. (실습 5-6)"""

    normalization: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    """채널별 (평균, 표준편차). **전처리는 모델의 일부다.**

    결과물 파일만 디바이스에 보내고 이 숫자를 안 보내면,
    디바이스는 다른 전처리를 하게 되고 모델은 학습 때와 다른 입력을 받는다.
    변환 동등성(실습 4-2)을 아무리 확인해도 이 어긋남은 잡히지 않는다 —
    같은 입력을 넣지 않았기 때문이다.
    """

    selected: bool = False
    """모듈 4 의 모델 선택 판정을 통과했는가."""

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise InvariantViolation(
                "무엇을 배포하는지 없으면 나중에 되돌릴 수 없다.", subject="artifact_id"
            )
        if len(self.class_labels) < 2:
            raise InvariantViolation("클래스가 둘 미만이다.", subject="class_labels")
        if self.size_bytes < 0:
            raise InvariantViolation("크기는 음수일 수 없다.", subject="size_bytes")
        if self.expected_p95_ms < 0:
            raise InvariantViolation(
                "기준 지연시간은 음수일 수 없다.", subject="expected_p95_ms"
            )
        if not 0.0 <= self.expected_accuracy <= 1.0:
            raise InvariantViolation(
                "expected_accuracy 는 0~1 이어야 한다.", subject="expected_accuracy"
            )

    @property
    def label(self) -> str:
        return f"{self.runtime}/{self.precision}"

    @property
    def missing_gates(self) -> tuple[str, ...]:
        """무엇을 통과하지 않았는가."""
        return () if self.selected else ("모듈 4 · 모델 선택 판정",)

    @property
    def has_baseline(self) -> bool:
        """비교할 기준을 들고 왔는가.

        기준 없이 배포하면 현장 숫자를 봐도 좋은지 나쁜지 말할 수 없다.
        """
        return self.expected_p95_ms > 0 and bool(self.expected_class_mix)

    @property
    def has_preprocessing(self) -> bool:
        """전처리 통계가 함께 있는가.

        입력 채널이 있는데 정규화 통계가 없으면, 디바이스가 무엇을 하는지 모른다.
        """
        return bool(self.normalization) or not self.input_fields

    def describe(self) -> str:
        return (
            f"{self.artifact_id} ({self.label}, {self.size_bytes:,}B)  "
            f"기준 p95 {self.expected_p95_ms:.4f}ms / 정확도 {self.expected_accuracy:.4f}"
        )
