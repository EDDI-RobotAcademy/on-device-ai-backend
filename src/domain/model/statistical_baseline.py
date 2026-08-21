"""통계 기반 이상 탐지의 명세. (실습 3-13)

AI 를 쓰기 전에 반드시 물어야 하는 질문이 있다.

    **"이거, 통계로는 안 되나?"**

3-시그마 규칙은 1920년대 공장에서 쓰던 것이다. 학습이 필요 없고,
GPU 도 필요 없고, 왜 그렇게 판단했는지 한 줄로 설명된다.
그것으로 잡히는 문제에 신경망을 얹으면 유지비만 늘어난다.

그러나 통계에는 한계가 뚜렷하다.

    통계   "평소와 다르다"까지 말한다.
    AI     "**무엇이** 평소와 다른지"를 말한다.

유형을 구분해야 하는 문제라면 통계로는 안 된다.
이 실습은 그 경계를 숫자로 확인하는 자리다.

여기에는 numpy 가 없다. 계산은 Infrastructure 가 한다 (CLAUDE.md §14).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class DetectionMethod(Enum):
    """어떤 규칙으로 '평소와 다르다'를 정할 것인가."""

    THREE_SIGMA = "THREE_SIGMA"
    """평균에서 표준편차 k배를 넘으면 이상.

    정규분포를 가정한다. 현장 신호는 대개 정규분포가 아니다 —
    그래서 잘 맞을 때와 전혀 안 맞을 때가 갈린다.
    """

    IQR = "IQR"
    """사분위 범위 밖이면 이상.

    분포 모양을 가정하지 않는다. 치우친 신호에 3-시그마보다 낫다.
    """

    EWMA = "EWMA"
    """지수가중이동평균에서 벗어나면 이상.

    **기준선이 천천히 따라 움직인다.** 계절·교대 변화를 흡수한다.
    대신 천천히 나빠지는 고장은 기준선이 따라가면서 놓친다 (실습 5-7 과 같은 함정).
    """


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    """통계 검출기 하나의 설정."""

    method: DetectionMethod
    threshold: float = 3.0
    """THREE_SIGMA 면 표준편차 배수, IQR 이면 IQR 배수, EWMA 면 표준편차 배수."""

    smoothing: float = 0.2
    """EWMA 의 가중치. 클수록 최근을 더 본다."""

    min_flagged_ratio: float = 0.2
    """창 안에서 이 비율 이상이 걸리면 그 창을 '이상'으로 본다.

    한 표본이 튀었다고 창 전체를 이상이라 하면 오탐이 쏟아진다.
    """

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise InvariantViolation("기준은 0보다 커야 한다.", subject="threshold")
        if not 0.0 < self.smoothing <= 1.0:
            raise InvariantViolation(
                "EWMA 가중치는 0 초과 1 이하여야 한다.", subject="smoothing"
            )
        if not 0.0 < self.min_flagged_ratio <= 1.0:
            raise InvariantViolation(
                "비율은 0 초과 1 이하여야 한다.", subject="min_flagged_ratio"
            )

    def describe(self) -> str:
        return f"{self.method.value}(k={self.threshold:g}, 창 비율≥{self.min_flagged_ratio:g})"


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """통계 기준선과 학습 모델을 나란히 놓은 결과. (실습 3-13)

    비교는 **같은 문제로 접어서** 한다.
    통계는 유형을 말하지 못하므로, AI 쪽도 '이상 여부'로 접어서 재야 공평하다.
    """

    detector: str
    statistical_recall: float
    statistical_precision: float
    model_recall: float
    model_precision: float
    model_type_accuracy: float
    """AI 만 답할 수 있는 것 — 이상의 **유형**까지 맞힌 비율."""

    type_count: int

    def __post_init__(self) -> None:
        for name in (
            "statistical_recall",
            "statistical_precision",
            "model_recall",
            "model_precision",
            "model_type_accuracy",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 이어야 한다.", subject=name)

    @property
    def recall_gain(self) -> float:
        return self.model_recall - self.statistical_recall

    @property
    def precision_gain(self) -> float:
        return self.model_precision - self.statistical_precision

    def render(self) -> str:
        header = f"{'':<26}{'재현율':>10}{'정밀도':>10}"
        return "\n".join(
            [
                f"[이상 여부만 놓고 비교]  검출기 {self.detector}",
                header,
                "-" * len(header),
                f"{'통계 기반':<24}{self.statistical_recall:>10.3f}"
                f"{self.statistical_precision:>10.3f}",
                f"{'학습 모델':<24}{self.model_recall:>10.3f}"
                f"{self.model_precision:>10.3f}",
                "-" * len(header),
                f"{'차이':<24}{self.recall_gain:>+10.3f}{self.precision_gain:>+10.3f}",
                "",
                f"  이상의 **유형**까지 맞힌 비율: {self.model_type_accuracy:.3f} "
                f"(유형 {self.type_count}종)",
                "  통계 기반은 이 칸을 채울 수 없다 — 유형이라는 개념이 없다.",
            ]
        )


@dataclass(frozen=True, slots=True)
class BaselineJustificationPolicy:
    """AI 를 쓸 근거가 있는가. (실습 3-13)

    "정확도가 더 높다"는 근거가 아니다.
    **얼마나 더 높은지**와 **그 대가가 얼마인지**가 근거다.
    """

    min_recall_gain: float = 0.05
    """이만큼도 못 이기면 통계로 충분하다."""

    def inspect(self, comparison: BaselineComparison) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        if comparison.recall_gain < self.min_recall_gain:
            severity = (
                Severity.CRITICAL if comparison.type_count <= 1 else Severity.WARNING
            )
            findings.append(
                Finding(
                    code="BASELINE_NOT_BEATEN",
                    message=(
                        f"학습 모델이 통계 기준선을 재현율에서 "
                        f"{comparison.recall_gain:+.3f} 밖에 못 이겼다. "
                        "**이 정도면 3-시그마 한 줄로 끝낼 문제다** — "
                        "신경망은 학습·변환·배포·재학습 비용을 평생 데리고 다닌다."
                    ),
                    severity=severity,
                    subject=comparison.detector,
                    measured=comparison.recall_gain,
                    threshold=self.min_recall_gain,
                )
            )

        if comparison.type_count > 1 and comparison.model_type_accuracy > 0.0:
            findings.append(
                Finding(
                    code="BASELINE_CANNOT_TYPE",
                    message=(
                        f"이상의 유형이 {comparison.type_count}종이다. "
                        f"모델은 유형까지 {comparison.model_type_accuracy:.1%} 맞혔고, "
                        "**통계 기반은 이 질문 자체에 답할 수 없다.** "
                        "AI 를 쓸 근거는 정확도가 아니라 여기에 있다."
                    ),
                    severity=Severity.INFO,
                    subject=comparison.detector,
                    measured=comparison.model_type_accuracy,
                )
            )

        if comparison.statistical_precision < 0.5 and comparison.model_precision > (
            comparison.statistical_precision + 0.1
        ):
            findings.append(
                Finding(
                    code="BASELINE_TOO_NOISY",
                    message=(
                        f"통계 기준선의 정밀도가 {comparison.statistical_precision:.3f} 다. "
                        "**절반 이상이 헛알람이면 현장은 알람을 꺼 버린다** — "
                        "그러면 재현율이 아무리 높아도 소용이 없다."
                    ),
                    severity=Severity.WARNING,
                    subject=comparison.detector,
                    measured=comparison.statistical_precision,
                    threshold=0.5,
                )
            )

        return tuple(findings)
