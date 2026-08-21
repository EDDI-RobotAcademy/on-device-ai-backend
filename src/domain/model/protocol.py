"""Training 과 Validation 을 분리한다는 것. (실습 3-8)

분할은 실습 1-8 에서 이미 했다. 그런데 분할해 놓고도 새는 경로가 남아 있다.

    1. 검증 집합을 보고 하이퍼파라미터를 고른다
       → 검증 집합이 사실상 학습에 쓰인 것이다. 그 점수는 더 이상 '미지'가 아니다.

    2. 같은 검증 집합으로 20번 실험하고 가장 좋은 것을 보고한다
       → 20번 중 하나는 우연히 잘 나온다. 그 우연을 성능이라고 부른다.

    3. test 집합을 중간에 한 번 본다
       → 그 순간 test 는 두 번째 validation 이 된다. 되돌릴 수 없다.

그래서 분할은 '나눴다'로 끝나지 않는다. **어떻게 쓸 것인지까지 계약이어야 한다.**
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class SplitUsage:
    """각 분할을 실제로 몇 번 들여다봤는가. Application 이 세어 준다."""

    train_sample_count: int
    validation_sample_count: int
    test_sample_count: int
    validation_evaluations: int = 0
    """검증 집합으로 점수를 낸 횟수 = 하이퍼파라미터를 고른 횟수."""

    test_evaluations: int = 0
    """test 집합으로 점수를 낸 횟수. 1을 넘으면 곤란하다."""

    overlapping_samples: int = 0
    """분할 경계에서 두 쪽이 공유하는 원본 표본 수.

    창이 겹치면(실습 3-4) 나눠도 겹친다. 그 사실이 여기 남는다."""

    def __post_init__(self) -> None:
        for name in (
            "train_sample_count",
            "validation_sample_count",
            "test_sample_count",
            "validation_evaluations",
            "test_evaluations",
            "overlapping_samples",
        ):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)

    @property
    def total_sample_count(self) -> int:
        return (
            self.train_sample_count
            + self.validation_sample_count
            + self.test_sample_count
        )

    def ratio_of(self, split: str) -> float:
        total = self.total_sample_count
        if total == 0:
            return 0.0
        return getattr(self, f"{split}_sample_count") / total


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """분할을 어떻게 쓸 것인지에 대한 계약."""

    min_validation_samples: int = 50
    min_test_samples: int = 50
    max_validation_evaluations: int = 20
    """검증 집합을 이보다 많이 보면 다중 검정 문제가 생긴다."""

    max_test_evaluations: int = 1
    """test 는 마지막에 딱 한 번. 그것이 test 인 이유다."""

    require_disjoint: bool = True

    def inspect(self, usage: SplitUsage) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        overlapping_samples = usage.overlapping_samples

        if self.require_disjoint and overlapping_samples > 0:
            findings.append(
                Finding(
                    code="PROTOCOL_SPLIT_OVERLAP",
                    message=(
                        f"분할 사이에 {overlapping_samples} 개 표본이 겹친다. "
                        "창이 겹치면(실습 3-4) 나눠도 겹친다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="split",
                    measured=float(overlapping_samples),
                    threshold=0.0,
                )
            )

        if usage.validation_sample_count < self.min_validation_samples:
            findings.append(
                Finding(
                    code="PROTOCOL_VALIDATION_TOO_SMALL",
                    message=(
                        "검증 집합이 너무 작다. 이 표본 수로 낸 점수는 "
                        "다음 실험에서 그대로 재현되지 않는다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="validation",
                    measured=float(usage.validation_sample_count),
                    threshold=float(self.min_validation_samples),
                )
            )

        if usage.test_sample_count < self.min_test_samples:
            findings.append(
                Finding(
                    code="PROTOCOL_TEST_TOO_SMALL",
                    message="평가 집합이 너무 작아 최종 보고 숫자를 신뢰할 수 없다.",
                    severity=Severity.CRITICAL,
                    subject="test",
                    measured=float(usage.test_sample_count),
                    threshold=float(self.min_test_samples),
                )
            )

        if usage.validation_evaluations > self.max_validation_evaluations:
            findings.append(
                Finding(
                    code="PROTOCOL_VALIDATION_OVERUSED",
                    message=(
                        f"검증 집합으로 {usage.validation_evaluations} 번 점수를 냈다. "
                        "그중 가장 좋은 것을 고르면, 그 점수에는 우연이 섞여 있다."
                    ),
                    severity=Severity.WARNING,
                    subject="validation",
                    measured=float(usage.validation_evaluations),
                    threshold=float(self.max_validation_evaluations),
                )
            )

        if usage.test_evaluations > self.max_test_evaluations:
            findings.append(
                Finding(
                    code="PROTOCOL_TEST_REUSED",
                    message=(
                        f"평가 집합을 {usage.test_evaluations} 번 봤다. "
                        "두 번째부터 그것은 test 가 아니라 두 번째 validation 이다. "
                        "되돌릴 수 없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="test",
                    measured=float(usage.test_evaluations),
                    threshold=float(self.max_test_evaluations),
                )
            )

        return tuple(findings)
