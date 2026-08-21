"""AI의 모든 판단을 로그로 남겨라. (실습 5-3)

예측만 남기면 아무것도 못 한다.
나중에 실제로 물어보게 되는 질문이 이런 것들이기 때문이다.

    "언제부터 이상해졌죠?"        → 시각이 필요하다
    "전부요, 아니면 한 대만요?"   → 디바이스가 필요하다
    "어느 모델이 낸 답이에요?"    → 배포 버전이 필요하다
    "느려진 건가요?"              → 지연시간이 필요하다
    "모델이 헷갈리기 시작한 건가요?" → 확신도가 필요하다
    "그때 그 입력 다시 넣어볼 수 있어요?" → 입력 지문이 필요하다

**이 여섯 개가 없으면 실습 5-4 부터 5-11 까지 전부 못 한다.**
로그는 나중에 추가할 수 없다. 지나간 시간은 다시 안 온다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class InferenceRecord:
    """추론 하나가 남긴 것."""

    occurred_at: str
    device_id: str
    deployment_version: int
    predicted_label: str
    confidence: float
    latency_ms: float
    input_digest: str = ""
    """입력 자체가 아니라 지문이다. 원본 신호를 전부 남길 수는 없다."""

    ground_truth: str | None = None
    """**대개 비어 있다.** 누군가 나중에 붙여 주기 전까지는 모른다."""

    def __post_init__(self) -> None:
        if not self.occurred_at.strip():
            raise InvariantViolation(
                "시각 없는 로그로는 '언제부터'에 답할 수 없다.", subject="occurred_at"
            )
        if not self.device_id.strip():
            raise InvariantViolation(
                "디바이스 없는 로그로는 '전부인가 한 대인가'에 답할 수 없다.",
                subject="device_id",
            )
        if self.deployment_version < 1:
            raise InvariantViolation(
                "배포 버전 없는 로그는 어느 모델이 낸 답인지 모른다.",
                subject="deployment_version",
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise InvariantViolation("confidence 는 0~1 이어야 한다.", subject="confidence")
        if self.latency_ms < 0:
            raise InvariantViolation("지연시간은 음수일 수 없다.", subject="latency_ms")

    @property
    def is_labeled(self) -> bool:
        return self.ground_truth is not None

    @property
    def is_correct(self) -> bool | None:
        """정답이 붙어 있을 때만 답할 수 있다."""
        if self.ground_truth is None:
            return None
        return self.ground_truth == self.predicted_label


@dataclass(frozen=True, slots=True)
class LogCoverage:
    """로그가 실제로 무엇을 담고 있는지 센 것.

    Infrastructure 가 세고, 아래 Policy 가 판단한다.
    """

    total_count: int
    with_timestamp: int
    with_device: int
    with_version: int
    with_confidence: int
    with_digest: int
    labeled_count: int
    distinct_devices: int
    distinct_versions: int

    def __post_init__(self) -> None:
        if self.total_count < 0:
            raise InvariantViolation("음수 개수는 없다.", subject="total_count")

    def _ratio(self, count: int) -> float:
        return count / self.total_count if self.total_count else 0.0

    @property
    def labeled_ratio(self) -> float:
        """정답이 붙은 비율. **현장에서는 대개 0에 가깝다.**"""
        return self._ratio(self.labeled_count)

    @property
    def digest_ratio(self) -> float:
        return self._ratio(self.with_digest)

    def describe(self) -> str:
        return (
            f"로그 {self.total_count:,}건  "
            f"디바이스 {self.distinct_devices}대  버전 {self.distinct_versions}종  "
            f"정답 {self.labeled_ratio:.1%}"
        )


@dataclass(frozen=True, slots=True)
class InferenceLogPolicy:
    """이 로그로 답할 수 있는 질문이 있는가.

    "로그를 남기고 있습니다"는 답이 아니다.
    **무엇을 남기고 있는가**가 답이다.
    """

    min_digest_ratio: float = 0.9
    min_labeled_ratio: float = 0.0
    """현장에서 정답이 붙는 비율. 0 이면 요구하지 않는다 — 그것이 정상이다."""

    def inspect(self, coverage: LogCoverage) -> tuple[Finding, ...]:
        if coverage.total_count == 0:
            return (
                Finding(
                    code="LOG_EMPTY",
                    message=(
                        "로그가 하나도 없다. 배포는 됐는데 아무것도 안 보고 있다는 뜻이다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="log",
                ),
            )

        findings: list[Finding] = []
        checks = (
            (
                "LOG_MISSING_TIMESTAMP",
                coverage.with_timestamp,
                "시각",
                "'언제부터 이상해졌는가'에 답할 수 없다 (실습 5-4).",
            ),
            (
                "LOG_MISSING_DEVICE",
                coverage.with_device,
                "디바이스",
                "전부 이상한 것인지 한 대만 이상한 것인지 구분할 수 없다.",
            ),
            (
                "LOG_MISSING_VERSION",
                coverage.with_version,
                "배포 버전",
                "어느 모델이 낸 답인지 모른다. 롤백해도 나아졌는지 알 수 없다.",
            ),
            (
                "LOG_MISSING_CONFIDENCE",
                coverage.with_confidence,
                "확신도",
                "모델이 헷갈리기 시작했는지 볼 수 없다 (실습 5-6).",
            ),
        )
        for code, present, name, consequence in checks:
            if present < coverage.total_count:
                findings.append(
                    Finding(
                        code=code,
                        message=f"{name} 가 빠진 로그가 있다. {consequence}",
                        severity=Severity.CRITICAL,
                        subject=name,
                        measured=present / coverage.total_count,
                        threshold=1.0,
                    )
                )

        if coverage.digest_ratio < self.min_digest_ratio:
            findings.append(
                Finding(
                    code="LOG_MISSING_DIGEST",
                    message=(
                        "입력 지문이 없다. 이상한 예측을 발견해도 "
                        "그때 무엇이 들어왔는지 되짚을 수 없다."
                    ),
                    severity=Severity.WARNING,
                    subject="input_digest",
                    measured=coverage.digest_ratio,
                    threshold=self.min_digest_ratio,
                )
            )

        if coverage.labeled_ratio < self.min_labeled_ratio:
            findings.append(
                Finding(
                    code="LOG_TOO_FEW_LABELS",
                    message=(
                        f"정답이 붙은 로그가 {coverage.labeled_ratio:.1%} 뿐이다. "
                        "현장 정확도를 말하려면 이 비율이 필요하다."
                    ),
                    severity=Severity.WARNING,
                    subject="ground_truth",
                    measured=coverage.labeled_ratio,
                    threshold=self.min_labeled_ratio,
                )
            )

        if coverage.distinct_versions > 1:
            findings.append(
                Finding(
                    code="LOG_MIXED_VERSIONS",
                    message=(
                        f"한 창에 배포 버전이 {coverage.distinct_versions}종 섞여 있다. "
                        "이 구간의 숫자는 어느 모델의 것도 아니다."
                    ),
                    severity=Severity.WARNING,
                    subject="deployment_version",
                    measured=float(coverage.distinct_versions),
                    threshold=1.0,
                )
            )

        return tuple(findings)


def summarize(records: Sequence[InferenceRecord]) -> LogCoverage:
    """로그 묶음을 세어 본다.

    이 함수는 Domain 에 있어도 된다 — 세는 것 말고는 아무것도 하지 않고,
    `InferenceRecord` 는 이미 Domain 의 것이기 때문이다.
    """
    return LogCoverage(
        total_count=len(records),
        with_timestamp=sum(1 for r in records if r.occurred_at),
        with_device=sum(1 for r in records if r.device_id),
        with_version=sum(1 for r in records if r.deployment_version > 0),
        with_confidence=len(records),
        with_digest=sum(1 for r in records if r.input_digest),
        labeled_count=sum(1 for r in records if r.is_labeled),
        distinct_devices=len({r.device_id for r in records}),
        distinct_versions=len({r.deployment_version for r in records}),
    )
