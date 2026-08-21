"""Cloud에서 만든 모델을 Edge로 보내라, 그리고 Version을 관리하라. (실습 6-6, 6-7)

모듈 5 에서 배웠다 — **결과물 파일만 보내면 조용히 다른 모델이 된다.**
전처리 통계도, 기준 숫자도 함께 가야 한다.

수천 대에 보낼 때는 거기에 두 가지가 더 붙는다.

    체크섬   좁은 회선에서 잘려 도착한 파일을 온전한 것으로 착각하지 않기 위해
    크기     디바이스 플래시에 들어가는지, 그리고 3,000대 × 파일 크기가 전송 예산에 맞는지

그래서 **릴리스는 파일이 아니라 묶음(bundle)**이다.

그리고 채널.

    canary   몇 대에만. 문제가 나도 몇 대다.
    stable   전부. 여기까지 온 것은 canary 를 통과한 것이다.

**한 채널에 두 버전이 동시에 있을 수 없다.**
그것을 허용하는 순간 "지금 stable 이 뭐죠?"에 답할 수 없게 된다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity, Verdict, derive_verdict


class ReleaseChannel(Enum):
    CANARY = "CANARY"
    """몇 대에만 내보낸다. 문제가 나도 몇 대다."""

    STABLE = "STABLE"
    """전부에게. canary 를 통과한 것만 온다."""

    ARCHIVED = "ARCHIVED"
    """더 이상 내보내지 않는다. **지우지는 않는다** — 롤백 대상일 수 있다."""


@dataclass(frozen=True, slots=True)
class ReleaseBundle:
    """디바이스로 내보낼 수 있게 묶인 한 덩어리.

    모듈 5 의 `DeployedArtifactRef` 가 요구하던 것들이 전부 여기 들어 있다.
    다른 점은 **회선을 건너간다**는 것뿐이고, 그래서 체크섬과 크기가 붙는다.
    """

    release_id: str
    version: str
    model_version_id: str
    artifact_uri: str
    artifact_bytes: int
    checksum: str
    runtime: str
    precision: str
    class_labels: tuple[str, ...]
    input_fields: tuple[str, ...] = field(default_factory=tuple)
    normalization: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    expected_p95_ms: float = 0.0
    expected_class_mix: Mapping[str, float] = field(default_factory=dict)
    sample_interval_seconds: int = 0
    """**표본 간격도 모델의 일부다** (실습 5-1). 디바이스가 이 값으로 솎아 넣는다."""

    window_length: int = 0
    channel: ReleaseChannel = ReleaseChannel.CANARY
    built_at: str = ""
    source_build_id: str = ""
    """어느 학습 데이터셋에서 왔는가. 계보의 한 칸이다 (실습 6-10)."""

    source_job_id: str = ""

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise InvariantViolation(
                "버전 없는 릴리스는 디바이스가 구분할 수 없다.", subject="version"
            )
        if self.artifact_bytes <= 0:
            raise InvariantViolation("빈 결과물은 보낼 수 없다.", subject="artifact_bytes")
        if not self.checksum.strip():
            raise InvariantViolation(
                "체크섬이 없으면 잘려 도착한 파일을 알아챌 수 없다.", subject="checksum"
            )
        if len(self.class_labels) < 2:
            raise InvariantViolation("클래스가 둘 미만이다.", subject="class_labels")

    @property
    def artifact_kib(self) -> float:
        return self.artifact_bytes / 1024

    @property
    def has_preprocessing(self) -> bool:
        """전처리가 함께 묶였는가. **전처리는 모델의 일부다** (실습 5-1)."""
        if not self.input_fields:
            return False
        return bool(self.normalization) and self.sample_interval_seconds > 0

    @property
    def has_baseline(self) -> bool:
        return self.expected_p95_ms > 0 and bool(self.expected_class_mix)

    def fleet_transfer_kib(self, device_count: int) -> float:
        """이 릴리스를 전부에게 보내면 얼마나 나가는가.

        한 대에 12KiB 면 작아 보인다. 3,000대면 36MiB 다.
        """
        return self.artifact_kib * device_count

    def describe(self) -> str:
        return (
            f"{self.version} ({self.runtime}/{self.precision}, "
            f"{self.artifact_bytes:,}B, {self.channel.value})"
        )


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    bundle: ReleaseBundle
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> Verdict:
        return derive_verdict(self.findings)

    @property
    def can_publish(self) -> bool:
        return self.verdict is not Verdict.FAILED

    def render(self) -> str:
        lines = [
            f"릴리스 점검: {self.verdict.value}",
            f"  {self.bundle.describe()}",
        ]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        else:
            lines.append("  걸리는 것이 없다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ReleasePolicy:
    """이 묶음을 수천 대에 보내도 되는가."""

    max_artifact_kib: float = 256.0
    """디바이스 플래시 상한. 모듈 4 의 DeviceBudget.storage_kib 과 같은 숫자여야 한다."""

    max_fleet_transfer_mib: float = 512.0
    """전체에 뿌릴 때의 전송량 상한. **한 대 크기 × 대수**다."""

    require_preprocessing: bool = True
    require_baseline: bool = True
    require_lineage: bool = True
    """어느 데이터에서 왔는지 없으면 나중에 되짚을 수 없다 (실습 6-10)."""

    def inspect(
        self, bundle: ReleaseBundle, *, device_count: int = 1
    ) -> ReleaseCheck:
        findings: list[Finding] = []

        if bundle.artifact_kib > self.max_artifact_kib:
            findings.append(
                Finding(
                    code="RELEASE_TOO_LARGE",
                    message="디바이스 플래시에 안 들어간다.",
                    severity=Severity.CRITICAL,
                    subject=bundle.version,
                    measured=bundle.artifact_kib,
                    threshold=self.max_artifact_kib,
                )
            )

        transfer_mib = bundle.fleet_transfer_kib(device_count) / 1024
        if transfer_mib > self.max_fleet_transfer_mib:
            findings.append(
                Finding(
                    code="RELEASE_FLEET_TRANSFER_TOO_LARGE",
                    message=(
                        f"{device_count:,}대에 보내면 {transfer_mib:,.1f}MiB 가 나간다. "
                        "**한 대 기준으로 보면 작아 보이는 것이 대수를 곱하면 달라진다.**"
                    ),
                    severity=Severity.WARNING,
                    subject=bundle.version,
                    measured=transfer_mib,
                    threshold=self.max_fleet_transfer_mib,
                )
            )

        if self.require_preprocessing and not bundle.has_preprocessing:
            findings.append(
                Finding(
                    code="RELEASE_NO_PREPROCESSING",
                    message=(
                        "전처리가 묶이지 않았다 — 정규화 통계나 표본 간격이 없다. "
                        "**전처리는 모델의 일부다** (실습 5-1)."
                    ),
                    severity=Severity.CRITICAL,
                    subject="preprocessing",
                )
            )

        if self.require_baseline and not bundle.has_baseline:
            findings.append(
                Finding(
                    code="RELEASE_NO_BASELINE",
                    message=(
                        "기준 숫자가 없다. 내보낸 뒤 '느려졌다'를 말할 근거가 없어진다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="baseline",
                )
            )

        if self.require_lineage and not bundle.source_build_id:
            findings.append(
                Finding(
                    code="RELEASE_NO_LINEAGE",
                    message=(
                        "어느 학습 데이터셋에서 왔는지 없다. "
                        "6개월 뒤 '이 모델 뭐로 만들었죠?'에 답할 수 없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="lineage",
                )
            )

        return ReleaseCheck(bundle=bundle, findings=tuple(findings))


@dataclass(frozen=True, slots=True)
class ChannelState:
    """지금 각 채널에 무엇이 올라가 있는가.

    **한 채널에 하나뿐이다.** 두 개를 허용하면 "지금 stable 이 뭐죠?"에 답할 수 없다.
    """

    canary: str = ""
    stable: str = ""
    archived: tuple[str, ...] = field(default_factory=tuple)

    def current(self, channel: ReleaseChannel) -> str:
        if channel is ReleaseChannel.CANARY:
            return self.canary
        if channel is ReleaseChannel.STABLE:
            return self.stable
        return ""

    def with_promotion(self, version: str, channel: ReleaseChannel) -> ChannelState:
        """새 버전을 채널에 올린다. 있던 것은 밀려난다."""
        if channel is ReleaseChannel.CANARY:
            displaced = self.canary
            return ChannelState(
                canary=version,
                stable=self.stable,
                archived=self.archived + ((displaced,) if displaced else ()),
            )
        if channel is ReleaseChannel.STABLE:
            displaced = self.stable
            return ChannelState(
                canary="" if self.canary == version else self.canary,
                stable=version,
                archived=self.archived + ((displaced,) if displaced else ()),
            )
        raise InvariantViolation(
            "ARCHIVED 로는 승격하지 않는다.", subject="channel"
        )

    def describe(self) -> str:
        return (
            f"canary={self.canary or '(없음)'}  "
            f"stable={self.stable or '(없음)'}  "
            f"archived={len(self.archived)}개"
        )
