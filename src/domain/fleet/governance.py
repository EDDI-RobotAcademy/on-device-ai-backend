"""S3에 버전과 권한을 걸어라. (실습 6-13)

실습 6-2 는 "어디에 둘 것인가"를 정했다. 여기서는 **"어떻게 지킬 것인가"**를 정한다.

버킷 하나에 회사의 현장 데이터와 배포될 모델이 전부 들어 있다.
그런데 기본값 그대로 만든 버킷에는 이런 성질이 있다.

    같은 키에 다시 쓰면 **옛 것이 사라진다.**
        모델을 덮어썼다는 것을 아무도 모른 채 되돌릴 방법이 없어진다 (실습 6-9).

    보관 규칙이 없으면 **원본이 영원히 쌓인다.**
        3,000대가 하루 7 MiB 씩 올리면 1년에 7.5 TiB 다.

    권한이 넓으면 **되돌릴 수 없는 사고가 난다.**
        s3:* 를 준 역할 하나가 버킷을 통째로 지울 수 있다.

이 셋은 사고가 난 뒤에 켤 수 없다.
**버저닝은 켠 시점 이후의 객체만 지킨다.**
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class AccessStatement:
    """권한 한 줄. 누가 무엇에 무엇을 할 수 있는가."""

    sid: str
    effect: str
    principal: str
    actions: tuple[str, ...]
    resources: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.effect not in ("Allow", "Deny"):
            raise InvariantViolation(
                "효과는 Allow 또는 Deny 여야 한다.", subject="effect"
            )
        if not self.actions or not self.resources:
            raise InvariantViolation(
                "행위와 대상이 없으면 권한이 아니다.", subject=self.sid
            )

    @property
    def is_wildcard_action(self) -> bool:
        return any(action.endswith("*") and ":" in action for action in self.actions) or (
            "*" in self.actions
        )

    @property
    def is_wildcard_principal(self) -> bool:
        return self.principal.strip() == "*"

    def describe(self) -> str:
        return (
            f"{self.effect:<5} {self.principal:<28} "
            f"{','.join(self.actions):<24} {','.join(self.resources)}"
        )


@dataclass(frozen=True, slots=True)
class BucketGovernance:
    """버킷 하나의 실제 설정. Infrastructure 가 **읽어서** 채운다.

    선언한 값이 아니라 **지금 켜져 있는 값**이다.
    "설정했다"와 "설정되어 있다"는 다른 이야기다.
    """

    bucket: str
    versioning_enabled: bool = False
    encryption_algorithm: str | None = None
    public_access_blocked: bool = False
    lifecycle_expiration_days: int | None = None
    statements: tuple[AccessStatement, ...] = field(default_factory=tuple)
    object_version_counts: dict[str, int] = field(default_factory=dict)
    """키 → 남아 있는 버전 수. 버저닝을 켠 뒤에 쓴 것만 2 이상이 된다."""

    def __post_init__(self) -> None:
        if not self.bucket.strip():
            raise InvariantViolation("버킷 이름이 없다.", subject="bucket")

    @property
    def overwritten_keys(self) -> tuple[str, ...]:
        return tuple(
            key for key, count in sorted(self.object_version_counts.items()) if count > 1
        )

    def describe(self) -> str:
        return "\n".join(
            [
                f"[{self.bucket}]",
                f"  버저닝      {'켜짐' if self.versioning_enabled else '**꺼짐**'}",
                f"  암호화      {self.encryption_algorithm or '**없음**'}",
                f"  공개 차단   {'켜짐' if self.public_access_blocked else '**꺼짐**'}",
                f"  보관 규칙   "
                + (
                    f"{self.lifecycle_expiration_days}일"
                    if self.lifecycle_expiration_days
                    else "**없음**"
                ),
                f"  권한 {len(self.statements)}줄",
            ]
            + [f"    {s.describe()}" for s in self.statements]
        )


@dataclass(frozen=True, slots=True)
class GovernancePolicy:
    """이 버킷에 현장 데이터와 모델을 둬도 되는가. (실습 6-13)"""

    require_versioning: bool = True
    require_encryption: bool = True
    max_retention_days: int = 400

    def inspect(self, governance: BucketGovernance) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        if self.require_versioning and not governance.versioning_enabled:
            findings.append(
                Finding(
                    code="GOV_NO_VERSIONING",
                    message=(
                        "버저닝이 꺼져 있다. **같은 키에 다시 쓰면 옛 것이 사라진다** — "
                        "모델을 덮어쓴 사실을 아무도 모르고, 되돌릴 방법도 없다 (실습 6-9). "
                        "그리고 이건 **사고가 난 뒤에 켤 수 없다** — "
                        "버저닝은 켠 시점 이후의 객체만 지킨다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=governance.bucket,
                )
            )

        if self.require_encryption and not governance.encryption_algorithm:
            findings.append(
                Finding(
                    code="GOV_NO_ENCRYPTION",
                    message=(
                        "기본 암호화가 없다. 현장 신호에는 생산량과 가동 패턴이 들어 있다 — "
                        "**경쟁사가 읽으면 원가 구조가 보인다.**"
                    ),
                    severity=Severity.CRITICAL,
                    subject=governance.bucket,
                )
            )

        if not governance.public_access_blocked:
            findings.append(
                Finding(
                    code="GOV_PUBLIC_ACCESS_OPEN",
                    message=(
                        "공개 접근 차단이 꺼져 있다. "
                        "**정책 한 줄 실수로 버킷이 인터넷에 열린다** — "
                        "차단을 켜 두면 그 실수가 막힌다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=governance.bucket,
                )
            )

        if governance.lifecycle_expiration_days is None:
            findings.append(
                Finding(
                    code="GOV_NO_LIFECYCLE",
                    message=(
                        "보관 규칙이 없다. **원본은 영원히 쌓인다** — "
                        "3,000대가 하루 7 MiB 씩 올리면 1년에 7.5 TiB 다. "
                        "지울 것을 정하지 않으면 청구서가 대신 정한다."
                    ),
                    severity=Severity.WARNING,
                    subject=governance.bucket,
                )
            )
        elif governance.lifecycle_expiration_days > self.max_retention_days:
            findings.append(
                Finding(
                    code="GOV_RETENTION_TOO_LONG",
                    message=(
                        f"{governance.lifecycle_expiration_days}일을 보관한다. "
                        "**개인정보가 섞여 있다면 보관 기간 자체가 규제 대상이다.**"
                    ),
                    severity=Severity.WARNING,
                    subject=governance.bucket,
                    measured=float(governance.lifecycle_expiration_days),
                    threshold=float(self.max_retention_days),
                )
            )

        for statement in governance.statements:
            if statement.effect == "Allow" and statement.is_wildcard_principal:
                findings.append(
                    Finding(
                        code="GOV_PUBLIC_STATEMENT",
                        message=(
                            f"'{statement.sid}' 가 누구에게나(*) 허용한다. "
                            "**이 한 줄이 버킷을 인터넷에 연다.**"
                        ),
                        severity=Severity.CRITICAL,
                        subject=statement.sid,
                    )
                )
            elif statement.effect == "Allow" and statement.is_wildcard_action:
                findings.append(
                    Finding(
                        code="GOV_OVERBROAD_ACTION",
                        message=(
                            f"'{statement.sid}' 가 {','.join(statement.actions)} 를 허용한다. "
                            "**s3:* 에는 DeleteBucket 이 들어 있다** — "
                            "학습 잡이 쓰는 역할에 그 권한이 있을 이유가 없다."
                        ),
                        severity=Severity.WARNING,
                        subject=statement.sid,
                    )
                )

        if governance.overwritten_keys and not governance.versioning_enabled:
            findings.append(
                Finding(
                    code="GOV_SILENT_OVERWRITE",
                    message=(
                        f"{len(governance.overwritten_keys)}개 키가 덮어써졌다. "
                        "**버저닝이 꺼진 상태에서는 옛 내용이 이미 없다.**"
                    ),
                    severity=Severity.CRITICAL,
                    subject=governance.bucket,
                    measured=float(len(governance.overwritten_keys)),
                )
            )

        return tuple(findings)
