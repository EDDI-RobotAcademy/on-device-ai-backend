"""버킷의 버전·암호화·권한을 **실제로 걸고 실제로 읽는다.** (실습 6-13)

가짜 클라이언트를 세워 두고 "호출됐다"만 확인하면
API 이름이 틀려도 통과한다. 여기서는 진짜 boto3 요청을 만든다.

읽는 쪽이 더 중요하다. **"설정했다"와 "설정되어 있다"는 다른 이야기다** —
누군가 콘솔에서 껐을 수도 있고, 코드가 다른 버킷에 걸었을 수도 있다.
"""

from __future__ import annotations

import json

import boto3
from botocore.exceptions import ClientError

from domain.fleet.governance import AccessStatement, BucketGovernance
from infrastructure.aws.config import AwsConfig


class S3Governance:
    """domain.fleet.ports.BucketGovernanceGateway 구현."""

    def __init__(self, config: AwsConfig) -> None:
        self._s3 = boto3.client("s3", **config.client_kwargs())
        self._bucket = config.lake_bucket

    def harden(
        self,
        *,
        versioning: bool = True,
        encryption: str | None = "AES256",
        block_public_access: bool = True,
        expiration_days: int | None = 365,
    ) -> None:
        """되돌릴 수 없는 사고를 막는 설정을 건다. **데이터가 들어오기 전에.**"""
        if versioning:
            self._s3.put_bucket_versioning(
                Bucket=self._bucket,
                VersioningConfiguration={"Status": "Enabled"},
            )
        if encryption:
            self._s3.put_bucket_encryption(
                Bucket=self._bucket,
                ServerSideEncryptionConfiguration={
                    "Rules": [
                        {
                            "ApplyServerSideEncryptionByDefault": {
                                "SSEAlgorithm": encryption
                            }
                        }
                    ]
                },
            )
        if block_public_access:
            self._s3.put_public_access_block(
                Bucket=self._bucket,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )
        if expiration_days:
            self._s3.put_bucket_lifecycle_configuration(
                Bucket=self._bucket,
                LifecycleConfiguration={
                    "Rules": [
                        {
                            "ID": "raw-sample-expiry",
                            "Status": "Enabled",
                            "Filter": {"Prefix": "uplinks/kind=raw_sample/"},
                            "Expiration": {"Days": expiration_days},
                        }
                    ]
                },
            )

    def put_policy(self, statements: tuple[AccessStatement, ...]) -> None:
        self._s3.put_bucket_policy(
            Bucket=self._bucket,
            Policy=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": s.sid,
                            "Effect": s.effect,
                            "Principal": s.principal
                            if s.principal == "*"
                            else {"AWS": s.principal},
                            "Action": list(s.actions),
                            "Resource": list(s.resources),
                        }
                        for s in statements
                    ],
                }
            ),
        )

    def inspect(self, *, version_prefix: str = "") -> BucketGovernance:
        """지금 켜져 있는 것을 **읽는다.** 선언한 값이 아니다."""
        return BucketGovernance(
            bucket=self._bucket,
            versioning_enabled=self._versioning(),
            encryption_algorithm=self._encryption(),
            public_access_blocked=self._public_access_blocked(),
            lifecycle_expiration_days=self._lifecycle_days(),
            statements=self._statements(),
            object_version_counts=self._version_counts(version_prefix),
        )

    # -- 내부 --------------------------------------------------------------
    def _versioning(self) -> bool:
        response = self._s3.get_bucket_versioning(Bucket=self._bucket)
        return response.get("Status") == "Enabled"

    def _encryption(self) -> str | None:
        try:
            response = self._s3.get_bucket_encryption(Bucket=self._bucket)
        except ClientError:
            return None
        rules = response["ServerSideEncryptionConfiguration"]["Rules"]
        if not rules:
            return None
        return rules[0]["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"]

    def _public_access_blocked(self) -> bool:
        try:
            block = self._s3.get_public_access_block(Bucket=self._bucket)
        except ClientError:
            return False
        config = block["PublicAccessBlockConfiguration"]
        return all(
            config.get(key, False)
            for key in (
                "BlockPublicAcls",
                "IgnorePublicAcls",
                "BlockPublicPolicy",
                "RestrictPublicBuckets",
            )
        )

    def _lifecycle_days(self) -> int | None:
        try:
            response = self._s3.get_bucket_lifecycle_configuration(Bucket=self._bucket)
        except ClientError:
            return None
        for rule in response.get("Rules", []):
            days = rule.get("Expiration", {}).get("Days")
            if days:
                return int(days)
        return None

    def _statements(self) -> tuple[AccessStatement, ...]:
        try:
            raw = self._s3.get_bucket_policy(Bucket=self._bucket)["Policy"]
        except ClientError:
            return ()
        document = json.loads(raw)
        parsed: list[AccessStatement] = []
        for statement in document.get("Statement", []):
            principal = statement.get("Principal", "*")
            if isinstance(principal, dict):
                principal = principal.get("AWS", "*")
            parsed.append(
                AccessStatement(
                    sid=statement.get("Sid", "(이름 없음)"),
                    effect=statement["Effect"],
                    principal=str(principal),
                    actions=tuple(_as_tuple(statement.get("Action", ()))),
                    resources=tuple(_as_tuple(statement.get("Resource", ()))),
                )
            )
        return tuple(parsed)

    def _version_counts(self, prefix: str) -> dict[str, int]:
        """키마다 남아 있는 버전 수. **덮어쓴 흔적이 여기 남는다.**"""
        counts: dict[str, int] = {}
        paginator = self._s3.get_paginator("list_object_versions")
        try:
            pages = paginator.paginate(Bucket=self._bucket, Prefix=prefix)
            for page in pages:
                for version in page.get("Versions", []):
                    counts[version["Key"]] = counts.get(version["Key"], 0) + 1
        except ClientError:
            return {}
        return counts


def _as_tuple(value) -> tuple[str, ...]:  # noqa: ANN001
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)
