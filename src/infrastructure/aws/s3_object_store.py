"""S3 객체 저장소 어댑터. (실습 6-1, 6-2)

`domain.fleet.ports.ObjectStore` 구현.

Domain 은 `ObjectKey` 를 준다. 이 어댑터가 그것을 S3 키 문자열로 바꾼다.
**그 변환이 이 파일의 전부다.** 파티션 설계는 Domain 이 정한다.
"""

from __future__ import annotations

from collections.abc import Sequence

import boto3
from botocore.exceptions import ClientError

from domain.fleet.object_key import ObjectKey, ObjectStats
from infrastructure.aws.config import AwsConfig
from infrastructure.errors import InfrastructureException


class ObjectNotFound(InfrastructureException):
    code = "OBJECT_NOT_FOUND"


class S3ObjectStore:
    """domain.fleet.ports.ObjectStore 구현."""

    def __init__(self, config: AwsConfig, *, bucket: str | None = None) -> None:
        self._config = config
        self._bucket = bucket or config.lake_bucket
        self._s3 = boto3.client("s3", **config.client_kwargs())

    # -- 준비 --------------------------------------------------------------
    def ensure_bucket(self) -> str:
        """버킷이 없으면 만든다. 실습·테스트용 편의다.

        실제 운영에서는 IaC(Terraform/CDK)가 만든다 —
        애플리케이션이 자기 인프라를 만드는 것은 권한을 너무 많이 요구한다.
        """
        try:
            self._s3.head_bucket(Bucket=self._bucket)
        except ClientError:
            kwargs: dict[str, object] = {"Bucket": self._bucket}
            if self._config.region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._config.region
                }
            self._s3.create_bucket(**kwargs)
        return self._bucket

    # -- Port --------------------------------------------------------------
    def put(self, key: ObjectKey, body: bytes) -> str:
        self._s3.put_object(Bucket=self._bucket, Key=key.render(), Body=body)
        return f"s3://{self._bucket}/{key.render()}"

    def get(self, key: ObjectKey) -> bytes:
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=key.render())
        except ClientError as exc:  # noqa: PERF203 - 경계에서 번역한다
            raise ObjectNotFound(
                f"객체가 없다: {key.render()}", subject=key.render()
            ) from exc
        return response["Body"].read()

    def list_prefix(self, prefix: str) -> Sequence[str]:
        """접두어로 좁혀 목록을 가져온다.

        **접두어가 곧 비용이다.** 좁히지 못하면 버킷 전체를 페이지 넘겨 가며 훑는다.
        """
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        return tuple(keys)

    def stats(self, prefix: str) -> ObjectStats:
        """이 접두어 아래가 어떻게 생겼는지 센다.

        객체 수, 합계 크기, 그리고 **서로 다른 파티션 접두어 수**.
        마지막 것이 작은 파일 문제를 드러낸다.
        """
        sizes: list[int] = []
        prefixes: set[str] = set()
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                sizes.append(item["Size"])
                prefixes.add(item["Key"].rsplit("/", 1)[0])

        return ObjectStats(
            object_count=len(sizes),
            total_bytes=sum(sizes),
            distinct_prefixes=len(prefixes),
            smallest_bytes=min(sizes) if sizes else 0,
            largest_bytes=max(sizes) if sizes else 0,
        )

    # -- 편의 --------------------------------------------------------------
    @property
    def bucket(self) -> str:
        return self._bucket

    def uri_of(self, key: ObjectKey | str) -> str:
        rendered = key if isinstance(key, str) else key.render()
        return f"s3://{self._bucket}/{rendered}"
