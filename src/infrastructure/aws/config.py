"""AWS 연결 설정.

자격증명을 코드에 두지 않는다. 환경변수나 인스턴스 역할에서 온다.
여기서 하는 일은 **어느 리전, 어느 버킷, 어느 테이블**을 한 군데 모으는 것뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AwsConfig:
    region: str = "ap-northeast-2"
    lake_bucket: str = "ondevice-ai-lake"
    artifact_bucket: str = "ondevice-ai-artifacts"
    device_table: str = "ondevice-ai-devices"
    uplink_table: str = "ondevice-ai-uplinks"
    training_role_arn: str = "arn:aws:iam::123456789012:role/ondevice-ai-training"
    training_image: str = "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/ondevice-ai:latest"
    endpoint_url: str | None = None
    """로컬 대체 구현(MinIO, LocalStack)을 붙일 때만 쓴다."""

    def client_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"region_name": self.region}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        return kwargs
