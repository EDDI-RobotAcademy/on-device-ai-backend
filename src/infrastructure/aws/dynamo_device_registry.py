"""DynamoDB 디바이스 레지스트리 어댑터. (실습 6-3)

`domain.fleet.ports.DeviceRegistry` 구현.

키 설계가 여기서도 문제다. S3 와 같은 이유다.

    PK = fleet_id          한 플릿의 디바이스를 한 번에 긁는다
    SK = device_id         한 대를 바로 찾는다

이렇게 두면 "이 플릿의 전 디바이스"가 **Query 한 번**이다.
PK 를 device_id 로 두면 같은 질문이 **Scan** 이 된다 —
1,000대면 1,000개를 다 읽고 필터링한다.

업링크 집계는 별도 테이블에 날짜별로 쌓는다.
`PK = fleet#device`, `SK = date` 로 두면 "오늘 얼마 올렸나"가 Query 한 번이다.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

from domain.fleet.device import Device, DeviceStatus
from domain.fleet.identifiers import DeviceId, FleetId
from domain.fleet.uplink import UplinkBatch
from infrastructure.aws.config import AwsConfig


class DynamoDeviceRegistry:
    """domain.fleet.ports.DeviceRegistry 구현."""

    def __init__(self, config: AwsConfig) -> None:
        self._config = config
        self._ddb = boto3.resource("dynamodb", **config.client_kwargs())
        self._devices = self._ddb.Table(config.device_table)
        self._uplinks = self._ddb.Table(config.uplink_table)

    # -- 준비 --------------------------------------------------------------
    def ensure_tables(self) -> None:
        """테이블이 없으면 만든다. 실습·테스트용 편의다.

        운영에서는 IaC 가 만든다. 여기서 만드는 이유는
        **키 설계를 코드로 보여주기 위해서**다 — 그게 이 실습의 요점이다.
        """
        client = self._ddb.meta.client
        for name, keys in (
            (
                self._config.device_table,
                [
                    {"AttributeName": "fleet_id", "KeyType": "HASH"},
                    {"AttributeName": "device_id", "KeyType": "RANGE"},
                ],
            ),
            (
                self._config.uplink_table,
                [
                    {"AttributeName": "device_key", "KeyType": "HASH"},
                    {"AttributeName": "date", "KeyType": "RANGE"},
                ],
            ),
        ):
            try:
                client.describe_table(TableName=name)
            except ClientError:
                client.create_table(
                    TableName=name,
                    KeySchema=keys,
                    AttributeDefinitions=[
                        {"AttributeName": key["AttributeName"], "AttributeType": "S"}
                        for key in keys
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
                client.get_waiter("table_exists").wait(TableName=name)

    # -- Port --------------------------------------------------------------
    def upsert(self, fleet_id: FleetId, device: Device) -> None:
        self._devices.put_item(
            Item={
                "fleet_id": str(fleet_id),
                "device_id": device.device_id,
                "group": device.group,
                "current_version": device.current_version,
                "last_seen_at": device.last_seen_at,
                "status": device.status.value,
                "site": device.site,
                "note": device.note,
            }
        )

    def find(self, fleet_id: FleetId, device_id: DeviceId) -> Device | None:
        response = self._devices.get_item(
            Key={"fleet_id": str(fleet_id), "device_id": str(device_id)}
        )
        item = response.get("Item")
        return _to_device(item) if item else None

    def list_devices(self, fleet_id: FleetId) -> Sequence[Device]:
        """이 플릿의 전 디바이스. **Query 한 번이다** — Scan 이 아니다."""
        from boto3.dynamodb.conditions import Key

        devices: list[Device] = []
        kwargs: dict[str, object] = {
            "KeyConditionExpression": Key("fleet_id").eq(str(fleet_id))
        }
        while True:
            response = self._devices.query(**kwargs)
            devices.extend(_to_device(item) for item in response.get("Items", []))
            token = response.get("LastEvaluatedKey")
            if not token:
                break
            kwargs["ExclusiveStartKey"] = token
        return tuple(devices)

    def record_uplink(self, fleet_id: FleetId, batch: UplinkBatch) -> None:
        """오늘 올린 양을 누적한다.

        읽고-더하고-쓰면 동시에 올라온 두 묶음 중 하나가 사라진다.
        DynamoDB 의 원자적 증가(ADD)를 쓴다 — **수천 대가 동시에 올린다.**
        """
        date = batch.window_start[:10]
        self._uplinks.update_item(
            Key={
                "device_key": f"{fleet_id}#{batch.device_id}",
                "date": date,
            },
            UpdateExpression="ADD payload_bytes :b, record_count :r",
            ExpressionAttributeValues={
                ":b": Decimal(batch.payload_bytes),
                ":r": Decimal(batch.record_count),
            },
        )

    def uplink_bytes_today(
        self, fleet_id: FleetId, device_id: DeviceId, date: str
    ) -> int:
        response = self._uplinks.get_item(
            Key={"device_key": f"{fleet_id}#{device_id}", "date": date}
        )
        item = response.get("Item")
        return int(item["payload_bytes"]) if item else 0


def _to_device(item: dict) -> Device:
    return Device(
        device_id=item["device_id"],
        group=item.get("group", "default"),
        current_version=item.get("current_version", ""),
        last_seen_at=item.get("last_seen_at", ""),
        status=DeviceStatus(item.get("status", DeviceStatus.HEALTHY.value)),
        site=item.get("site", ""),
        note=item.get("note", ""),
    )
