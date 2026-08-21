"""실험 기록을 S3 에 남기고 다시 읽는다. (실습 6-12)

노트북이 아니라 **S3 에** 둔다. 이유는 하나다.
사람이 바뀌어도 남아야 하기 때문이다.

키 구조는 실습 6-2 의 규칙을 그대로 따른다.

    experiments/{실험}/trials/{시행}/record.json

자주 걸러내는 것(실험)을 앞에 둔다. 그래야 "이 실험의 시행 전부"가 접두어로 좁혀진다.
"""

from __future__ import annotations

import json

import boto3
from botocore.exceptions import ClientError

from domain.fleet.experiment_record import ExperimentLedger, ExperimentRecord
from infrastructure.aws.config import AwsConfig


class S3ExperimentStore:
    """domain.fleet.ports.ExperimentStore 구현."""

    def __init__(self, config: AwsConfig) -> None:
        self._s3 = boto3.client("s3", **config.client_kwargs())
        # 데이터 레이크와 같은 버킷을 쓴다. 접두어로 나눈다 (실습 6-2).
        self._bucket = config.lake_bucket

    def record(self, entry: ExperimentRecord) -> str:
        self._s3.put_object(
            Bucket=self._bucket,
            Key=entry.key,
            Body=json.dumps(
                {
                    "experiment_id": entry.experiment_id,
                    "trial_id": entry.trial_id,
                    "dataset_version": entry.dataset_version,
                    "code_version": entry.code_version,
                    "parameters": dict(entry.parameters),
                    "metrics": dict(entry.metrics),
                    "artifact_uri": entry.artifact_uri,
                    "created_at": entry.created_at,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8"),
            ContentType="application/json",
        )
        return entry.key

    def load(self, experiment_id: str) -> ExperimentLedger:
        """기록을 읽고, **가리키는 아티팩트가 실제로 있는지도 확인한다.**

        기록만 있고 파일이 없는 경우가 실제로 생긴다 —
        수명 주기 규칙이 지웠거나, 다른 계정으로 옮겼거나 (실습 6-13).
        """
        prefix = f"experiments/{experiment_id}/trials/"
        records: list[ExperimentRecord] = []
        missing: list[str] = []

        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                if not item["Key"].endswith("record.json"):
                    continue
                body = self._s3.get_object(Bucket=self._bucket, Key=item["Key"])[
                    "Body"
                ].read()
                payload = json.loads(body)
                entry = ExperimentRecord(
                    experiment_id=payload["experiment_id"],
                    trial_id=payload["trial_id"],
                    dataset_version=payload.get("dataset_version", ""),
                    code_version=payload.get("code_version", ""),
                    parameters=payload.get("parameters", {}),
                    metrics=payload.get("metrics", {}),
                    artifact_uri=payload.get("artifact_uri", ""),
                    created_at=payload.get("created_at", ""),
                )
                records.append(entry)
                if entry.artifact_uri and not self._exists(entry.artifact_uri):
                    missing.append(entry.trial_id)

        records.sort(key=lambda r: r.trial_id)
        return ExperimentLedger(
            experiment_id=experiment_id,
            records=tuple(records),
            missing_artifacts=tuple(missing),
        )

    def _exists(self, uri: str) -> bool:
        if not uri.startswith("s3://"):
            return True  # 우리 버킷이 아니면 판단하지 않는다
        _, _, remainder = uri.partition("s3://")
        bucket, _, key = remainder.partition("/")
        try:
            self._s3.head_object(Bucket=bucket, Key=key)
        except ClientError:
            return False
        return True
