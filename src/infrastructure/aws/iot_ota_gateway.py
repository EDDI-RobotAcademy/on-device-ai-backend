"""AWS IoT Jobs OTA 어댑터. (실습 6-8, 6-9)

`domain.fleet.ports.OtaGateway` 구현.

IoT Jobs 의 모양이 이 Domain 과 잘 맞는다.

    Job          한 번의 롤아웃
    Job Document 무엇을 받아야 하는지 (릴리스 번들의 위치와 체크섬)
    Job Execution 디바이스 한 대의 결과

그리고 **디바이스가 스스로 말한 것만 결과로 센다.**
서버가 "보냈으니 됐겠지"라고 적으면 실제로 안 올라간 대수를 영영 모른다.

응답이 없는 디바이스는 PENDING 이다. **실패가 아니다** —
꺼져 있는 것과 설치에 실패한 것은 다른 사건이고, 다른 조치를 요구한다.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import boto3
from botocore.exceptions import ClientError

from domain.fleet.identifiers import RolloutId
from domain.fleet.release import ReleaseBundle
from domain.fleet.rollout import DeviceOutcome
from infrastructure.aws.config import AwsConfig

OUTCOME_MAP: dict[str, DeviceOutcome] = {
    "QUEUED": DeviceOutcome.PENDING,
    "IN_PROGRESS": DeviceOutcome.PENDING,
    "SUCCEEDED": DeviceOutcome.SUCCEEDED,
    "FAILED": DeviceOutcome.FAILED,
    "REJECTED": DeviceOutcome.FAILED,
    "TIMED_OUT": DeviceOutcome.UNREACHABLE,
    "REMOVED": DeviceOutcome.SKIPPED,
    "CANCELED": DeviceOutcome.SKIPPED,
}


class IotJobsOtaGateway:
    """domain.fleet.ports.OtaGateway 구현."""

    def __init__(self, config: AwsConfig, *, account_id: str = "123456789012") -> None:
        self._config = config
        self._account_id = account_id
        self._iot = boto3.client("iot", **config.client_kwargs())

    # -- 준비 --------------------------------------------------------------
    def ensure_things(self, device_ids: Sequence[str]) -> int:
        """디바이스를 IoT 사물로 등록한다. 실습·테스트용 편의다."""
        created = 0
        for device_id in device_ids:
            try:
                self._iot.describe_thing(thingName=device_id)
            except ClientError:
                self._iot.create_thing(thingName=device_id)
                created += 1
        return created

    # -- Port --------------------------------------------------------------
    def announce(
        self, rollout_id: RolloutId, bundle: ReleaseBundle, device_ids: Sequence[str]
    ) -> str:
        """새 버전이 있다고 알린다.

        문서에는 **디바이스가 검증할 수 있는 것**을 넣는다 —
        위치와 체크섬. 그래야 잘려 도착한 파일을 스스로 걸러낸다.
        """
        job_id = self._job_id(rollout_id, bundle.version)
        document = {
            "operation": "install-model",
            "version": bundle.version,
            "artifactUri": bundle.artifact_uri,
            "checksum": bundle.checksum,
            "sizeBytes": bundle.artifact_bytes,
            "runtime": bundle.runtime,
            "precision": bundle.precision,
            # **전처리는 모델의 일부다** (실습 5-1, 6-6)
            "normalization": {
                name: list(stats) for name, stats in bundle.normalization.items()
            },
            "inputFields": list(bundle.input_fields),
            "sampleIntervalSeconds": bundle.sample_interval_seconds,
            "windowLength": bundle.window_length,
        }
        self._iot.create_job(
            jobId=job_id,
            targets=[self._thing_arn(device_id) for device_id in device_ids],
            document=json.dumps(document, ensure_ascii=False),
            targetSelection="SNAPSHOT",
            description=f"rollout {rollout_id} → {bundle.version}",
        )
        return job_id

    def collect(
        self, rollout_id: RolloutId, device_ids: Sequence[str]
    ) -> dict[str, DeviceOutcome]:
        """디바이스들이 뭐라고 했는지 걷어 온다.

        **응답이 없으면 PENDING 이다.** 없는 것을 실패로 만들지 않는다.
        """
        outcomes: dict[str, DeviceOutcome] = {}
        for device_id in device_ids:
            outcomes[device_id] = self._outcome_of(rollout_id, device_id)
        return outcomes

    def cancel(self, rollout_id: RolloutId, reason: str) -> None:
        for job in self._jobs_of(rollout_id):
            try:
                self._iot.cancel_job(jobId=job, reasonCode="ROLLED_BACK", comment=reason)
            except ClientError:
                pass

    # -- 내부 --------------------------------------------------------------
    def _outcome_of(self, rollout_id: RolloutId, device_id: str) -> DeviceOutcome:
        try:
            response = self._iot.list_job_executions_for_thing(thingName=device_id)
        except ClientError:
            return DeviceOutcome.PENDING

        prefix = f"ota-{rollout_id}-"
        for summary in response.get("executionSummaries", []):
            job_id = summary.get("jobId", "")
            if not job_id.startswith(prefix):
                continue
            status = summary.get("jobExecutionSummary", {}).get("status", "QUEUED")
            return OUTCOME_MAP.get(status, DeviceOutcome.PENDING)
        return DeviceOutcome.PENDING

    def _jobs_of(self, rollout_id: RolloutId) -> list[str]:
        prefix = f"ota-{rollout_id}-"
        try:
            response = self._iot.list_jobs()
        except ClientError:
            return []
        return [
            job["jobId"]
            for job in response.get("jobs", [])
            if job.get("jobId", "").startswith(prefix)
        ]

    def _job_id(self, rollout_id: RolloutId, version: str) -> str:
        # IoT jobId 는 영숫자·대시·언더스코어만 받는다.
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in version)
        return f"ota-{rollout_id}-{safe}"

    def _thing_arn(self, device_id: str) -> str:
        return (
            f"arn:aws:iot:{self._config.region}:{self._account_id}:thing/{device_id}"
        )
