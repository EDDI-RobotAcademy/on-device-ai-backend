"""SageMaker 학습 어댑터. (실습 6-5)

`domain.fleet.ports.RemoteTrainingGateway` 구현.

이 어댑터가 하는 일은 셋이다.

    제출한다   Domain 의 ComputeSpec → SageMaker 의 ResourceConfig
    물어본다   SageMaker 의 TrainingJobStatus → Domain 의 RemoteJobStatus
    번역한다   실패 이유를 Job 에 붙인다 — 로그 어딘가에 있는 것은 없는 것과 같다

**기다리지 않는다.** `submit()` 은 즉시 돌아온다 (CLAUDE.md §11).
"""

from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from domain.fleet.training_job import ComputeSpec, RemoteJobStatus, RemoteTrainingJob
from infrastructure.aws.config import AwsConfig
from infrastructure.errors import InfrastructureException

STATUS_MAP: dict[str, RemoteJobStatus] = {
    "InProgress": RemoteJobStatus.RUNNING,
    "Completed": RemoteJobStatus.SUCCEEDED,
    "Failed": RemoteJobStatus.FAILED,
    "Stopping": RemoteJobStatus.RUNNING,
    "Stopped": RemoteJobStatus.STOPPED,
}


class TrainingJobUnavailable(InfrastructureException):
    code = "TRAINING_JOB_UNAVAILABLE"


class SageMakerTrainingGateway:
    """domain.fleet.ports.RemoteTrainingGateway 구현."""

    def __init__(self, config: AwsConfig) -> None:
        self._config = config
        self._sm = boto3.client("sagemaker", **config.client_kwargs())

    def submit(
        self,
        job_id: str,
        *,
        dataset_uri: str,
        output_uri: str,
        compute: ComputeSpec,
        hyperparameters: dict[str, str] | None = None,
    ) -> RemoteTrainingJob:
        self._sm.create_training_job(
            TrainingJobName=job_id,
            AlgorithmSpecification={
                "TrainingImage": self._config.training_image,
                "TrainingInputMode": "File",
            },
            RoleArn=self._config.training_role_arn,
            InputDataConfig=[
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": dataset_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
            ],
            OutputDataConfig={"S3OutputPath": output_uri},
            ResourceConfig={
                "InstanceType": compute.instance_type,
                "InstanceCount": compute.instance_count,
                "VolumeSizeInGB": 30,
            },
            StoppingCondition={
                # **상한이 곧 안전장치다.** 멈추지 않는 학습은 과금만 된다.
                "MaxRuntimeInSeconds": compute.max_runtime_seconds
            },
            HyperParameters={k: str(v) for k, v in (hyperparameters or {}).items()},
        )
        # 비용 추정치는 **우리 설정에서 온다.** AWS 응답에는 가격이 없다.
        # describe() 가 다시 만든 ComputeSpec 에는 그것이 빠지므로 여기서 되살린다.
        from dataclasses import replace

        return replace(self.describe(job_id), compute=compute)

    def describe(self, job_id: str) -> RemoteTrainingJob:
        try:
            response = self._sm.describe_training_job(TrainingJobName=job_id)
        except ClientError as exc:
            raise TrainingJobUnavailable(
                f"학습 작업을 찾을 수 없다: {job_id}", subject=job_id
            ) from exc
        return _to_job(response)

    def stop(self, job_id: str, reason: str) -> RemoteTrainingJob:
        try:
            self._sm.stop_training_job(TrainingJobName=job_id)
        except ClientError:
            # 이미 끝났으면 멈출 것이 없다. 그것을 오류로 만들지 않는다.
            pass
        return self.describe(job_id)


def _to_job(response: dict) -> RemoteTrainingJob:
    """SageMaker 응답 → Domain.

    **이 함수가 어댑터의 본체다.** 바깥 세계의 어휘를 Domain 의 어휘로 바꾼다.
    """
    raw_status = response.get("TrainingJobStatus", "InProgress")
    status = STATUS_MAP.get(raw_status, RemoteJobStatus.PENDING)

    resource = response.get("ResourceConfig", {})
    stopping = response.get("StoppingCondition", {})
    compute = ComputeSpec(
        instance_type=resource.get("InstanceType", "unknown"),
        instance_count=int(resource.get("InstanceCount", 1)),
        max_runtime_seconds=int(stopping.get("MaxRuntimeInSeconds", 3_600)),
    )

    artifacts = response.get("ModelArtifacts", {}) or {}
    artifact_uri = artifacts.get("S3ModelArtifacts", "")
    failure = response.get("FailureReason", "")

    # Domain 의 불변식을 지켜서 넘긴다.
    #   FAILED 인데 이유가 없으면 Domain 이 거부한다 — 여기서 채워 준다.
    #   SUCCEEDED 인데 결과물 위치가 없으면 출력 경로로 대신한다.
    if status is RemoteJobStatus.FAILED and not failure:
        failure = f"{raw_status} (이유가 응답에 없다 — CloudWatch 로그를 확인한다)"
    if status is RemoteJobStatus.SUCCEEDED and not artifact_uri:
        artifact_uri = response.get("OutputDataConfig", {}).get("S3OutputPath", "")

    metrics = {
        item["MetricName"]: float(item["Value"])
        for item in response.get("FinalMetricDataList", [])
    }

    return RemoteTrainingJob(
        job_id=response.get("TrainingJobName", ""),
        dataset_uri=_input_uri(response),
        output_uri=response.get("OutputDataConfig", {}).get("S3OutputPath", "s3://unknown"),
        compute=compute,
        status=status,
        submitted_at=str(response.get("CreationTime", "")),
        finished_at=str(response.get("TrainingEndTime", "")),
        failure_reason=failure,
        metrics=metrics,
        artifact_uri=artifact_uri,
    )


def _input_uri(response: dict) -> str:
    for channel in response.get("InputDataConfig", []) or []:
        source = channel.get("DataSource", {}).get("S3DataSource", {})
        if source.get("S3Uri"):
            return source["S3Uri"]
    return "s3://unknown"
