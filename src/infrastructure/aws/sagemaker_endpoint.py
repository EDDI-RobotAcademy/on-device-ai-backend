"""SageMaker 실시간 추론 Endpoint. (실습 6-14)

**진짜 boto3 요청을 만든다.** moto 안에서 돌지만 API 이름이 틀리면 여기서 터진다.

정직하게 밝혀 둘 것:
    moto 의 `invoke_endpoint` 는 **고정된 응답**을 돌려준다.
    실제 추론이 도는 것이 아니다. 그래서 이 실습에서 확인하는 것은
    "모델이 잘 맞히는가"가 아니라 **"엔드포인트라는 구조가 무엇을 요구하는가"**다.

    `update_endpoint` 는 moto 에 없다. 그래서 가중치를 바꾸는 A/B 갱신은
    여기 코드로는 있고 테스트로는 확인하지 못한다 — 그 사실을 테스트에 적어 둔다.
"""

from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from domain.fleet.endpoint import EndpointSpec, EndpointState
from infrastructure.aws.config import AwsConfig


class SageMakerEndpointGateway:
    """domain.fleet.ports.EndpointGateway 구현."""

    def __init__(self, config: AwsConfig) -> None:
        self._sagemaker = boto3.client("sagemaker", **config.client_kwargs())
        self._runtime = boto3.client(
            "sagemaker-runtime", **config.client_kwargs()
        )
        self._role_arn = config.training_role_arn
        self._bucket = config.lake_bucket

    def deploy(self, spec: EndpointSpec, *, image_uri: str) -> EndpointState:
        for variant in spec.variants:
            self._ensure_model(
                f"{spec.name}-{variant.name}", variant.model_reference, image_uri
            )

        config_name = f"{spec.name}-config"
        self._sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": variant.name,
                    "ModelName": f"{spec.name}-{variant.name}",
                    "InitialInstanceCount": variant.instance_count,
                    "InstanceType": variant.instance_type,
                    "InitialVariantWeight": variant.weight,
                }
                for variant in spec.variants
            ],
        )
        self._sagemaker.create_endpoint(
            EndpointName=spec.name, EndpointConfigName=config_name
        )
        return self.describe(spec.name)

    def describe(self, name: str) -> EndpointState:
        response = self._sagemaker.describe_endpoint(EndpointName=name)
        return EndpointState(
            name=name,
            status=response["EndpointStatus"],
            variants=tuple(
                (v["VariantName"], float(v.get("CurrentWeight", 1.0)))
                for v in response.get("ProductionVariants", [])
            ),
        )

    def invoke(self, name: str, body: bytes) -> bytes:
        response = self._runtime.invoke_endpoint(
            EndpointName=name, Body=body, ContentType="application/json"
        )
        return response["Body"].read()

    def shift_traffic(self, name: str, weights: dict[str, float]) -> EndpointState:
        """갈래별 가중치를 바꾼다 — 클라우드 쪽의 A/B (실습 5-14).

        **moto 에는 이 API 가 없다.** 실제 AWS 에서만 동작한다.
        """
        self._sagemaker.update_endpoint_weights_and_capacities(
            EndpointName=name,
            DesiredWeightsAndCapacities=[
                {"VariantName": variant, "DesiredWeight": weight}
                for variant, weight in weights.items()
            ],
        )
        return self.describe(name)

    def teardown(self, name: str) -> None:
        """**실습이 끝나면 반드시 지운다.** 켜 두면 시간당 과금된다."""
        for call, kwargs in (
            (self._sagemaker.delete_endpoint, {"EndpointName": name}),
            (self._sagemaker.delete_endpoint_config, {"EndpointConfigName": f"{name}-config"}),
        ):
            try:
                call(**kwargs)
            except ClientError:
                continue

    # -- 내부 --------------------------------------------------------------
    def _ensure_model(self, model_name: str, artifact_key: str, image_uri: str) -> None:
        """모델 등록. **아티팩트는 S3 에 있어야 한다** — 여기서는 위치만 가리킨다."""
        self._sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": image_uri,
                "ModelDataUrl": f"s3://{self._bucket}/{artifact_key}",
            },
            ExecutionRoleArn=self._role_arn,
        )
