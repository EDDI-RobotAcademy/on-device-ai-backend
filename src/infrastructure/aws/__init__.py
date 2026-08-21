"""AWS 어댑터. **boto3 는 여기까지만 온다.** (CLAUDE.md §15)

    s3_object_store.py       S3        — 객체 저장소
    dynamo_device_registry.py DynamoDB — 수천 대의 상태
    sagemaker_training.py    SageMaker — 원격 학습
    iot_ota_gateway.py       IoT Jobs  — OTA 배포

네 파일 모두 domain/fleet/ports.py 의 Protocol 을 구현한다.
Domain 은 이 파일들의 존재를 모른다.

테스트는 moto 로 **실제 boto3 호출**을 시험한다.
가짜 클라이언트를 만들어 놓고 "호출됐다"만 확인하면
API 이름이 틀려도 통과한다 — 그건 어댑터를 시험한 것이 아니다.
"""
