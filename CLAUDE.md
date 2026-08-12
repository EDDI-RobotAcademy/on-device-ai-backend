# Backend Engineering Guide

## 1. Project Identity

이 디렉터리는 산업용 On-Device AI 시스템의 Backend를 담당한다.

기술 스택:

* Python
* FastAPI
* Pydantic
* PyTorch
* ONNX
* TFLite
* AWS SDK
* Database / Storage

Backend의 핵심 목적은 단순한 REST API 서버를 만드는 것이 아니다.

다음 전체 AI 시스템을 구현한다.

```text
Data
→ Data Quality
→ Model
→ Optimization
→ Deployment
→ Operations
→ Cloud
→ Fleet
```

Backend는 이 시스템의 Application / Domain / Infrastructure를 담당한다.

---

# 2. Architecture Principles

Backend는 다음 Layer를 기본으로 한다.

```text
interfaces
    ↓
application
    ↓
domain
    ↑
infrastructure
```

의존성 방향은 반드시 안쪽으로 향한다.

```text
Interfaces → Application → Domain
Infrastructure → Application / Domain
```

Domain은 FastAPI, AWS, PyTorch, ONNX 등의 기술에 의존하지 않는다.

---

# 3. Directory Structure

기본 구조:

```text
backend/
├── src/
│   ├── domain/
│   │   ├── data/
│   │   ├── data_quality/
│   │   ├── model/
│   │   ├── optimization/
│   │   ├── deployment/
│   │   ├── operations/
│   │   └── fleet/
│   │
│   ├── application/
│   │   ├── data/
│   │   ├── data_quality/
│   │   ├── model/
│   │   ├── optimization/
│   │   ├── deployment/
│   │   ├── operations/
│   │   └── fleet/
│   │
│   ├── infrastructure/
│   │   ├── persistence/
│   │   ├── ml/
│   │   ├── runtime/
│   │   ├── edge/
│   │   ├── aws/
│   │   └── monitoring/
│   │
│   └── interfaces/
│       └── http/
│           ├── routes/
│           ├── schemas/
│           └── dependencies/
│
└── tests/
    ├── domain/
    ├── application/
    ├── infrastructure/
    └── interfaces/
```

실제 구현 과정에서 더 적절한 Bounded Context가 발견되면 구조를 개선할 수 있다.

단순히 위 디렉터리를 기계적으로 생성하지 마라.

---

# 4. Domain First

Domain을 먼저 정의한다.

구현 순서는 기본적으로 다음을 따른다.

```text
Problem
→ Capability
→ Use Case
→ Domain Model
→ Application
→ Infrastructure
→ API
→ Test
```

FastAPI Route부터 만들지 마라.

DB Schema부터 만들지 마라.

AWS SDK부터 연결하지 마라.

먼저 "이 시스템이 어떤 문제를 해결하는가?"를 정의한다.

---

# 5. Domain Responsibilities

Domain에는 다음과 같은 산업 AI 핵심 개념이 존재할 수 있다.

### Data

* Dataset
* DatasetId
* DataSource
* DataSchema
* DataSample
* TimeSeries
* ImageSample
* DataPartition

### Data Quality

* DataQualityGate
* QualityRule
* QualityReport
* MissingValue
* Outlier
* Duplicate
* LabelQuality
* ClassDistribution
* DataIssue

### Model

* Model
* ModelVersion
* TrainingRun
* Evaluation
* Metric
* Prediction
* ModelArtifact

### Optimization

* OptimizationRun
* Quantization
* Precision
* Benchmark
* Latency
* MemoryUsage
* ModelSize
* RuntimeProfile

### Deployment

* Deployment
* DeploymentTarget
* DeploymentVersion
* DeploymentStatus

### Operations

* InferenceLog
* DeviceHealth
* DriftReport
* Alert
* Incident
* Rollback

### Fleet

* Device
* DeviceGroup
* Fleet
* OTADeployment
* DeviceModelVersion

위 이름을 무조건 그대로 사용하지 마라.

실제 Domain 분석을 통해 Entity, Value Object, Aggregate, Domain Service를 결정한다.

---

# 6. DDD Rules

Domain Model을 CRUD Entity의 모음으로 만들지 마라.

다음 개념을 적극적으로 검토한다.

* Entity
* Value Object
* Aggregate
* Aggregate Root
* Domain Service
* Domain Event
* Repository Interface
* Domain Policy

Domain Object는 자신의 불변식과 Business Rule을 가능한 한 스스로 보호해야 한다.

나쁜 예:

```python
dataset.status = "VALID"
dataset.quality_score = 98.2
```

좋은 방향:

```python
dataset.validate_quality(report)
```

또는:

```python
quality_gate.evaluate(dataset)
```

단, 모든 것을 억지로 Domain Method로 만들지는 마라.

---

# 7. Application Layer

Application Layer는 Use Case를 구현한다.

예:

```text
ValidateDataset
TrainModel
EvaluateModel
OptimizeModel
DeployModel
RollbackDeployment
DetectDrift
CreateRetrainingJob
DeployOTA
RollbackOTA
```

Application Layer의 역할:

* Use Case orchestration
* Transaction boundary
* Domain object 조합
* Repository 호출
* External service 호출
* 결과 DTO 생성

Business Rule 자체를 Application Layer에 쌓지 마라.

---

# 8. Infrastructure

Infrastructure는 기술 구현이다.

예:

```text
PyTorch
ONNX
TFLite
NumPy
Pandas
S3
SageMaker
DynamoDB
Database
Device Runtime
Monitoring
```

이러한 기술은 Domain에 침투해서는 안 된다.

나쁜 예:

```python
class Dataset:
    def save_to_s3(self, ...):
        ...
```

좋은 방향:

```text
Domain
    DatasetRepository

Infrastructure
    S3DatasetRepository
```

---

# 9. FastAPI

FastAPI는 Interface Adapter다.

Route에서는 다음만 수행한다.

```text
HTTP Request
→ Validation
→ Application Use Case
→ Response Mapping
```

Route에 Business Logic을 작성하지 마라.

나쁜 예:

```python
@app.post("/datasets/{id}/validate")
def validate(id):
    if missing_ratio > 0.2:
        ...
```

좋은 구조:

```python
@app.post("/datasets/{id}/validate")
def validate(...):
    result = validate_dataset.execute(...)
    return DatasetValidationResponse.from_domain(result)
```

---

# 10. API DTO

Pydantic Schema와 Domain Model을 분리한다.

```text
CreateDatasetRequest
DatasetResponse
ValidationResponse
ModelResponse
DeploymentResponse
```

등은 HTTP DTO다.

다음은 서로 다른 개념이다.

```text
HTTP DTO
≠
Domain Entity
≠
Persistence Model
```

필요하면 Mapper를 사용한다.

---

# 11. Long Running Jobs

다음 작업은 장시간 실행될 수 있다.

* Dataset Validation
* Model Training
* Model Evaluation
* Model Optimization
* Deployment
* OTA
* Retraining

HTTP Request가 작업 완료까지 기다리는 구조를 기본값으로 만들지 마라.

필요하면 다음 구조를 사용한다.

```text
Command
↓
Job
↓
Job Status
↓
Result
```

예:

```text
POST /models/{id}/training
        ↓
TrainingJob
        ↓
RUNNING
        ↓
COMPLETED
        ↓
TrainingResult
```

---

# 12. Error Handling

Domain Error와 HTTP Error를 분리한다.

예:

```text
DomainException
ApplicationException
InfrastructureException
HTTPException
```

Domain Exception을 FastAPI `HTTPException`으로 직접 만들지 마라.

HTTP Layer에서 변환한다.

---

# 13. Testing

테스트 피라미드를 따른다.

```text
        E2E
       /   \
 Integration
    /       \
 Application
      |
    Domain
```

가장 중요한 것은 Domain Test다.

최소한 다음을 테스트한다.

### Domain

* Entity invariant
* Value Object
* Domain Service
* Domain Event
* Business Rule

### Application

* Use Case
* Repository interaction
* External service interaction

### Infrastructure

* Repository
* ML adapter
* AWS adapter

### API

* Request validation
* Response
* Error mapping

---

# 14. AI Framework Isolation

PyTorch는 Domain이 아니다.

ONNX도 Domain이 아니다.

TFLite도 Domain이 아니다.

AWS SageMaker도 Domain이 아니다.

이들은 Infrastructure / Adapter다.

예:

```text
Domain
  Model
    ↓
Application
  OptimizeModel
    ↓
Infrastructure
  PyTorchModelAdapter
  ONNXModelAdapter
  TFLiteModelAdapter
```

따라서 ML Framework를 교체하더라도 Domain이 무너지지 않아야 한다.

---

# 15. AWS Isolation

AWS 역시 Infrastructure다.

Domain에서 다음과 같은 코드를 작성하지 마라.

```python
import boto3
```

Domain은 AWS를 몰라야 한다.

```text
Domain
  ↓
Port / Repository
  ↓
AWS Adapter
  ↓
S3 / SageMaker / DynamoDB
```

AWS를 다른 Cloud Provider로 교체할 수 있는 구조를 유지한다.

---

# 16. Curriculum Traceability

이 프로젝트는 교육과정과 코드가 연결되어야 한다.

전체 관계:

```text
Curriculum
↓
Learning Objective
↓
Capability
↓
Use Case
↓
Domain
↓
Application
↓
Infrastructure
↓
API
↓
Test
```

모든 실습 제목은 하나 이상의 Capability / Use Case와 연결되어야 한다.

단,

교육 제목과 Domain을 1:1로 매핑하지 마라.

예:

```text
"결측치를 숨기면 AI가 대신 대가를 치른다"
```

때문에

```text
MissingValueDomain
```

을 만드는 것은 잘못된 설계다.

대신 기존 Data Quality Domain의 기능을 통해 해당 교육 내용을 구현한다.

---

# 17. Curriculum Reference

코드가 어떤 교육 내용을 지원하는지 추적할 수 있어야 한다.

그러나 Domain 코드에 교육 제목을 하드코딩하지 마라.

권장:

```text
docs/curriculum/
```

또는 별도의 Curriculum Metadata를 사용한다.

예:

```yaml
id: DQ-10
title: AI를 학습시키기 전에 Data Quality Gate를 통과시켜라

capabilities:
  - dataset-validation

use_cases:
  - validate-dataset

domain:
  - DataQualityGate

application:
  - ValidateDataset

api:
  - POST /datasets/{id}/validation
```

---

# 18. Backend ↔ Frontend Contract

React는 Backend Domain을 직접 참조하지 않는다.

통신:

```text
React
↓
HTTP API
↓
FastAPI
↓
Application
↓
Domain
```

Frontend에 Business Rule을 복제하지 마라.

Backend가 Single Source of Truth다.

---

# 19. Observability

산업 AI Backend는 실행 상태를 추적할 수 있어야 한다.

최소한 고려할 것:

* Request Log
* Application Log
* Inference Log
* Training Log
* Deployment Log
* Model Version
* Device Status
* Latency
* Error
* Drift
* Incident

관측 가능성을 나중에 추가하지 말고 Architecture 단계부터 고려한다.

---

# 20. Code Quality Rules

다음 원칙을 지킨다.

* Type Hint 사용
* 명확한 함수/클래스 이름
* 작은 책임
* 명시적인 의존성
* Dependency Injection
* 테스트 가능한 구조
* 숨겨진 전역 상태 금지
* Domain에서 Framework 의존 금지
* Circular Dependency 금지

"교육용이라서 대충 만든다"는 접근을 하지 마라.

실제 산업 AI Backend로 설명할 수 있는 수준을 유지한다.

---

# 21. Implementation Rule

새 기능을 추가할 때 반드시 먼저 다음을 설명한다.

```text
Problem
Capability
Use Case
Domain Change
Application Change
Infrastructure Change
API Change
Test Change
Curriculum Mapping
```

그 다음 코드를 작성한다.

Architecture가 불명확하면 코드를 먼저 만들지 마라.

---

# 22. 금지사항

다음 구조를 만들지 마라.

### God Service

```text
AIService
ModelService
DataService
```

안에 모든 것을 넣는 구조.

### CRUD-only Architecture

```text
DatasetService.create()
DatasetService.update()
DatasetService.delete()
```

만 존재하는 구조.

### Framework-driven Domain

```text
FastAPI
Pydantic
SQLAlchemy
PyTorch
```

가 Domain Model을 결정하는 구조.

### Cloud-driven Domain

```text
S3
SageMaker
DynamoDB
```

가 Business Model을 결정하는 구조.

### Curriculum-driven Domain

교육 제목 하나마다 Domain을 만드는 구조.

---

# 23. Definition of Done

Backend 기능은 다음을 만족해야 한다.

* Domain 책임이 명확하다.
* Use Case가 명확하다.
* Business Rule이 적절한 위치에 있다.
* Infrastructure 의존성이 격리되어 있다.
* FastAPI Route가 얇다.
* DTO와 Domain Model이 분리되어 있다.
* Unit Test가 존재한다.
* Integration Test가 필요한 경우 존재한다.
* API Contract가 정의되어 있다.
* Curriculum Mapping이 존재한다.
* Frontend에서 사용할 수 있는 명확한 API가 존재한다.
