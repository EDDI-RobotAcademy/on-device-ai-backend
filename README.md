# on-device-ai-backend

산업용 On-Device AI 시스템의 Backend.
**온디바이스 AI 엔지니어 과정**의 실습 코드이자, 그대로 산업 백엔드로 설명할 수 있는 구조를 목표로 한다.

```text
Data → Data Quality → Model → Optimization → Deployment → Operations → Cloud → Fleet
```

현재 구현 범위

| 모듈 | 묻는 것 | 실습 | Bounded Context |
|---|---|---|---|
| **1 — 데이터** | 이 데이터가 **무엇인지 아는가** | 1-1 ~ 1-11 | `domain/data` |
| **2 — 데이터 품질** | 이 데이터가 **쓸 만한가** | 2-1 ~ 2-11 | `domain/data_quality` |
| **3 — 모델** | 이 **모델**이 쓸 만한가 | 3-1 ~ 3-15 | `domain/model` |
| **4 — 최적화** | 이 모델을 **이 디바이스에** 올릴 수 있는가 | 4-1 ~ 4-14 | `domain/optimization` |
| **5 — 운영** | 이것이 **아직** 괜찮은가 | 5-1 ~ 5-15 | `domain/operations` |
| **6 — AWS** | 이것을 **수천 대에서** 할 수 있는가 | 6-1 ~ 6-14 | `domain/fleet` |

실습은 모두 **80개**다. 교과목 1(직무 이해)에 대응하는 문서는
[docs/role](docs/role/README.md) 에 따로 있다 — 그건 코드가 아니라 개념이다.

그리고 보드에 올라가는 쪽은 [`edge-agent/`](edge-agent/README.md) 에 있다.
**`domain` 만 가져가는 참조 구현이다** — 판정 코드가 서버와 디바이스에서 같은 것이다.

앞의 네 게이트는 **내보내기 전에** 묻고, 모듈 5 는 **내보낸 뒤에 계속** 묻는다.
그리고 현장에는 정답이 없으므로, 정확도 대신 잴 수 있는 것으로 답한다.
모듈 6 은 그 순환을 수천 대 규모로 닫는다.

```text
데이터 → 품질 → 모델 → 최적화 → 운영 → 클라우드 → **다시 데이터**
```

---

## 시작하기

```bash
python3 -m venv .venv          # 없다면
bash scripts/setup.sh
```

`setup.sh` 가 하는 일:

1. 과정 전체에서 쓰는 라이브러리를 **한 번에 전부** 설치한다
   (PyTorch / ONNX / TensorFlow(TFLite) / boto3 / pandas / scikit-learn …)
2. 설치 상태를 점검한다 (`scripts/preflight.py`)
3. 실습 데이터를 만든다 (`data/samples/`)
4. 전체 테스트를 돌린다

실습 도중에 `pip install` 로 멈추는 일이 없도록 처음에 다 깔고 시작한다.

### 실습

```bash
pytest -m lesson_1_1 -s     # 실습 1-1 의 결과를 출력과 함께 본다
pytest -q                   # 전체 테스트
```

문서는 [`docs/curriculum/`](docs/curriculum/README.md) 에 있다.

### API

```bash
.venv/bin/uvicorn main:app --reload
# http://127.0.0.1:8000/docs
```

---

## 구조

```text
src/
├── domain/          문제 그 자체. FastAPI/pandas/PyTorch/boto3 를 모른다
│   ├── shared/      Identifier, DomainEvent, DomainException, Finding/Severity/Verdict
│   ├── data/          Data Bounded Context         (Dataset Aggregate)
│   ├── data_quality/  Data Quality Bounded Context (QualityAssessment Aggregate)
│   ├── model/         Model Bounded Context        (TrainingRun Aggregate)
│   ├── optimization/  Optimization Bounded Context (OptimizationRun Aggregate)
│   ├── operations/    Operations Bounded Context   (Deployment + HealthWatch)
│   └── fleet/         Fleet Bounded Context        (Fleet + Rollout)
├── application/     Use Case. 조합하고 시키기만 한다
├── infrastructure/  기술 구현. pandas, Pillow, numpy, PyTorch, structlog
│   ├── analysis/    측정 어댑터
│   ├── ml/          PyTorch 어댑터 — 학습
│   ├── optimization/ 변환·측정 어댑터 — TorchScript/ONNX/TFLite 는 여기까지만 온다
│   ├── edge/        디바이스 시뮬레이터 — 모듈 6 에서 실제 에이전트가 온다
│   ├── monitoring/  추론 로그 저장소, 현장 관측 어댑터
│   ├── aws/         S3/DynamoDB/SageMaker/IoT — **boto3 는 여기까지만 온다**
│   ├── persistence/ 저장소
│   ├── sample_data/ 실습 데이터 생성기
│   └── config/      의존성 조립 (Composition Root)
└── interfaces/
    └── http/        FastAPI 라우트 / Pydantic DTO / 예외 변환

tests/
├── domain/          불변식과 Policy — 파일도 pandas 도 필요 없다
├── application/     Use Case 조립 — 가짜 측정기를 끼워서
├── infrastructure/  실제 어댑터 — 심어 놓은 결함을 정말 세는가
├── interfaces/      API 계약과 상태 코드
└── curriculum/      실습별 결과 확인 (lesson_1_1 ~ lesson_6_11)
```

의존성은 안쪽으로만 향한다.

```text
interfaces → application → domain ← infrastructure
```

그리고 **여섯 Bounded Context 는 서로를 모른다.**
어느 Context 도 다른 Context 를 한 줄도 import 하지 않는다.
번역은 Application Layer 의 `*_mapper.py`(Anti-Corruption Layer)가 한다.

```text
Dataset            → AssessmentTarget   (모듈 2 가 보는 것)
Dataset + 품질평가  → TrainingDataRef    (모듈 3 이 보는 것, 게이트 통과 여부 포함)
TrainingRun        → BaselineModelRef   (모듈 4 가 보는 것, 승인 여부 포함)
OptimizationRun    → DeployedArtifactRef (모듈 5 가 보는 것, 선택 여부 + 기준 숫자)
ReleaseBundle      → 디바이스로 나가는 묶음 (모듈 6, 전처리·기준·계보 포함)
```

`domain/model` 에도 `domain/optimization` 에도
**`torch` / `onnx` / `tensorflow` 라는 단어가 하나도 없다.** (CLAUDE.md §14)
그리고 `domain/fleet` 의 코드에는 **`boto3` / `S3` / `SageMaker` 가 하나도 없다.** (§15)

`tests/test_architecture.py` 가 이것을 매번 확인한다 — AST 로 코드를 읽어서.
docstring 은 오히려 AWS 를 이야기한다. **"S3 라고 부르지 않는다"** 를 설명해야 하기 때문이다.

### 설계에서 가장 중요한 한 가지

**측정은 Infrastructure 가, 판정은 Domain 이 한다.**

```python
# Infrastructure — 숫자만 센다
SensorChannelMeasurement(field_name="voltage_v", longest_constant_run=600, ...)

# Domain — 기준과 판단을 갖는다
SignalPlausibilityPolicy(max_constant_run_ratio=0.05).inspect_sensors(...)
# → Finding(code="SIGNAL_STUCK", severity=CRITICAL, measured=0.0709, threshold=0.05)
```

덕분에 Domain 테스트는 파일 없이 0.03초에 끝나고,
pandas 를 polars 로 바꿔도 규칙은 한 줄도 바뀌지 않는다.

---

## 실습 데이터

`data/samples/` 는 git 에 올리지 않는다. 언제든 다시 만든다.

```bash
.venv/bin/python -m scripts.generate_sample_data
```

캡스톤 3개 산업군 중 두 가지에 대응한다.

| 파일 | 대응 캡스톤 |
|------|-------------|
| `plant_power_*.csv` | 제조 공장 실시간 전력 데이터 기반 이상 징후 탐지 |
| `castings/` | 자동차 다이캐스팅 부품 이미지 기반 불량 판별 |

**결함이 의도적으로 심어져 있다.** 무엇이 몇 개인지는
`src/infrastructure/sample_data/` 의 docstring 에 전부 적혀 있다.

| 데이터 | 결함 | 성격 |
|---|---|---|
| `plant_power_raw.csv` | D01~D13 | **구조**가 깨졌다 (모듈 1) |
| `plant_power_quality_dirty.csv` | Q01~Q10 | 구조는 멀쩡하고 **내용**이 오염됐다 (모듈 2) |
| `plant_power_model_train.csv` | — | 두 게이트를 통과한 학습 데이터 (모듈 3) |
| `plant_power_model_field.csv` | — | **현장 홀드아웃** — 다른 날 데이터 (모듈 3) |
| `plant_power_operations.csv` | O01~O06 | 오염이 아니라 **시간** — 4일치가 하루씩 변한다 (모듈 5) |
| `castings/` | I01~I05 | 이미지 결함 — **찾는 연습용** (모듈 1) |
| `plant_power_quality_holdout.csv` | — | 두 학습 파일과 겹치지 않는 공통 시험지 (실습 3-15) |
| `castings-train/` | — | 다이캐스팅 **학습용** 280장 (실습 3-11, 캡스톤 주제 1) |
| `food/` | — | 식품 표면 **학습용** 320장 3분류 (실습 3-11, 캡스톤 주제 2) |

모듈 4 는 새 데이터를 쓰지 않는다. **모듈 3 이 학습해 둔 모델 그 자체**를 변환한다.
모듈 5 는 **모듈 4 가 고른 결과물을** 4일치 현장 신호에 실제로 돌려 추론 로그를 만든다.
그래서 컨테이너들이 저장소를 공유한다 — 따로 만들면 "학습된 모델을 찾을 수 없다"가 된다.

모듈 6 도 새 CSV 를 쓰지 않는다. **moto 위에서 진짜 boto3 로** S3·DynamoDB·SageMaker·IoT 를 부른다.

정답을 아는 데이터로 눈을 만든 뒤에 현장으로 나간다.

---

## 규칙

이 저장소의 설계 규칙은 [`CLAUDE.md`](CLAUDE.md) 에 있다. 요약하면:

- Domain 을 먼저 정의한다. FastAPI Route 부터 만들지 않는다.
- Business Rule 은 Domain 에 둔다. Route 와 Use Case 는 얇다.
- HTTP DTO ≠ Domain Entity ≠ Persistence Model
- PyTorch / ONNX / TFLite / AWS 는 전부 Infrastructure 다
- **측정은 Infrastructure, 판정은 Domain** — 이 한 줄이 다섯 모듈을 관통한다
- 교육 제목 하나마다 Domain 을 만들지 않는다
