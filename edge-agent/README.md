# edge-agent — 온디바이스 추론 실행 코드 (참조 구현)

캡스톤 공통 산출물 중 **"온디바이스 배포 구성 및 추론 실행 코드"** 에 해당한다.

이 디렉터리는 **백엔드가 아니다.** 보드에 올라가서 도는 프로그램이다.

```text
backend  (src/)          서버에서 돈다. 판정하고, 기록하고, 내보낸다.
edge-agent/              보드에서 돈다. 받고, 판단하고, 올린다.
```

---

## 이 코드가 존재하는 이유

실습 [5-12](../docs/curriculum/5-12.md) 는 다섯 단계를 **백엔드에서 한 번에** 돌려
단계별 통계를 냈다. 그건 재현과 분석을 위한 것이었다.

보드에는 **미래가 없다.** 표본이 하나씩 들어오고, 그때마다 지금 판단해야 한다.
그리고 회선은 끊기고, 전원은 나가고, 새 모델은 한밤중에 내려온다.

이 코드는 그 조건에서 도는 모양이다.

---

## 가장 중요한 한 줄

```python
# edge-agent 는 domain 만 import 한다.
from domain.operations.alerting import AlertRule, StreamingAlertGate
from domain.operations.pipeline import PipelineContract, PipelinePolicy
```

`application` 도 `infrastructure` 도 가져오지 않는다.
하나라도 가져오면 pandas·torch·tensorflow 가 따라 올라오고,
보드에는 그것을 올릴 자리가 없다.

**그런데 판정은 백엔드와 똑같다.**

```text
서버   AlertGate           사고를 재현할 때 (실습 5-13)
보드   StreamingAlertGate  지금 판단할 때
       → 같은 신호에 **같은 결과**
```

`domain` 에 프레임워크가 하나도 없기 때문에 가능한 일이다 (CLAUDE.md §14, §15).
이것이 이 저장소의 구조가 실제로 값을 하는 지점이다.

`tests/test_architecture.py` 가 이 규칙을 매번 강제한다 —
`test_디바이스_에이전트는_domain_만_가져간다`.

보드에 올라가는 의존성 전부:

```text
numpy
tflite-runtime      (개발 기계에서는 ai-edge-litert)
pyserial / opencv   실제 하드웨어를 붙일 때만
```

---

## 구조

```text
device_agent/
├── bundle.py      내려온 묶음을 읽고 **검증한다** (체크섬 · 계약 · 라벨)
├── sources.py     ACQUIRE — CSV 재생 / 시리얼 센서 / 카메라
├── preprocess.py  PREPROCESS — 링 버퍼 + 학습 때 통계로 정규화
├── runtime.py     INFER — TFLite 인터프리터 (스레드 1 고정)
├── store.py       EMIT — 로컬 버퍼(store-and-forward) + 업링크
├── slots.py       OTA — A/B 슬롯, 원자적 교체, 롤백
├── agent.py       다섯 단계 루프 + 상태 + 종료 처리
└── __main__.py    CLI
```

각 파일이 어느 실습에서 왔는지:

| 파일 | 실습 |
|---|---|
| `bundle.py` | [6-1](../docs/curriculum/6-1.md) 체크섬 · [5-12](../docs/curriculum/5-12.md) 계약 · [5-15](../docs/curriculum/5-15.md) 무엇을 함께 보낼 것인가 |
| `preprocess.py` | [1-7](../docs/curriculum/1-7.md) 정규화 통계 · [3-4](../docs/curriculum/3-4.md) 창 · [5-12](../docs/curriculum/5-12.md) 구간 경계 |
| `runtime.py` | [4-1](../docs/curriculum/4-1.md) 스레드 1 · [4-4](../docs/curriculum/4-4.md) TFLite · [4-13](../docs/curriculum/4-13.md) 코어 |
| `store.py` | [6-1](../docs/curriculum/6-1.md) 묶어 올리기 · 개인정보 · 하루 예산 |
| `slots.py` | [5-15](../docs/curriculum/5-15.md) 저장 여유 · [6-8](../docs/curriculum/6-8.md) OTA · [6-9](../docs/curriculum/6-9.md) 롤백 |
| `agent.py` | [5-12](../docs/curriculum/5-12.md) 다섯 단계 · [5-13](../docs/curriculum/5-13.md) 알람 규율 · [5-5](../docs/curriculum/5-5.md) 느려짐 |

---

## 돌려 보기

```bash
# 백엔드 데이터와 도메인을 함께 본다
export PYTHONPATH=$PWD/src:$PWD/edge-agent

# 실습 데이터가 없다면
.venv/bin/python -m scripts.generate_sample_data

# 상태 확인
.venv/bin/python -m device_agent status --slots /tmp/ondevice/slots

# 돌린다 (회선 없이 — 로컬에만 쌓는다)
.venv/bin/python -m device_agent run \
    --slots /tmp/ondevice/slots \
    --source data/samples/plant_power_operations.csv \
    --device-id DEV-01 \
    --max-samples 4000
```

묶음을 만드는 것은 서버 쪽 일이다.
테스트(`tests/edge_agent/conftest.py`)가 **모듈 4 가 고른 진짜 TFLite 결과물**로
묶음을 만드는 방법을 보여 준다 — 가짜 모델 파일을 쓰지 않는다.

백엔드로 올리려면:

```bash
--backend http://127.0.0.1:8000 --fleet-id line3
```

끊겨 있으면 로컬에 쌓고, 살아나면 보낸다. **하나도 안 버린다.**

---

## 새 모델을 받고, 되돌리기

```bash
# 받는다 — 아직 안 켠다. **옛 것은 그대로 있다.**
#   (OTA 가 slots/b/ 에 내려놓는다)

# 검증하고 켠다. 체크섬이나 계약이 안 맞으면 표시는 그대로다.
python -m device_agent install --slots /tmp/ondevice/slots --bundle /tmp/ondevice/slots/b

# 되돌린다 — 표시 하나를 옮기는 것으로 끝난다
python -m device_agent rollback --slots /tmp/ondevice/slots
```

첫 배포에는 되돌릴 곳이 없다. 그때는 그렇게 말한다 —
**"되돌릴 곳이 없다"는 실패가 아니라 상태다** ([6-9](../docs/curriculum/6-9.md)).

---

## 이 코드가 하지 않는 것

정직하게 적어 둔다.

- **실제 보드에서 돌려 보지 않았다.** macOS/Linux 개발 기계에서만 검증했다.
  `tflite_runtime` 은 보드에서, `ai_edge_litert` 는 개발 기계에서 쓰도록 갈라 두었다.
- **시리얼·카메라 경로는 하드웨어 없이 테스트하지 못했다.**
  대신 라이브러리가 없으면 **즉시 실패한다** — 0으로 채워 흘려보내지 않는다.
- **OTA 수신(다운로드)은 없다.** 묶음이 이미 디스크에 있다고 본다.
  내려받는 쪽은 [6-8](../docs/curriculum/6-8.md) 의 IoT Jobs 가 담당한다.
- **워치독·systemd 유닛은 없다.** 배치 방식은 현장마다 다르다.

확인하지 못한 것을 확인한 척하지 않는다.

---

## 테스트

```bash
.venv/bin/python -m pytest tests/edge_agent -v
```

21개가 확인하는 것:

```text
증분 게이트가 배치 게이트와 **같은 답**을 낸다   ← 가장 중요하다
잘려 온 묶음을 거절한다
계약이 없는 묶음을 거절한다
모델에 없는 라벨을 알람으로 걸면 거절한다
회선이 끊겨도 하나도 안 버린다
개인정보를 디바이스에서 지운다
받는 동안 옛 것을 안 지운다
되돌릴 곳이 없으면 그렇게 말한다
다섯 단계가 **진짜 TFLite 모델로** 실제로 돈다
```
