"""Model Context 의 Use Case.

    PrepareTrainingRun     (3-1, 3-2, 3-4)
    ProfileArchitecture    (3-2)
    MaterializeImages      (3-3)
    ExecuteTrainingRun     (3-5 ~ 3-8)
    EvaluateModel          (3-9)
    AcceptModel            (3-10)

학습은 오래 걸린다 (CLAUDE.md §11).
그래서 '준비'와 '실행'을 나눈다 — HTTP 요청이 학습을 붙잡고 기다리지 않는다.

    POST /training-runs            → PREPARED (즉시 반환)
    POST /training-runs/{id}/start → RUNNING → COMPLETED (백그라운드)
    GET  /training-runs/{id}       → 진행 상황
"""
