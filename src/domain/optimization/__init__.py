"""Optimization Bounded Context.

해결하는 문제:
    "PC에서 도는 모델을 디바이스 예산 안에 넣을 수 있는가, 그리고 그 대가는 무엇인가?"

Aggregate Root:
    OptimizationRun     한 모델에 대한 최적화 시도의 기록

이 Context 가 다루는 두 축:
    실행 경로(RuntimeTarget)  PyTorch → TorchScript → ONNX → TFLite
    정밀도(Precision)         FP32 → FP16 → INT8

둘은 독립이다. 경로만 바꿔도 빨라지고, 정밀도만 줄여도 작아진다.
그리고 **둘 다 공짜가 아니다.**

절대 하지 않는 것 (CLAUDE.md §14):
    이 Context 는 torch 도 tensorflow 도 onnx 도 모른다.
    아는 것은 **바이트 수와 밀리초와 정확도**뿐이다.
"""
