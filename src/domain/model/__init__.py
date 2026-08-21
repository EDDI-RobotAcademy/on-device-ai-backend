"""Model Bounded Context.

해결하는 문제:
    "이 데이터로 판단하는 기계를 만들 수 있는가, 그리고 그 기계를 믿어도 되는가?"

Aggregate Root:
    TrainingRun     한 번의 학습. 시작되고, 진행되고, 끝난다.
    ModelVersion    그 결과물. 학습이 끝나야 존재한다.

절대 하지 않는 것 (CLAUDE.md §14):
    이 Context 는 PyTorch 를 모른다. Tensor 를 들고 있지 않다.
    아는 것은 **모양(shape)과 숫자(metric)** 뿐이다.
    그래서 PyTorch 를 다른 프레임워크로 바꿔도 여기는 한 줄도 바뀌지 않는다.

다른 Context 와의 관계:
    domain.data / domain.data_quality 를 import 하지 않는다.
    학습에 쓸 데이터는 TrainingDataRef(원시 값만 담은 VO)로 번역되어 들어온다.
"""
