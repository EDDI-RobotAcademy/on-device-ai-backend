"""Data Bounded Context.

해결하는 문제:
    "현장에서 받은 이 데이터를, 학습을 시작해도 되는 데이터라고 말할 수 있는가?"

이 Context 는 데이터의 *구조/의미/시간/라벨/분할/대표성*을 다룬다.
결측치 비율·이상치·중복 같은 *오염도* 판정은 data_quality Context 의 책임이다.

Aggregate Root:
    Dataset

핵심 설계 원칙:
    측정(measurement)은 Infrastructure 가 한다.  (pandas / numpy / Pillow)
    판정(verdict)은 Domain 이 한다.               (Policy → InspectionReport)
"""
