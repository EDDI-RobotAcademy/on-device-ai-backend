"""Data Quality Bounded Context.

해결하는 문제:
    "구조는 맞다. 그런데 이 데이터가 **쓸 만한가**?"

Data Context 가 "이 데이터가 무엇인지 아는가"를 물었다면,
여기서는 "이 데이터가 얼마나 오염되었는가"를 묻는다.
둘은 다른 질문이고, 하나를 통과했다고 다른 하나가 통과되지 않는다.

Aggregate Root:
    QualityAssessment

품질의 여섯 축:
    COMPLETENESS   결측       — 값이 없다
    VALIDITY       이상치     — 값이 말이 안 된다
    LABEL_QUALITY  라벨 오류  — 정답이 틀렸다
    BALANCE        불균형     — 한쪽으로 쏠렸다
    NOISE          잡음       — 신호보다 잡음이 크다
    UNIQUENESS     중복       — 같은 것을 반복해서 가르친다

여섯 개의 실습이 하나의 개념(품질)의 여섯 축을 다룬다.
실습 제목마다 Domain 을 만든 것이 아니다. (CLAUDE.md §16, §22)

다른 Context 와의 관계:
    이 Context 는 domain.data 를 import 하지 않는다.
    필요한 정보는 AssessmentTarget(원시 값만 담은 VO)으로 번역되어 들어온다.
    번역은 Application Layer 의 책임이다.
"""
