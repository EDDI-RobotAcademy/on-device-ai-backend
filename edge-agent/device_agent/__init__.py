"""보드 위에서 도는 온디바이스 AI 에이전트 (참조 구현).

이 패키지는 **백엔드가 아니다.** 디바이스에 올라가서 도는 프로그램이다.
그래서 지키는 규칙이 하나 더 있다.

    `domain` 만 import 한다.
    `application` 도 `infrastructure` 도 import 하지 않는다.

이유는 무게다. `infrastructure` 를 하나라도 끌어오면 pandas·torch·tensorflow 가
따라 올라온다. 보드에는 그것을 올릴 자리가 없다.

**그런데 판정은 백엔드와 똑같아야 한다.** 그게 이 구조의 값어치다 —
`domain` 에는 프레임워크가 하나도 없으므로 (CLAUDE.md §14, §15)
같은 파일을 그대로 보드에 올릴 수 있다.

    서버   AlertGate         사고를 재현할 때 (실습 5-13)
    보드   StreamingAlertGate 지금 판단할 때
    → 같은 신호에 **같은 결과**. 테스트가 매번 확인한다.

`tests/test_architecture.py` 가 이 규칙을 강제한다.
"""

from device_agent.agent import DeviceAgent, AgentSettings
from device_agent.bundle import DeployedBundle, load_bundle

__all__ = ["AgentSettings", "DeviceAgent", "DeployedBundle", "load_bundle"]
