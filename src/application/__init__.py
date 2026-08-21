"""Application Layer.

Use Case 를 구현한다. Business Rule 을 여기에 쌓지 않는다. (CLAUDE.md §7)
여기서 하는 일은 네 가지다.
    1. Repository 에서 Aggregate 를 꺼낸다.
    2. Port(측정기)를 호출해 사실을 얻는다.
    3. Aggregate / Policy 에게 판단을 시킨다.
    4. 결과를 DTO 로 만들어 돌려주고, Event 를 발행한다.
"""
