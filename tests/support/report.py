"""실습 결과를 눈으로 보기 위한 출력 헬퍼.

    pytest -m lesson_1_5 -s

로 실행하면 이 출력이 그대로 터미널에 나온다.
숫자를 보지 않고 통과/실패만 보는 실습은 아무것도 가르치지 못한다.
"""

from __future__ import annotations

WIDTH = 78


def section(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def block(title: str, body: str) -> None:
    print()
    print(f"── {title} " + "─" * max(WIDTH - len(title) - 4, 0))
    print(body)


def note(text: str) -> None:
    print(f"   · {text}")
