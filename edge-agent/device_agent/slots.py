"""A/B 슬롯 — 새 모델을 받고, 되돌린다. (실습 5-15, 6-8, 6-9)

디바이스에는 슬롯이 둘 있다.

    slots/a/   지금 도는 것
    slots/b/   직전에 돌던 것 (또는 방금 받은 것)
    current    → a 또는 b 를 가리키는 표시

교체는 **표시 하나를 바꾸는 것**으로 끝난다. 그래서 원자적이고, 그래서 되돌리기도 원자적이다.

지키는 규칙 셋:

    받는 동안 옛 것을 지우지 않는다
        지우고 받으면 **되돌릴 곳이 없어진다** (실습 5-15 `PKG_NO_STORAGE`).

    검증한 뒤에 표시를 옮긴다
        체크섬과 계약을 먼저 본다. 옮기고 나서 확인하면 이미 늦다.

    되돌릴 곳이 없으면 되돌린다고 하지 않는다
        첫 배포에는 직전 버전이 없다 (실습 6-9).
"""

from __future__ import annotations

import os
from pathlib import Path

from device_agent.bundle import BundleRejected, DeployedBundle, load_bundle

SLOTS = ("a", "b")
POINTER = "current"


class NoPreviousVersion(RuntimeError):
    """되돌릴 곳이 없다. **첫 배포였다는 뜻이다.**"""


class SlotStore:
    """두 슬롯과 하나의 표시를 관리한다."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for slot in SLOTS:
            (self.root / slot).mkdir(exist_ok=True)

    # -- 조회 --------------------------------------------------------------
    @property
    def pointer_path(self) -> Path:
        return self.root / POINTER

    def active_slot(self) -> str | None:
        if not self.pointer_path.is_file():
            return None
        value = self.pointer_path.read_text(encoding="utf-8").strip()
        return value if value in SLOTS else None

    def standby_slot(self) -> str:
        """비어 있는 쪽. 새 것은 항상 여기로 받는다."""
        active = self.active_slot()
        return "b" if active == "a" else "a"

    def load_active(self) -> DeployedBundle:
        slot = self.active_slot()
        if slot is None:
            raise BundleRejected("올라와 있는 묶음이 없다")
        return load_bundle(self.root / slot)

    def has_previous(self) -> bool:
        standby = self.standby_slot()
        try:
            load_bundle(self.root / standby)
        except BundleRejected:
            return False
        return True

    # -- 변경 --------------------------------------------------------------
    def stage_slot(self) -> Path:
        """받을 자리. **옛 것은 그대로 둔다.**"""
        return self.root / self.standby_slot()

    def activate(self, slot: str) -> DeployedBundle:
        """검증하고 나서 표시를 옮긴다.

        `os.replace` 는 원자적이다 — 중간에 전원이 나가도
        표시가 반쪽만 쓰인 상태로 남지 않는다.
        """
        if slot not in SLOTS:
            raise ValueError(f"슬롯은 {SLOTS} 중 하나여야 한다")
        bundle = load_bundle(self.root / slot)  # ← 옮기기 **전에** 검증한다

        temporary = self.root / f"{POINTER}.tmp"
        temporary.write_text(slot, encoding="utf-8")
        os.replace(temporary, self.pointer_path)
        return bundle

    def install(self, staged: Path) -> DeployedBundle:
        """받아 둔 것을 검증하고 켠다. 실패하면 표시는 그대로다."""
        slot = staged.name
        return self.activate(slot)

    def rollback(self) -> DeployedBundle:
        """직전 버전으로 되돌린다. **표시 하나를 옮기는 것으로 끝난다.**"""
        standby = self.standby_slot()
        try:
            bundle = load_bundle(self.root / standby)
        except BundleRejected as exc:
            raise NoPreviousVersion(
                "되돌릴 곳이 없다. **첫 배포였다는 뜻이다** (실습 6-9). "
                "이건 실패가 아니라 상태다 — 다른 대응이 필요하다."
            ) from exc
        return self.activate(standby)

    def describe(self) -> str:
        lines = [f"슬롯 저장소: {self.root}"]
        active = self.active_slot()
        for slot in SLOTS:
            mark = " ← 지금 도는 것" if slot == active else ""
            try:
                bundle = load_bundle(self.root / slot)
                lines.append(f"  {slot}  {bundle.version}{mark}")
            except BundleRejected as exc:
                lines.append(f"  {slot}  (없음: {exc}){mark}")
        return "\n".join(lines)
