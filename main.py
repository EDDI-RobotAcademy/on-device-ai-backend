"""실행 진입점.

    .venv/bin/uvicorn main:app --reload
    → http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from interfaces.http.app import app  # noqa: E402

__all__ = ["app"]
