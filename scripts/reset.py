"""Reset runtime state (platform writes + agent memory) for a clean demo.

Run:  python -m scripts.reset
Does NOT touch generated datasets — only runtime side-effects and memory.db.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

runtime = settings.ROOT / "data" / "runtime"
if runtime.exists():
    shutil.rmtree(runtime)
    print(f"cleared {runtime}")
if settings.MEMORY_DB.exists():
    settings.MEMORY_DB.unlink()
    print(f"cleared {settings.MEMORY_DB}")
print("reset complete.")
