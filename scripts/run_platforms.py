"""Launch all five mock platform services as independent uvicorn processes.

Run:  python -m scripts.run_platforms
Stop: Ctrl-C (terminates all child services).
"""
from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

SERVICES = {
    "crm": "platforms.crm.app:app",
    "payments": "platforms.payments.app:app",
    "support": "platforms.support.app:app",
    "knowledge": "platforms.knowledge.app:app",
    "workflow": "platforms.workflow.app:app",
}


def main() -> None:
    procs: list[subprocess.Popen] = []
    for name, target in SERVICES.items():
        port = settings.PLATFORM_PORTS[name]
        print(f"  starting {name:10s} -> http://127.0.0.1:{port}")
        procs.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", target, "--port", str(port), "--log-level", "warning"],
                cwd=str(settings.ROOT),
            )
        )
        time.sleep(0.4)

    print("\nAll 5 platforms running. Press Ctrl-C to stop.\n")

    def shutdown(*_):
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()
