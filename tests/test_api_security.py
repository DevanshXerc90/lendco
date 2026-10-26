from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def test_production_requires_both_secrets():
    env = os.environ.copy()
    env.update({"VERCEL": "1", "ENV": "production", "API_SECRET": "", "DEMO_API_KEY": ""})
    result = subprocess.run(
        [sys.executable, "-c", "import api.index"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "API_SECRET and DEMO_API_KEY" in result.stderr


def test_configured_api_key_protects_and_limits_body(monkeypatch):
    import api.index as api

    monkeypatch.setattr(api, "DEMO_API_KEY", "demo-test-key")
    client = TestClient(api.app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/eval").status_code == 401
    assert client.post(
        "/api/scenario",
        content=b"x" * (api.MAX_BODY_BYTES + 1),
        headers={"X-API-Key": "demo-test-key"},
    ).status_code == 413
    assert client.get("/api/eval", headers={"X-API-Key": "demo-test-key"}).status_code in {200, 404}
