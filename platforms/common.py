"""Shared helpers for the mock platform services.

Each platform is an INDEPENDENT FastAPI app (own port, own process, own domain
slice of the data). They emulate five real SaaS systems:

  CRM        (Zoho/HubSpot-like)   -> borrower master + KYC
  Payments   (Razorpay-like)       -> payment history, gateway logs, payment links
  Support    (Freshdesk-like)      -> support tickets
  Knowledge  (Notion-like)         -> policy / FAQ documents
  Workflow   (n8n-like)            -> callbacks, escalations, automation triggers

Read-only domain data is loaded from data/generated/. Writes (new tickets,
callbacks, payment links, escalations) are appended to data/runtime/ so the
rest of the system (memory, dashboard) can observe side effects.
"""
from __future__ import annotations

import hmac
import json
import threading
from pathlib import Path

from fastapi import Header, HTTPException

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

RUNTIME = settings.ROOT / "data" / "runtime"
RUNTIME.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()


def load(name: str) -> list[dict]:
    path = settings.DATA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m data.generate` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def require_platform_key(x_platform_key: str | None = Header(None)) -> None:
    """Guard for mutating endpoints on these internal services.

    The platforms are meant to be reached only by the agent/API (loopback by
    default). When PLATFORM_API_KEY is configured, writes must also carry it in
    X-Platform-Key. When it is unset the key check is a no-op and the services
    rely on loopback binding alone.
    """
    if not settings.PLATFORM_API_KEY:
        return
    if not x_platform_key or not hmac.compare_digest(x_platform_key, settings.PLATFORM_API_KEY):
        raise HTTPException(403, "Missing or invalid X-Platform-Key.")


def runtime_path(name: str) -> Path:
    return RUNTIME / f"{name}.json"


def append_runtime(name: str, record: dict) -> dict:
    """Append a record to a runtime store (thread-safe) and return it."""
    with _lock:
        RUNTIME.mkdir(parents=True, exist_ok=True)  # resilient to a runtime reset mid-run
        path = runtime_path(name)
        items = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        items.append(record)
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return record


def read_runtime(name: str) -> list[dict]:
    path = runtime_path(name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def next_id(name: str, prefix: str) -> str:
    existing = read_runtime(name)
    return f"{prefix}{len(existing) + 1:05d}"
