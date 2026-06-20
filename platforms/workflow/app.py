"""Workflow-automation platform (n8n/Make/Zapier-like) — callbacks, human
escalations, and automation triggers (e.g. re-register NACH mandate).

Run:  uvicorn platforms.workflow.app:app --port 8105
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from platforms.common import append_runtime, read_runtime, next_id  # noqa: E402

app = FastAPI(title="Workflow Platform", version="1.0")


class Callback(BaseModel):
    borrower_id: str
    reason: str
    preferred_window: str = "tomorrow 10:00-12:00"


class Escalation(BaseModel):
    borrower_id: str
    reason: str
    priority: str = "high"
    context: str | None = None


class Trigger(BaseModel):
    borrower_id: str
    params: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "service": "workflow"}


@app.post("/callbacks")
def schedule_callback(cb: Callback):
    record = {"callback_id": next_id("callbacks", "CB"), "status": "scheduled", **cb.model_dump()}
    return append_runtime("callbacks", record)


@app.post("/escalations")
def escalate(e: Escalation):
    record = {"escalation_id": next_id("escalations", "ESC"), "status": "routed_to_human", **e.model_dump()}
    return append_runtime("escalations", record)


@app.post("/workflows/{name}/trigger")
def trigger(name: str, t: Trigger):
    """Fire a named automation, e.g. 'reregister_mandate', 'send_foreclosure_statement'."""
    record = {
        "run_id": next_id("workflow_runs", "WF"),
        "workflow": name,
        "borrower_id": t.borrower_id,
        "params": t.params,
        "status": "completed",
    }
    return append_runtime("workflow_runs", record)


@app.get("/callbacks")
def list_callbacks(borrower_id: str | None = None):
    items = read_runtime("callbacks")
    return [c for c in items if not borrower_id or c["borrower_id"] == borrower_id]


@app.get("/escalations")
def list_escalations(borrower_id: str | None = None):
    items = read_runtime("escalations")
    return [e for e in items if not borrower_id or e["borrower_id"] == borrower_id]
