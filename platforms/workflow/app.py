"""Workflow-automation platform (n8n/Make/Zapier-like) — callbacks, human
escalations, and automation triggers (e.g. re-register NACH mandate).

Run:  uvicorn platforms.workflow.app:app --port 8105
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings  # noqa: E402
from platforms.common import append_runtime, read_runtime, next_id, require_platform_key  # noqa: E402

app = FastAPI(title="Workflow Platform", version="1.0")

# The automations this back office actually knows how to run. Anything else is
# rejected so a stray tool call cannot fire an undefined workflow.
WORKFLOWS = set(settings.WORKFLOW_NAMES)


class Callback(BaseModel):
    borrower_id: str = Field(pattern=r"^BRW\d{5}$")
    reason: str = Field(min_length=1, max_length=300)
    preferred_window: str = Field(default="tomorrow 10:00-12:00", max_length=80)


class Escalation(BaseModel):
    borrower_id: str = Field(pattern=r"^BRW\d{5}$")
    reason: str = Field(min_length=1, max_length=300)
    priority: str = "high"
    context: str | None = Field(default=None, max_length=1000)


class Trigger(BaseModel):
    borrower_id: str = Field(pattern=r"^BRW\d{5}$")
    params: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "service": "workflow"}


@app.post("/callbacks", dependencies=[Depends(require_platform_key)])
def schedule_callback(cb: Callback):
    record = {"callback_id": next_id("callbacks", "CB"), "status": "scheduled", **cb.model_dump()}
    return append_runtime("callbacks", record)


@app.post("/escalations", dependencies=[Depends(require_platform_key)])
def escalate(e: Escalation):
    if e.priority not in settings.TICKET_PRIORITIES:
        raise HTTPException(422, f"Unknown priority '{e.priority}'.")
    record = {"escalation_id": next_id("escalations", "ESC"), "status": "routed_to_human", **e.model_dump()}
    return append_runtime("escalations", record)


@app.post("/workflows/{name}/trigger", dependencies=[Depends(require_platform_key)])
def trigger(name: str, t: Trigger):
    """Fire a named automation, e.g. 'reregister_mandate', 'send_foreclosure_statement'."""
    if name not in WORKFLOWS:
        raise HTTPException(404, f"Unknown workflow '{name}'. Allowed: {sorted(WORKFLOWS)}")
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
