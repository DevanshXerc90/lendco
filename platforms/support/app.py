"""Support platform (Freshdesk/Zoho Desk-like) — borrower support tickets.

Run:  uvicorn platforms.support.app:app --port 8103
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from platforms.common import load, append_runtime, read_runtime, next_id  # noqa: E402

app = FastAPI(title="Support Platform", version="1.0")
_SEED_TICKETS = load("tickets")


class NewTicket(BaseModel):
    borrower_id: str
    loan_id: str | None = None
    category: str
    subject: str
    description: str
    priority: str = "medium"
    channel: str = "voice"


def _all_tickets() -> list[dict]:
    return _SEED_TICKETS + read_runtime("tickets")


@app.get("/health")
def health():
    return {"status": "ok", "service": "support", "tickets": len(_all_tickets())}


@app.get("/tickets")
def list_tickets(borrower_id: str | None = None, status: str | None = None):
    items = _all_tickets()
    if borrower_id:
        items = [t for t in items if t["borrower_id"] == borrower_id]
    if status:
        items = [t for t in items if t["status"] == status]
    return sorted(items, key=lambda t: t["created_at"], reverse=True)


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    for t in _all_tickets():
        if t["ticket_id"] == ticket_id:
            return t
    raise HTTPException(404, "ticket not found")


@app.post("/tickets")
def create_ticket(t: NewTicket):
    ticket_id = next_id("tickets", "TKTN")
    record = {
        "ticket_id": ticket_id,
        "borrower_id": t.borrower_id,
        "loan_id": t.loan_id,
        "category": t.category,
        "subject": t.subject,
        "description": t.description,
        "priority": t.priority,
        "status": "open",
        "channel": t.channel,
        "created_at": "2026-06-09T10:00:00",
        "updated_at": "2026-06-09T10:00:00",
        "resolution": None,
        "created_by": "voice_agent",
    }
    return append_runtime("tickets", record)
