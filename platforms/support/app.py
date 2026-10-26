"""Support platform (Freshdesk/Zoho Desk-like) — borrower support tickets.

Run:  uvicorn platforms.support.app:app --port 8103
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings  # noqa: E402
from platforms.common import load, append_runtime, read_runtime, next_id, require_platform_key  # noqa: E402
from core.dates import iso_now  # noqa: E402

app = FastAPI(title="Support Platform", version="1.0")
_SEED_TICKETS = load("tickets")


PRIORITIES = set(settings.TICKET_PRIORITIES)


class NewTicket(BaseModel):
    borrower_id: str = Field(pattern=r"^BRW\d{5}$")
    loan_id: str | None = Field(default=None, max_length=32)
    # Shape-checked, not allowlisted: categories are open-ended by design
    # (seeded data uses payment_failure, penalty_waiver, emi_dispute, ...).
    category: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    subject: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    priority: str = "medium"
    channel: str = Field(default="voice", max_length=24)


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


@app.post("/tickets", dependencies=[Depends(require_platform_key)])
def create_ticket(t: NewTicket):
    if t.priority not in PRIORITIES:
        raise HTTPException(422, f"Unknown priority '{t.priority}'. Allowed: {sorted(PRIORITIES)}")
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
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "resolution": None,
        "created_by": "voice_agent",
    }
    return append_runtime("tickets", record)
