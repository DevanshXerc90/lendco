"""Payments platform (Razorpay/Stripe-sandbox-like) — payment history, gateway
logs, payment-link generation.

Run:  uvicorn platforms.payments.app:app --port 8102
"""
from __future__ import annotations

from collections import defaultdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from platforms.common import load, append_runtime, next_id  # noqa: E402

app = FastAPI(title="Payments Platform", version="1.0")
_PAYMENTS = load("payments")
_BY_BORROWER: dict[str, list[dict]] = defaultdict(list)
for p in _PAYMENTS:
    _BY_BORROWER[p["borrower_id"]].append(p)
_BY_ID = {p["payment_id"]: p for p in _PAYMENTS}


class PaymentLinkRequest(BaseModel):
    borrower_id: str
    amount: float
    purpose: str = "EMI payment"


@app.get("/health")
def health():
    return {"status": "ok", "service": "payments", "records": len(_PAYMENTS)}


@app.get("/payments")
def list_payments(borrower_id: str, status: str | None = None):
    items = sorted(_BY_BORROWER.get(borrower_id, []), key=lambda x: x["due_date"])
    if status:
        items = [p for p in items if p["status"] == status]
    return items


@app.get("/payments/{payment_id}")
def get_payment(payment_id: str):
    p = _BY_ID.get(payment_id)
    if not p:
        raise HTTPException(404, "payment not found")
    return p


@app.get("/gateway-logs")
def gateway_logs(borrower_id: str):
    """Raw gateway responses for failed/partial attempts — the source of truth
    for root-cause analysis in the payment-failure scenarios."""
    items = _BY_BORROWER.get(borrower_id, [])
    return [
        {
            "payment_id": p["payment_id"],
            "due_date": p["due_date"],
            "status": p["status"],
            "method": p["method"],
            "gateway": p["gateway"],
            "response_code": p["gateway_response_code"],
            "response_message": p["gateway_response_message"],
            "failure_reason": p["failure_reason"],
            "root_cause": p["root_cause"],
            "penalty_charged": p["penalty_charged"],
        }
        for p in items
        if p["status"] != "success"
    ]


@app.post("/payment-links")
def create_payment_link(req: PaymentLinkRequest):
    link_id = next_id("payment_links", "PL")
    record = {
        "link_id": link_id,
        "borrower_id": req.borrower_id,
        "amount": req.amount,
        "purpose": req.purpose,
        "url": f"https://pay.lendco.example/l/{link_id.lower()}",
        "status": "active",
        "valid_for_hours": 24,
    }
    return append_runtime("payment_links", record)
