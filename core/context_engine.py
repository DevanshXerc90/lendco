"""Context Engine.

Builds a single unified borrower-context object before every interaction by
aggregating across the independent platforms and the memory store:

  Borrower context     : loan info, KYC profile, payment history
  Interaction context  : previous calls/tickets, previous commitments (PTP)
  Operational context  : delinquency status, open cases, recent failures

It also derives the loan analytics the agent needs (EMIs paid / remaining,
interest vs principal paid to date, outstanding principal, next due date,
overdue amount, penalty totals) so downstream agents reason over facts, not raw
rows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from core import clients  # noqa: E402

TODAY = date(2026, 6, 9)


@dataclass
class LoanAnalytics:
    tenure_months: int
    emi_amount: float
    installments_paid: int
    installments_remaining: int
    principal_paid: float
    interest_paid: float
    outstanding_principal: float
    next_due_date: str | None
    overdue_amount: float
    total_penalty_charged: float


@dataclass
class UnifiedContext:
    borrower_id: str
    found: bool
    # borrower context
    profile: dict = field(default_factory=dict)
    kyc: dict = field(default_factory=dict)
    analytics: dict = field(default_factory=dict)
    payments: list = field(default_factory=list)
    # interaction context
    prior_conversations: list = field(default_factory=list)
    prior_commitments: list = field(default_factory=list)
    # operational context
    delinquency_status: str = "unknown"
    dpd_days: int = 0
    open_tickets: list = field(default_factory=list)
    recent_failures: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _amortize(principal: float, annual_rate: float, emi: float, paid_count: int) -> tuple[float, float, float]:
    """Walk the amortization schedule for `paid_count` installments.
    Returns (principal_paid, interest_paid, outstanding_principal)."""
    r = annual_rate / 12 / 100
    outstanding = principal
    prin_paid = int_paid = 0.0
    for _ in range(paid_count):
        interest = outstanding * r
        prin = min(emi - interest, outstanding)
        outstanding -= prin
        prin_paid += prin
        int_paid += interest
        if outstanding <= 0:
            break
    return round(prin_paid, 2), round(int_paid, 2), round(max(outstanding, 0.0), 2)


def _derive_analytics(profile: dict, payments: list[dict]) -> LoanAnalytics:
    successful = [p for p in payments if p["status"] == "success"]
    installments_paid = len(successful)
    tenure = profile["tenure_months"]
    emi = profile["emi_amount"]
    prin_paid, int_paid, outstanding = _amortize(
        profile["loan_amount"], profile["interest_rate"], emi, installments_paid
    )
    # next due = first installment that is not successfully paid, by due date.
    # If every recorded installment is paid, project the next one from the schedule
    # (the synthetic history only spans elapsed months) until the loan is complete.
    unpaid = sorted([p for p in payments if p["status"] != "success"], key=lambda x: x["due_date"])
    if unpaid:
        next_due = unpaid[0]["due_date"]
    elif installments_paid < tenure:
        from datetime import date as _date, timedelta as _td
        start = _date.fromisoformat(profile["loan_start_date"])
        next_due = (start + _td(days=(installments_paid + 1) * 30)).isoformat()
    else:
        next_due = None  # loan fully repaid
    overdue = sum(
        p["amount_due"] - p["amount_paid"]
        for p in payments
        if p["status"] in ("missed", "failed", "auto_debit_failed", "partial")
        and p["due_date"] <= TODAY.isoformat()
    )
    total_penalty = sum(p.get("penalty_charged", 0.0) for p in payments)
    return LoanAnalytics(
        tenure_months=tenure,
        emi_amount=emi,
        installments_paid=installments_paid,
        installments_remaining=max(tenure - installments_paid, 0),
        principal_paid=prin_paid,
        interest_paid=int_paid,
        outstanding_principal=outstanding,
        next_due_date=next_due,
        overdue_amount=round(overdue, 2),
        total_penalty_charged=round(total_penalty, 2),
    )


def _load_prior_conversations(borrower_id: str) -> list[dict]:
    path = settings.DATA_DIR / "conversations.json"
    if not path.exists():
        return []
    convos = json.loads(path.read_text(encoding="utf-8"))
    mine = [c for c in convos if c["borrower_id"] == borrower_id]
    return sorted(mine, key=lambda c: c["timestamp"], reverse=True)


class ContextEngine:
    """Assembles the unified context. Memory is injected (optional) so the engine
    has no hard dependency on the memory subsystem."""

    def __init__(self, memory=None):
        self.memory = memory

    def build(self, borrower_id: str | None = None, phone: str | None = None) -> UnifiedContext:
        profile = (
            clients.crm.get_borrower(borrower_id) if borrower_id
            else clients.crm.find_by_phone(phone) if phone
            else None
        )
        if not profile:
            return UnifiedContext(borrower_id=borrower_id or phone or "unknown", found=False)

        bid = profile["borrower_id"]
        kyc = clients.crm.get_kyc(bid) or profile.get("kyc", {})
        payments = clients.payments.history(bid)
        analytics = _derive_analytics(profile, payments)

        tickets = clients.support.tickets(bid)
        open_tickets = [t for t in tickets if t["status"] in ("open", "in_progress")]

        recent_failures = [
            p for p in clients.payments.gateway_logs(bid)
            if p.get("status") in ("failed", "auto_debit_failed", "partial")
        ][-5:]

        prior_conversations = _load_prior_conversations(bid)
        prior_commitments = self._collect_commitments(bid, prior_conversations)

        return UnifiedContext(
            borrower_id=bid,
            found=True,
            profile=profile,
            kyc=kyc,
            analytics=analytics.__dict__,
            payments=payments,
            prior_conversations=prior_conversations[:5],
            prior_commitments=prior_commitments,
            delinquency_status=profile.get("delinquency_status", "unknown"),
            dpd_days=profile.get("dpd_days", 0),
            open_tickets=open_tickets,
            recent_failures=recent_failures,
        )

    def _collect_commitments(self, borrower_id: str, conversations: list[dict]) -> list[dict]:
        commitments: list[dict] = []
        # from seeded conversation history
        for c in conversations:
            ptp = c.get("promise_to_pay")
            if ptp:
                commitments.append({"source": "conversation", "conversation_id": c["conversation_id"],
                                    "timestamp": c["timestamp"], **ptp})
        # from the live memory store (commitments captured by the agent itself)
        if self.memory is not None:
            for m in self.memory.get_commitments(borrower_id):
                commitments.append({"source": "memory", **m})
        return commitments
