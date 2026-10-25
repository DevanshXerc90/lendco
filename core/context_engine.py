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
from decimal import Decimal, ROUND_HALF_UP

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from core import clients  # noqa: E402
from core import dates  # noqa: E402


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


def _money(x: Decimal) -> float:
    """Round to 2 decimal places (paise precision) as a float."""
    return float(x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _amortize(principal: float, annual_rate: float, emi: float, paid_count: int) -> tuple[float, float, float]:
    """Walk the amortization schedule for `paid_count` installments.
    Returns (principal_paid, interest_paid, outstanding_principal).
    Decimal arithmetic throughout so financial totals never accumulate float error."""
    r = Decimal(str(annual_rate)) / Decimal(12) / Decimal(100)
    outstanding = Decimal(str(principal))
    emi_due = Decimal(str(emi))
    prin_paid = Decimal(0)
    int_paid = Decimal(0)
    for _ in range(paid_count):
        interest = outstanding * r
        prin = min(emi_due - interest, outstanding)
        outstanding -= prin
        prin_paid += prin
        int_paid += interest
        if outstanding <= 0:
            break
    return _money(prin_paid), _money(int_paid), _money(max(outstanding, Decimal(0)))


def _group_by_installment(payments: list[dict]) -> list[list[dict]]:
    """Drop duplicate records (same payment_id) and group the rest by
    installment, so one installment is never counted or charged twice."""
    seen: set[str] = set()
    groups: dict[str, list[dict]] = {}
    for p in payments:
        pid = str(p.get("payment_id") or "") or f"row{id(p)}"
        if pid in seen:
            continue
        seen.add(pid)
        key = str(p.get("installment_no") or pid)
        groups.setdefault(key, []).append(p)
    return list(groups.values())


def _due_of(group: list[dict]) -> str:
    return min(p["due_date"] for p in group)


def _is_paid(group: list[dict]) -> bool:
    """An installment counts as paid only when a record shows a full success."""
    return any(p["status"] == "success" for p in group)


def _shortfall(group: list[dict]) -> Decimal:
    """Still-owed part of an installment: due amount minus everything actually
    paid across attempts — partial payments reduce it rather than being ignored."""
    due = max(Decimal(str(p.get("amount_due") or 0)) for p in group)
    paid = sum((Decimal(str(p.get("amount_paid") or 0)) for p in group), Decimal(0))
    return max(due - paid, Decimal(0))


def _derive_analytics(profile: dict, payments: list[dict]) -> LoanAnalytics:
    groups = _group_by_installment(payments)
    records = [p for g in groups for p in g]
    paid_groups = [g for g in groups if _is_paid(g)]
    installments_paid = len(paid_groups)
    tenure = profile["tenure_months"]
    emi = profile["emi_amount"]
    prin_paid, int_paid, outstanding = _amortize(
        profile["loan_amount"], profile["interest_rate"], emi, installments_paid
    )

    today = dates.today()
    # Unpaid installments, earliest due date first.
    unpaid = sorted((g for g in groups if not _is_paid(g)), key=_due_of)
    if unpaid:
        # First installment still outstanding (it may already be overdue).
        next_due = _due_of(unpaid[0])
    elif installments_paid < tenure:
        # Everything recorded is paid: the next due date is the following
        # installment on the schedule (calendar months, never 30-day months).
        # We deliberately do NOT push a past date forward — if that date has
        # already gone by, the ledger is behind the schedule (a missing or
        # unrecorded payment) and hiding it would misrepresent the account.
        start = profile.get("loan_start_date")
        next_due = None
        if start:
            next_due = dates.add_months(date.fromisoformat(start), installments_paid + 1).isoformat()
    else:
        next_due = None  # loan fully repaid

    # Overdue = still-owed amount on unpaid installments whose due date has arrived.
    overdue = sum(
        (_shortfall(g) for g in unpaid if _due_of(g) <= today.isoformat()),
        Decimal(0),
    )
    total_penalty = sum((Decimal(str(p.get("penalty_charged") or 0)) for p in records), Decimal(0))
    return LoanAnalytics(
        tenure_months=tenure,
        emi_amount=emi,
        installments_paid=installments_paid,
        installments_remaining=max(tenure - installments_paid, 0),
        principal_paid=prin_paid,
        interest_paid=int_paid,
        outstanding_principal=outstanding,
        next_due_date=next_due,
        overdue_amount=_money(overdue),
        total_penalty_charged=_money(total_penalty),
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
