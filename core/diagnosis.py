"""Diagnosis Layer.

Given an utterance + unified context, it determines:
  - intent (what the borrower wants)
  - what is already KNOWN (resolved from systems/context)
  - what is UNKNOWN (information gaps)
  - what to ASK NEXT (prioritized, de-duplicated, dynamically phrased questions)
  - which tools/agents the orchestrator should engage

The design is slot-based. Each intent declares the slots it needs; a slot is
either *derivable* (a resolver pulls it from context — never asked) or
*ask-only* (only the borrower can supply it). The layer only ever asks for slots
that are still missing AND cannot be derived, which is how redundant questioning
is avoided.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# ----------------------------------------------------------------- intent model
INTENT_KEYWORDS: dict[str, list[str]] = {
    "remaining_emi": ["how many emi", "emis remaining", "remaining emi", "installments left", "tenure left", "how many installments", "emis left"],
    "interest_paid": ["interest paid", "how much interest", "interest so far", "interest till"],
    "penalty_inquiry": ["why penalty", "penalty charged", "late fee", "why was i charged", "charge on my account"],
    "payment_failure": ["payment failed", "payment did not go", "debit failed", "auto debit failed", "had enough balance", "transaction failed", "emi bounced"],
    "penalty_waiver": ["waive", "waiver", "reverse the penalty", "remove the penalty", "refund the charge"],
    "promise_to_pay": ["i will pay", "pay next", "pay by", "salary is delayed", "salary delay", "pay after", "promise to pay", "make the payment on"],
    "settlement_request": ["settle", "settlement", "one time settlement", "ots", "close at a lower"],
    "foreclosure": ["foreclose", "foreclosure", "close my loan", "pay off the loan", "preclose", "payoff amount"],
    "account_info": ["my loan details", "account information", "loan summary", "my emi amount", "due date"],
    "general_faq": ["how does", "what is", "policy", "explain", "can i"],
}


def classify_intent(utterance: str) -> tuple[str, float]:
    """Lightweight keyword classifier. The orchestrator may override with the LLM,
    but this guarantees a deterministic baseline and an offline fallback."""
    u = utterance.lower()
    best, best_hits = "general_faq", 0
    for intent, kws in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in u)
        if hits > best_hits:
            best, best_hits = intent, hits
    confidence = min(1.0, 0.4 + 0.3 * best_hits) if best_hits else 0.25
    return best, round(confidence, 2)


# ------------------------------------------------------------------- slot model
@dataclass
class Slot:
    name: str
    label: str
    resolver: Callable[[dict], object] | None  # derive from context; None => ask-only
    question: str                               # dynamic follow-up phrasing
    priority: int = 5                           # lower = asked first
    ask_only: bool = False                      # borrower is the only source


def _ctx_get(ctx: dict, path: str):
    cur = ctx
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


# Resolvers ------------------------------------------------------------------
def _r(path):
    return lambda ctx: _ctx_get(ctx, path)


def _recent_failure(ctx: dict):
    fails = ctx.get("recent_failures") or []
    return fails[-1] if fails else None


def _open_commitment(ctx: dict):
    coms = ctx.get("prior_commitments") or []
    pending = [c for c in coms if c.get("kept") in (None, False)]
    return pending[-1] if pending else None


# Intent -> slots ------------------------------------------------------------
INTENT_SLOTS: dict[str, list[Slot]] = {
    "remaining_emi": [
        Slot("tenure", "loan tenure", _r("analytics.tenure_months"), "", 1),
        Slot("paid", "installments paid", _r("analytics.installments_paid"), "", 1),
        Slot("next_due", "next due date", _r("analytics.next_due_date"), "", 2),
    ],
    "interest_paid": [
        Slot("interest_paid", "interest paid", _r("analytics.interest_paid"), "", 1),
        Slot("principal_paid", "principal paid", _r("analytics.principal_paid"), "", 1),
        Slot("outstanding", "outstanding balance", _r("analytics.outstanding_principal"), "", 2),
    ],
    "penalty_inquiry": [
        Slot("penalty_total", "penalty charged", _r("analytics.total_penalty_charged"), "", 1),
        Slot("failure_record", "the late/failed payment", _recent_failure,
             "Which month's charge are you asking about — the most recent one?", 3),
    ],
    "payment_failure": [
        Slot("failure_record", "the failed payment", _recent_failure,
             "Which payment are you referring to — the most recent failed EMI?", 2),
        Slot("borrower_reason", "the borrower's account of what happened", None,
             "Could you tell me what you saw when the payment failed - any error message or message from your bank?", 4, ask_only=True),
    ],
    "penalty_waiver": [
        Slot("failure_record", "the failed payment", _recent_failure, "", 2),
        Slot("failure_root_cause", "root cause of failure",
             lambda c: (_recent_failure(c) or {}).get("root_cause"), "", 2),
        Slot("borrower_reason", "the borrower's reason for the waiver", None,
             "Can you confirm the reason you believe the penalty should be waived?", 5, ask_only=True),
    ],
    "promise_to_pay": [
        Slot("ptp_date", "committed pay date", None,
             "When do you expect to be able to make the payment?", 1, ask_only=True),
        Slot("ptp_amount", "committed amount", _r("analytics.emi_amount"),
             "How much do you plan to pay — the full EMI, or a part of it?", 2),
        Slot("ptp_reason", "reason for delay", None,
             "May I ask the reason for the delay, so I can note it on your account?", 3, ask_only=True),
    ],
    "settlement_request": [
        Slot("dpd", "delinquency status", _r("dpd_days"), "", 1),
        Slot("hardship_reason", "hardship reason", None,
             "Could you share the financial difficulty you're facing? It helps us assess your request.", 2, ask_only=True),
    ],
    "foreclosure": [
        Slot("outstanding", "outstanding principal", _r("analytics.outstanding_principal"), "", 1),
        Slot("installments_paid", "installments paid (lock-in)", _r("analytics.installments_paid"), "", 1),
        Slot("loan_type", "loan type (charge calc)", _r("profile.loan_type"), "", 2),
    ],
    "account_info": [
        Slot("emi", "EMI amount", _r("analytics.emi_amount"), "", 1),
        Slot("next_due", "next due date", _r("analytics.next_due_date"), "", 1),
        Slot("outstanding", "outstanding", _r("analytics.outstanding_principal"), "", 2),
    ],
    "general_faq": [],
}

# Which knowledge-base topics each intent should ground answers in (for RAG).
INTENT_KB_HINTS: dict[str, str] = {
    "penalty_inquiry": "late payment penalty policy and how penalties are charged",
    "penalty_waiver": "penalty waiver eligibility policy bank failure",
    "payment_failure": "payment failure handling process gateway NACH bounce",
    "settlement_request": "one time settlement policy eligibility credit impact",
    "foreclosure": "foreclosure policy lock-in charges payoff amount",
    "promise_to_pay": "promise to pay hardship assistance policy",
    "interest_paid": "how interest is calculated reducing balance",
    "remaining_emi": "EMI calculation and remaining tenure",
    "account_info": "loan FAQs",
    "general_faq": "loan FAQs",
}

# Tools the orchestrator should consider per intent.
INTENT_TOOLS: dict[str, list[str]] = {
    "remaining_emi": ["get_loan_analytics"],
    "interest_paid": ["get_loan_analytics", "get_payment_history"],
    "penalty_inquiry": ["get_payment_history", "get_gateway_logs", "search_knowledge_base"],
    "payment_failure": ["get_gateway_logs", "search_knowledge_base", "create_payment_link"],
    "penalty_waiver": ["get_gateway_logs", "search_knowledge_base", "create_support_ticket"],
    "promise_to_pay": ["record_commitment", "schedule_callback"],
    "settlement_request": ["search_knowledge_base", "create_support_ticket", "escalate_to_human"],
    "foreclosure": ["get_loan_analytics", "search_knowledge_base", "trigger_workflow"],
    "account_info": ["get_loan_analytics"],
    "general_faq": ["search_knowledge_base"],
}


@dataclass
class DiagnosisResult:
    intent: str
    confidence: float
    known: dict = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)
    kb_query: str | None = None
    suggested_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


class DiagnosisLayer:
    def diagnose(self, utterance: str, context: dict, intent: str | None = None,
                 already_asked: set[str] | None = None) -> DiagnosisResult:
        already_asked = already_asked or set()
        if intent is None:
            intent, conf = classify_intent(utterance)
        else:
            _, conf = classify_intent(utterance)

        slots = INTENT_SLOTS.get(intent, [])
        known: dict = {}
        unknown: list[str] = []
        questions: list[dict] = []

        for slot in slots:
            value = slot.resolver(context) if slot.resolver else None
            if value not in (None, "", []):
                known[slot.name] = value
            else:
                unknown.append(slot.name)
                # Only ask if the slot has a question and we haven't asked it already.
                if slot.question and slot.name not in already_asked:
                    questions.append({
                        "slot": slot.name,
                        "label": slot.label,
                        "question": slot.question,
                        "priority": slot.priority,
                        "ask_only": slot.ask_only,
                    })

        questions.sort(key=lambda q: q["priority"])

        return DiagnosisResult(
            intent=intent,
            confidence=conf,
            known=known,
            unknown=unknown,
            questions=questions,
            kb_query=INTENT_KB_HINTS.get(intent),
            suggested_tools=INTENT_TOOLS.get(intent, []),
        )
