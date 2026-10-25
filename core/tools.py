"""Agent tool registry.

Each tool is a real action the agent can take against the platforms / memory /
RAG. Tools are exposed to the LLM as Anthropic tool-use schemas AND are callable
by the deterministic planner (the offline fallback), so behaviour is identical
with or without an LLM key.

A tool function signature is `fn(args: dict, ctx: ToolContext) -> dict`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import math
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from core import clients  # noqa: E402
from core.rag import get_rag  # noqa: E402


@dataclass
class ToolContext:
    borrower_id: str
    context: dict                      # unified context (analytics, profile, ...)
    memory: object                     # MemoryManager
    session_id: str
    trace: list = field(default_factory=list)
    retrieved_docs: set = field(default_factory=set)  # KB docs actually retrieved this turn


def _log(ctx: ToolContext, name: str, args: dict, result_summary: str):
    ctx.trace.append({"tool": name, "args": args, "result": result_summary})


# ------------------------------------------------------------- argument checks
_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


def _amount(value) -> str | None:
    """A money argument must be a positive, finite number within a sane cap."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "amount must be a number"
    if not math.isfinite(value) or value <= 0:
        return "amount must be a positive number"
    if value > 10_000_000:
        return "amount is above the supported limit"
    return None


def _validate(name: str, args: dict) -> str | None:
    """Return an error message when `args` are not usable, else None.

    Validation happens *before* any tool runs, so bad arguments can never
    reach a platform service (and the LLM gets a clear message back).
    """
    if name == "get_payment_history":
        status = args.get("status")
        if status is not None and status not in settings.PAYMENT_STATUSES:
            return f"status must be one of {list(settings.PAYMENT_STATUSES)}"
    elif name == "search_knowledge_base":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "query must be a non-empty string"
        k = args.get("k", 3)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
            return "k must be an integer between 1 and 10"
    elif name == "create_payment_link":
        if args.get("amount") is not None:
            err = _amount(args["amount"])
            if err:
                return err
    elif name == "record_commitment":
        date = args.get("date")
        if not isinstance(date, str) or not date.strip() or len(date) > 40:
            return "date must be a short, non-empty string (e.g. 2026-11-05)"
        if args.get("amount") is not None:
            err = _amount(args["amount"])
            if err:
                return err
    elif name == "create_support_ticket":
        if not isinstance(args.get("category"), str) or not _CATEGORY_RE.match(args["category"]):
            return "category must be a short lowercase identifier (e.g. penalty_waiver)"
        if not isinstance(args.get("subject"), str) or not args["subject"].strip():
            return "subject must be a non-empty string"
        if args.get("priority") is not None and args["priority"] not in settings.TICKET_PRIORITIES:
            return f"priority must be one of {list(settings.TICKET_PRIORITIES)}"
    elif name == "escalate_to_human":
        if not isinstance(args.get("reason"), str) or not args["reason"].strip():
            return "reason must be a non-empty string"
        if args.get("priority") is not None and args["priority"] not in settings.TICKET_PRIORITIES:
            return f"priority must be one of {list(settings.TICKET_PRIORITIES)}"
    elif name == "update_crm":
        if args.get("field") not in settings.CRM_EDITABLE_FIELDS:
            return f"field must be one of {list(settings.CRM_EDITABLE_FIELDS)}"
        if not isinstance(args.get("value"), str):
            return "value must be a string"
    elif name == "trigger_workflow":
        if args.get("name") not in settings.WORKFLOW_NAMES:
            return f"name must be one of {list(settings.WORKFLOW_NAMES)}"
    return None


# Tools with a real-world side effect — at most one call of each per turn.
SIDE_EFFECT_TOOLS = {
    "create_support_ticket", "create_payment_link", "record_commitment",
    "schedule_callback", "escalate_to_human", "update_crm", "trigger_workflow",
}


# --------------------------------------------------------------- tool functions
def get_loan_analytics(args, ctx: ToolContext) -> dict:
    a = ctx.context.get("analytics", {})
    p = ctx.context.get("profile", {})
    out = {
        "loan_id": p.get("loan_id"), "loan_type": p.get("loan_type"),
        "loan_amount": p.get("loan_amount"), "interest_rate": p.get("interest_rate"),
        "emi_amount": a.get("emi_amount"), "tenure_months": a.get("tenure_months"),
        "installments_paid": a.get("installments_paid"),
        "installments_remaining": a.get("installments_remaining"),
        "interest_paid": a.get("interest_paid"), "principal_paid": a.get("principal_paid"),
        "outstanding_principal": a.get("outstanding_principal"),
        "next_due_date": a.get("next_due_date"), "overdue_amount": a.get("overdue_amount"),
        "total_penalty_charged": a.get("total_penalty_charged"),
        "delinquency_status": ctx.context.get("delinquency_status"),
        "dpd_days": ctx.context.get("dpd_days"),
    }
    _log(ctx, "get_loan_analytics", args, f"remaining={out['installments_remaining']} outstanding={out['outstanding_principal']}")
    return out


def get_payment_history(args, ctx: ToolContext) -> dict:
    status = args.get("status")
    items = clients.payments.history(ctx.borrower_id, status=status)
    summary = {
        "count": len(items),
        "by_status": {},
        "recent": items[-6:],
    }
    for p in items:
        summary["by_status"][p["status"]] = summary["by_status"].get(p["status"], 0) + 1
    _log(ctx, "get_payment_history", args, f"{summary['by_status']}")
    return summary


def get_gateway_logs(args, ctx: ToolContext) -> dict:
    logs = clients.payments.gateway_logs(ctx.borrower_id)
    latest = logs[-1] if logs else None
    out = {"count": len(logs), "logs": logs[-5:], "latest_failure": latest}
    _log(ctx, "get_gateway_logs", args, f"latest={latest['failure_reason'] if latest else None} root={latest['root_cause'] if latest else None}")
    return out


def search_knowledge_base(args, ctx: ToolContext) -> dict:
    query = args.get("query", "")
    hits = get_rag().retrieve(query, k=args.get("k", 3))
    # record what was actually retrieved — citations are checked against this
    for h in hits:
        ctx.retrieved_docs.add(h.doc_id)
    out = {"query": query, "results": [{"doc_id": h.doc_id, "title": h.title, "text": h.text, "score": h.score} for h in hits]}
    if not hits:
        out["note"] = "no matching policy document found — do not claim a policy here"
    _log(ctx, "search_knowledge_base", args, f"hits={[h.doc_id for h in hits]}")
    return out


def create_support_ticket(args, ctx: ToolContext) -> dict:
    res = clients.support.create_ticket(
        borrower_id=ctx.borrower_id,
        category=args.get("category", "general"),
        subject=args.get("subject", "Borrower request"),
        description=args.get("description", ""),
        loan_id=ctx.context.get("profile", {}).get("loan_id"),
        priority=args.get("priority", "medium"),
    )
    _log(ctx, "create_support_ticket", args, f"ticket={res.get('ticket_id')}")
    return res


def create_payment_link(args, ctx: ToolContext) -> dict:
    amount = args.get("amount") or ctx.context.get("analytics", {}).get("emi_amount", 0)
    res = clients.payments.create_payment_link(ctx.borrower_id, amount, args.get("purpose", "EMI payment"))
    _log(ctx, "create_payment_link", args, f"link={res.get('link_id')}")
    return res


def record_commitment(args, ctx: ToolContext) -> dict:
    res = ctx.memory.record_commitment(
        ctx.borrower_id,
        date=args.get("date", "unspecified"),
        amount=args.get("amount") or ctx.context.get("analytics", {}).get("emi_amount", 0),
        reason=args.get("reason", "not stated"),
    )
    _log(ctx, "record_commitment", args, f"ptp={res.get('date')} amt={res.get('amount')}")
    return res


def schedule_callback(args, ctx: ToolContext) -> dict:
    res = clients.workflow.schedule_callback(
        ctx.borrower_id, reason=args.get("reason", "follow-up"),
        preferred_window=args.get("preferred_window", "tomorrow 10:00-12:00"),
    )
    _log(ctx, "schedule_callback", args, f"cb={res.get('callback_id')}")
    return res


def escalate_to_human(args, ctx: ToolContext) -> dict:
    res = clients.workflow.escalate(
        ctx.borrower_id, reason=args.get("reason", "borrower request"),
        priority=args.get("priority", "high"), context=args.get("context"),
    )
    _log(ctx, "escalate_to_human", args, f"esc={res.get('escalation_id')}")
    return res


def update_crm(args, ctx: ToolContext) -> dict:
    res = clients.crm.update(ctx.borrower_id, args.get("field", ""), args.get("value", ""), args.get("note"))
    _log(ctx, "update_crm", args, f"updated {args.get('field')}")
    return res


def trigger_workflow(args, ctx: ToolContext) -> dict:
    res = clients.workflow.trigger(args.get("name", "generic"), ctx.borrower_id, args.get("params", {}))
    _log(ctx, "trigger_workflow", args, f"wf={args.get('name')} run={res.get('run_id')}")
    return res


# --------------------------------------------------------------- tool schemas
REGISTRY = {
    "get_loan_analytics": get_loan_analytics,
    "get_payment_history": get_payment_history,
    "get_gateway_logs": get_gateway_logs,
    "search_knowledge_base": search_knowledge_base,
    "create_support_ticket": create_support_ticket,
    "create_payment_link": create_payment_link,
    "record_commitment": record_commitment,
    "schedule_callback": schedule_callback,
    "escalate_to_human": escalate_to_human,
    "update_crm": update_crm,
    "trigger_workflow": trigger_workflow,
}

SCHEMAS = [
    {"name": "get_loan_analytics", "description": "Get the borrower's loan summary: EMIs paid/remaining, interest & principal paid, outstanding principal, next due date, overdue amount, penalties, delinquency.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_payment_history", "description": "Get the borrower's payment history with status breakdown and recent installments.",
     "input_schema": {"type": "object", "properties": {"status": {"type": "string", "description": "optional filter: success|failed|partial|missed|auto_debit_failed"}}}},
    {"name": "get_gateway_logs", "description": "Get raw payment-gateway logs for failed/partial payments, including response codes, failure reason and root cause (system vs customer). Use this to diagnose payment failures and penalty disputes.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "search_knowledge_base", "description": "Search company policy & FAQ documents (RAG). Use to ground any policy explanation (penalties, waivers, foreclosure, settlement, failures). Always cite what you retrieve.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}},
    {"name": "create_support_ticket", "description": "Create a support ticket when human review/approval is needed (e.g., borderline penalty waiver, settlement, formal grievance).",
     "input_schema": {"type": "object", "properties": {"category": {"type": "string"}, "subject": {"type": "string"}, "description": {"type": "string"}, "priority": {"type": "string"}}, "required": ["category", "subject", "description"]}},
    {"name": "create_payment_link", "description": "Generate a payment link for the borrower to pay an EMI or overdue amount.",
     "input_schema": {"type": "object", "properties": {"amount": {"type": "number"}, "purpose": {"type": "string"}}}},
    {"name": "record_commitment", "description": "Record a promise-to-pay: the committed date, amount and reason for delay. Use whenever the borrower commits to pay later.",
     "input_schema": {"type": "object", "properties": {"date": {"type": "string"}, "amount": {"type": "number"}, "reason": {"type": "string"}}, "required": ["date"]}},
    {"name": "schedule_callback", "description": "Schedule a follow-up callback at the borrower's preferred time.",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}, "preferred_window": {"type": "string"}}}},
    {"name": "escalate_to_human", "description": "Route the case to a human agent for issues beyond policy or requiring authority.",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}, "priority": {"type": "string"}, "context": {"type": "string"}}, "required": ["reason"]}},
    {"name": "update_crm", "description": "Update a CRM field on the borrower record (e.g., preferred language, contact note).",
     "input_schema": {"type": "object", "properties": {"field": {"type": "string"}, "value": {"type": "string"}, "note": {"type": "string"}}, "required": ["field", "value"]}},
    {"name": "trigger_workflow", "description": "Trigger a back-office automation, e.g. 'reregister_mandate', 'send_foreclosure_statement', 'initiate_penalty_reversal'.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "params": {"type": "object"}}, "required": ["name"]}},
]

SCHEMA_BY_NAME = {s["name"]: s for s in SCHEMAS}


def execute(name: str, args: dict, ctx: ToolContext) -> dict:
    """Run one registered tool with validated arguments.

    Unknown tools, bad arguments and repeated side effects in the same turn
    are refused here, before anything reaches a platform service.
    """
    fn = REGISTRY.get(name)
    if not fn:
        return {"error": f"unknown tool '{name}'"}
    args = args or {}
    err = _validate(name, args)
    if err:
        return {"error": err}
    if name in SIDE_EFFECT_TOOLS and any(t["tool"] == name for t in ctx.trace):
        return {"duplicate": True, "tool": name,
                "note": f"{name} was already executed this turn; not repeating it"}
    try:
        return fn(args, ctx)
    except Exception as e:  # tools must never crash the agent loop
        return {"error": str(e)}
