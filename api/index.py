"""LendCo Voice Agent — Vercel Serverless API.

Single FastAPI app that serves platform data, context engine analytics,
agent chat (deterministic planner), evaluation results, and dashboard
analytics. All 5 platform endpoints are consolidated into one serverless
function with local-data fallback (no running services required).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# ---------- project root on sys.path ----------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from config import settings  # noqa: E402

# ---------- app setup -------------------------------------------------------
app = FastAPI(title="LendCo Voice Agent API", docs_url="/api/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- data helpers ----------------------------------------------------
def _ensure_data():
    """Generate synthetic data if the generated directory is empty."""
    if not (settings.DATA_DIR / "borrowers.json").exists():
        from data.generate import main as gen
        gen()


def _load(name: str) -> list[dict]:
    _ensure_data()
    p = settings.DATA_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── health ──────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "platforms": 5, "scenarios": 6, "mode": "serverless"}


# ── CRM (borrowers) ────────────────────────────────────────────────────────
@app.get("/api/borrowers")
def list_borrowers():
    return [
        {
            "borrower_id": b["borrower_id"],
            "name": b["name"],
            "phone": b["phone"],
            "loan_type": b.get("loan_type", "personal"),
            "loan_amount": b.get("loan_amount"),
            "emi_amount": b.get("emi_amount"),
            "delinquency_status": b.get("delinquency_status", "current"),
            "dpd_days": b.get("dpd_days", 0),
        }
        for b in _load("borrowers")
    ]


@app.get("/api/borrowers/{borrower_id}")
def get_borrower(borrower_id: str):
    b = next((x for x in _load("borrowers") if x["borrower_id"] == borrower_id), None)
    if not b:
        raise HTTPException(404, f"Borrower {borrower_id} not found")
    return b


# ── Payments ────────────────────────────────────────────────────────────────
@app.get("/api/payments")
def get_payments(borrower_id: str = Query(...)):
    return [p for p in _load("payments") if p["borrower_id"] == borrower_id]


@app.get("/api/gateway-logs")
def get_gateway_logs(borrower_id: str = Query(...)):
    return [
        {
            "payment_id": p["payment_id"],
            "due_date": p["due_date"],
            "status": p["status"],
            "method": p.get("method", ""),
            "gateway": p.get("gateway", ""),
            "response_code": p.get("gateway_response_code", p.get("response_code", "")),
            "failure_reason": p.get("failure_reason", ""),
            "root_cause": p.get("root_cause", ""),
            "penalty_charged": p.get("penalty_charged", 0.0),
        }
        for p in _load("payments")
        if p["borrower_id"] == borrower_id and p["status"] != "success"
    ]


# ── Support (tickets) ──────────────────────────────────────────────────────
@app.get("/api/tickets")
def get_tickets(borrower_id: str = Query(None)):
    tickets = _load("tickets")
    if borrower_id:
        return [t for t in tickets if t["borrower_id"] == borrower_id]
    return tickets[:50]  # cap for list view


# ── Knowledge Base ──────────────────────────────────────────────────────────
@app.get("/api/documents")
def list_documents():
    _ensure_data()
    return [
        {"doc_id": p.stem, "filename": p.name}
        for p in sorted(settings.KB_DIR.glob("*.md"))
    ]


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    _ensure_data()
    p = settings.KB_DIR / f"{doc_id}.md"
    if not p.exists():
        raise HTTPException(404, f"Document {doc_id} not found")
    return {"doc_id": doc_id, "content": p.read_text(encoding="utf-8")}


# ── Context Engine analytics ───────────────────────────────────────────────
@app.get("/api/analytics/{borrower_id}")
def get_analytics(borrower_id: str):
    from core.context_engine import ContextEngine
    ctx = ContextEngine().build(borrower_id=borrower_id)
    if not ctx.found:
        raise HTTPException(404, f"Borrower {borrower_id} not found")
    return {
        "borrower_id": borrower_id,
        "analytics": ctx.analytics,
        "borrower": ctx.borrower,
        "operational": ctx.operational,
    }


# ── Agent chat ──────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    borrower_id: str
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    from core.memory import MemoryManager
    from core.llm import NullLLM
    from agents.orchestrator import Orchestrator

    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    orc = Orchestrator(memory=mem, llm=NullLLM())
    start_info = orc.start(borrower_id=req.borrower_id)
    result = orc.handle(req.message)
    summary = orc.end()
    return {
        "text": result.get("text", ""),
        "intent": result.get("intent", ""),
        "action": result.get("action", ""),
        "tools_used": result.get("tools_used", []),
        "grounding": result.get("grounding", []),
        "greeting": start_info.get("greeting", ""),
        "summary": summary,
    }


# ── Scenario runner ─────────────────────────────────────────────────────────
class ScenarioRequest(BaseModel):
    scenario: str


@app.post("/api/scenario")
def run_scenario(req: ScenarioRequest):
    from app.scenarios import SCRIPTS, PTP_MEMORY, pick_borrower
    from core.memory import MemoryManager
    from core.llm import NullLLM
    from agents.orchestrator import Orchestrator

    if req.scenario == "ptp_memory":
        mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
        bid = pick_borrower("ptp_memory")

        # Call 1
        o1 = Orchestrator(memory=mem, llm=NullLLM())
        s1_start = o1.start(borrower_id=bid)
        call1 = []
        for msg in PTP_MEMORY["call1"]:
            r = o1.handle(msg)
            call1.append({"user": msg, "agent": r.get("text", ""), "intent": r.get("intent", ""),
                          "tools": r.get("tools_used", []), "action": r.get("action", "")})
        s1 = o1.end()

        # Call 2 — same memory, new orchestrator
        o2 = Orchestrator(memory=mem, llm=NullLLM())
        s2_start = o2.start(borrower_id=bid)
        call2 = []
        for msg in PTP_MEMORY["call2"]:
            r = o2.handle(msg)
            call2.append({"user": msg, "agent": r.get("text", ""), "intent": r.get("intent", ""),
                          "tools": r.get("tools_used", []), "action": r.get("action", "")})
        s2 = o2.end()

        return {
            "scenario": "ptp_memory",
            "borrower_id": bid,
            "call1": {"greeting": s1_start.get("greeting", ""), "turns": call1, "summary": s1},
            "call2": {
                "greeting": s2_start.get("greeting", ""),
                "memory_aware": s2_start.get("memory_aware", False),
                "turns": call2,
                "summary": s2,
            },
        }

    # Single-call scenarios
    scripts = SCRIPTS
    if req.scenario not in scripts:
        raise HTTPException(400, f"Unknown scenario: {req.scenario}")
    bid = pick_borrower(req.scenario)
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    orc = Orchestrator(memory=mem, llm=NullLLM())
    start_info = orc.start(borrower_id=bid)
    turns = []
    for msg in scripts[req.scenario]:
        r = orc.handle(msg)
        turns.append({"user": msg, "agent": r.get("text", ""), "intent": r.get("intent", ""),
                      "tools": r.get("tools_used", []), "action": r.get("action", "")})
    summary = orc.end()
    return {
        "scenario": req.scenario,
        "borrower_id": bid,
        "greeting": start_info.get("greeting", ""),
        "turns": turns,
        "summary": summary,
    }


# ── Scenarios list ──────────────────────────────────────────────────────────
@app.get("/api/scenarios")
def list_scenarios():
    return [
        {"id": "remaining",       "name": "Remaining EMI",       "icon": "📊", "description": "How many EMIs are left + next due date"},
        {"id": "interest",        "name": "Interest Paid",       "icon": "💰", "description": "Interest vs principal breakdown"},
        {"id": "penalty",         "name": "Penalty Inquiry",     "icon": "⚠️",  "description": "Why a penalty was charged + RAG policy"},
        {"id": "payment_failure", "name": "Payment Failure",     "icon": "🔴", "description": "Gateway logs + root cause diagnosis"},
        {"id": "waiver",          "name": "Penalty Waiver",      "icon": "✅", "description": "Root cause → eligibility → auto-reversal"},
        {"id": "ptp_memory",      "name": "Memory Demo",         "icon": "🧠", "description": "Two calls showing memory persistence"},
    ]


# ── Evaluation results ─────────────────────────────────────────────────────
@app.get("/api/eval")
def get_eval():
    p = ROOT / "eval" / "results.json"
    if not p.exists():
        return {"error": "eval/results.json not found. Run: python -m eval.evaluator"}
    return json.loads(p.read_text(encoding="utf-8"))


# ── Dashboard data (portfolio + conversation analytics) ─────────────────────
@app.get("/api/dashboard-data")
def dashboard_data():
    borrowers = _load("borrowers")
    payments = _load("payments")
    tickets = _load("tickets")
    conversations = _load("conversations")

    # Portfolio metrics
    total_loan = sum(b.get("loan_amount", 0) for b in borrowers)
    avg_interest = (sum(b.get("interest_rate", 0) for b in borrowers) / len(borrowers)) if borrowers else 0
    auto_debit = (sum(1 for b in borrowers if b.get("auto_debit")) / len(borrowers) * 100) if borrowers else 0
    ptp_count = sum(1 for c in conversations if c.get("promise_to_pay"))

    # Distributions
    delinquency: dict[str, int] = defaultdict(int)
    for b in borrowers:
        delinquency[b.get("delinquency_status", "current")] += 1

    payment_status: dict[str, int] = defaultdict(int)
    root_cause: dict[str, int] = defaultdict(int)
    failure_modes: dict[str, int] = defaultdict(int)
    for p in payments:
        payment_status[p["status"]] += 1
        if p["status"] != "success":
            root_cause[p.get("root_cause", "unknown")] += 1
            failure_modes[p.get("failure_reason", "unknown")] += 1

    intent_dist: dict[str, int] = defaultdict(int)
    sentiment_dist: dict[str, int] = defaultdict(int)
    for c in conversations:
        intent_dist[c.get("intent", "unknown")] += 1
        sentiment_dist[c.get("sentiment", "neutral")] += 1

    # Eval results
    eval_path = ROOT / "eval" / "results.json"
    eval_results = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None

    return {
        "portfolio": {
            "total_borrowers": len(borrowers),
            "total_loan_book": total_loan,
            "avg_interest": round(avg_interest, 2),
            "auto_debit_pct": round(auto_debit, 1),
            "total_payments": len(payments),
            "total_conversations": len(conversations),
            "total_tickets": len(tickets),
            "ptp_count": ptp_count,
        },
        "delinquency": dict(sorted(delinquency.items(), key=lambda x: -x[1])),
        "payment_status": dict(sorted(payment_status.items(), key=lambda x: -x[1])),
        "root_cause": dict(sorted(root_cause.items(), key=lambda x: -x[1])),
        "failure_modes": dict(sorted(failure_modes.items(), key=lambda x: -x[1])),
        "intent_distribution": dict(sorted(intent_dist.items(), key=lambda x: -x[1])),
        "sentiment_distribution": dict(sorted(sentiment_dist.items(), key=lambda x: -x[1])),
        "eval": eval_results,
    }
