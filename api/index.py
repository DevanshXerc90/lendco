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
import time
from collections import defaultdict, deque
from pathlib import Path

# ---------- project root on sys.path ----------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from api import auth  # noqa: E402
from config import settings  # noqa: E402

# ---------- app setup -------------------------------------------------------
ENVIRONMENT = os.getenv("ENV", "").strip().lower()
API_SECRET = os.getenv("API_SECRET", "").strip()
DEMO_API_KEY = os.getenv("DEMO_API_KEY", "").strip()
if (os.getenv("VERCEL") or ENVIRONMENT == "production") and (not API_SECRET or not DEMO_API_KEY):
    raise RuntimeError("API_SECRET and DEMO_API_KEY are required in production")

MAX_BODY_BYTES = 16 * 1024
RATE_LIMIT = 30
RATE_WINDOW_SECONDS = 60
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)

app = FastAPI(title="LendCo Voice Agent API", docs_url="/api/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",") if x.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


@app.middleware("http")
async def request_limits(request: Request, call_next):
    protected = request.url.path.startswith("/api/") and request.url.path not in {
        "/api/health", "/api/docs", "/api/openapi.json"
    }
    if protected and (DEMO_API_KEY or API_SECRET or os.getenv("VERCEL") or ENVIRONMENT == "production"):
        supplied = request.headers.get("x-api-key", "")
        valid_keys = {key for key in (DEMO_API_KEY, API_SECRET) if key}
        if supplied not in valid_keys:
            return JSONResponse({"detail": "X-API-Key is required"}, status_code=401)
    if request.url.path in {"/api/chat", "/api/scenario"}:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        key = f"{request.client.host if request.client else 'unknown'}:{request.url.path}"
        now = time.monotonic()
        bucket = _rate_buckets[key]
        while bucket and now - bucket[0] >= RATE_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT:
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        bucket.append(now)
    return await call_next(request)


def _require_demo_key(request: Request) -> None:
    """Local development is open; deployed or configured demo mode is keyed."""
    if not DEMO_API_KEY and not API_SECRET and not (os.getenv("VERCEL") or ENVIRONMENT == "production"):
        return
    supplied = request.headers.get("x-api-key", "")
    valid_keys = {key for key in (DEMO_API_KEY, API_SECRET) if key}
    if supplied not in valid_keys:
        raise HTTPException(401, "X-API-Key is required")


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


# ---------------------------------------------------------------- validation --
BORROWER_ID_RE = re.compile(r"^BRW\d{5}$")


def _check_borrower_id(borrower_id: str) -> str:
    """Reject malformed ids early instead of scanning the data with them."""
    if not BORROWER_ID_RE.match(borrower_id):
        raise HTTPException(400, "Invalid borrower id: expected e.g. BRW00001.")
    return borrower_id


def _borrower_or_404(borrower_id: str) -> dict:
    b = next((x for x in _load("borrowers") if x["borrower_id"] == borrower_id), None)
    if not b:
        raise HTTPException(404, f"Borrower {borrower_id} not found")
    return b


def _mask_phone(phone: str) -> str:
    """Show the country code + first digits and the last three only."""
    if len(phone) <= 8:
        return "*" * len(phone)
    return phone[:5] + "*" * (len(phone) - 8) + phone[-3:]


def _mask_pan(pan: str) -> str:
    """PAN policy (see KB): full PAN is never exposed — first 5 and last 1 only."""
    if len(pan) < 10:
        return "****"
    return pan[:5] + "****" + pan[-1]


def _redact_borrower(b: dict) -> dict:
    """Copy of a borrower record with KYC identifiers masked."""
    out = dict(b)
    kyc = dict(out.get("kyc") or {})
    if kyc.get("pan"):
        kyc["pan"] = _mask_pan(kyc["pan"])
    out["kyc"] = kyc
    return out


def _tools(result: dict) -> list[str]:
    """Tool names executed during a turn (orchestrator returns `tool_trace`)."""
    return [t.get("tool") for t in result.get("tool_trace", [])]


def _greeting(start_info: dict) -> str:
    """The session opener (orchestrator.start returns it as `text`)."""
    return start_info.get("greeting") or start_info.get("text", "")


# Fail fast on invalid configuration, then print the running modes (never keys).
settings.validate()
for _line in settings.startup_summary():
    print(_line)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── health ──────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "platforms": 5, "scenarios": 6, "mode": "serverless"}


# ── Auth ──────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    borrower_id: str = Field(min_length=1, max_length=32)
    phone: str = Field(min_length=4, max_length=32)


@app.post("/api/auth/login")
def login(req: LoginRequest):
    """Sign in as a borrower with their registered phone number.

    Returns a bearer token scoped to that borrower; every borrower-scoped
    endpoint then rejects requests for anyone else's records.
    """
    _check_borrower_id(req.borrower_id)
    b = _borrower_or_404(req.borrower_id)

    supplied = re.sub(r"[\s\-]", "", req.phone)
    known = b["phone"]
    if supplied not in (known, known.removeprefix("+91")) and f"+91{supplied}" != known:
        raise HTTPException(401, "Registered phone number does not match our records.")

    token, expires = auth.issue_token(req.borrower_id)
    return {"access_token": token, "token_type": "Bearer",
            "expires_at": expires, "borrower_id": req.borrower_id}


# ── CRM (borrowers) ────────────────────────────────────────────────────────
@app.get("/api/borrowers")
def list_borrowers(x_api_key: str | None = Header(None)):
    auth.require_operator(x_api_key)
    return [
        {
            "borrower_id": b["borrower_id"],
            "name": b["name"],
            "phone": _mask_phone(b["phone"]),  # never list full phone numbers
            "loan_type": b.get("loan_type", "personal"),
            "loan_amount": b.get("loan_amount"),
            "emi_amount": b.get("emi_amount"),
            "delinquency_status": b.get("delinquency_status", "current"),
            "dpd_days": b.get("dpd_days", 0),
        }
        for b in _load("borrowers")
    ]


@app.get("/api/borrowers/{borrower_id}")
def get_borrower(borrower_id: str, authorization: str | None = Header(None),
                 x_api_key: str | None = Header(None)):
    _check_borrower_id(borrower_id)
    auth.check_borrower_access(borrower_id, authorization, x_api_key)
    return _redact_borrower(_borrower_or_404(borrower_id))


# ── Payments ────────────────────────────────────────────────────────────────
@app.get("/api/payments")
def get_payments(borrower_id: str = Query(...), authorization: str | None = Header(None),
                 x_api_key: str | None = Header(None)):
    _check_borrower_id(borrower_id)
    auth.check_borrower_access(borrower_id, authorization, x_api_key)
    return [p for p in _load("payments") if p["borrower_id"] == borrower_id]


@app.get("/api/gateway-logs")
def get_gateway_logs(borrower_id: str = Query(...), authorization: str | None = Header(None),
                      x_api_key: str | None = Header(None)):
    _check_borrower_id(borrower_id)
    auth.check_borrower_access(borrower_id, authorization, x_api_key)
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
def get_tickets(borrower_id: str = Query(None), authorization: str | None = Header(None),
                x_api_key: str | None = Header(None)):
    tickets = _load("tickets")
    if borrower_id:
        _check_borrower_id(borrower_id)
        auth.check_borrower_access(borrower_id, authorization, x_api_key)
        return [t for t in tickets if t["borrower_id"] == borrower_id]
    auth.require_operator(x_api_key)  # unfiltered list spans every borrower
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
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", doc_id):
        raise HTTPException(400, "Invalid document id.")  # never let '..' escape the KB folder
    p = settings.KB_DIR / f"{doc_id}.md"
    if not p.exists():
        raise HTTPException(404, f"Document {doc_id} not found")
    return {"doc_id": doc_id, "content": p.read_text(encoding="utf-8")}


# ── Context Engine analytics ───────────────────────────────────────────────
@app.get("/api/analytics/{borrower_id}")
def get_analytics(borrower_id: str, authorization: str | None = Header(None),
                  x_api_key: str | None = Header(None)):
    _check_borrower_id(borrower_id)
    auth.check_borrower_access(borrower_id, authorization, x_api_key)
    from core.context_engine import ContextEngine
    ctx = ContextEngine().build(borrower_id=borrower_id)
    if not ctx.found:
        raise HTTPException(404, f"Borrower {borrower_id} not found")
    return {            "borrower_id": borrower_id,
            "analytics": ctx.analytics,
            "borrower": ctx.profile,       # UnifiedContext has profile, not borrower
            "operational": {
                "delinquency_status": ctx.delinquency_status,
                "dpd_days": ctx.dpd_days,
                "open_tickets": ctx.open_tickets,
                "recent_failures": ctx.recent_failures,
            },
    }


# ── Agent chat ──────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    borrower_id: str
    message: str = Field(..., min_length=1, max_length=4000)


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    _require_demo_key(request)
    from agents.orchestrator import Orchestrator
    from core.llm import NullLLM
    from core.memory import MemoryManager

    auth.check_borrower_access(req.borrower_id, authorization, x_api_key)
    _borrower_or_404(req.borrower_id)
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    orc = Orchestrator(memory=mem, llm=NullLLM())
    start_info = orc.start(borrower_id=req.borrower_id)
    result = orc.handle(req.message)
    summary = orc.end()
    return {
        "text": result.get("text", ""),
        "intent": result.get("intent", ""),
        "action": result.get("action", ""),
        "tools_used": _tools(result),
        "grounding": result.get("grounding", []),
        "greeting": _greeting(start_info),
        "summary": summary,
    }


# ── Scenario runner ─────────────────────────────────────────────────────────
class ScenarioRequest(BaseModel):
    scenario: str = Field(..., min_length=1, max_length=64)


@app.post("/api/scenario")
def run_scenario(req: ScenarioRequest, request: Request):
    _require_demo_key(request)
    from agents.orchestrator import Orchestrator
    from app.scenarios import PTP_MEMORY, SCRIPTS, pick_borrower
    from core.llm import NullLLM
    from core.memory import MemoryManager

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
                          "tools": _tools(r), "action": r.get("action", ""),
                          "grounding": r.get("grounding", [])})
        s1 = o1.end()

        # Call 2 — same memory, new orchestrator
        o2 = Orchestrator(memory=mem, llm=NullLLM())
        s2_start = o2.start(borrower_id=bid)
        call2 = []
        for msg in PTP_MEMORY["call2"]:
            r = o2.handle(msg)
            call2.append({"user": msg, "agent": r.get("text", ""), "intent": r.get("intent", ""),
                          "tools": _tools(r), "action": r.get("action", ""),
                          "grounding": r.get("grounding", [])})
        s2 = o2.end()

        return {
            "scenario": "ptp_memory",
            "borrower_id": bid,
            "call1": {"greeting": _greeting(s1_start), "turns": call1, "summary": s1},
            "call2": {
                "greeting": _greeting(s2_start),
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
                      "tools": _tools(r), "action": r.get("action", ""),
                      "grounding": r.get("grounding", [])})
    summary = orc.end()
    return {
        "scenario": req.scenario,
        "borrower_id": bid,
        "greeting": _greeting(start_info),
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
