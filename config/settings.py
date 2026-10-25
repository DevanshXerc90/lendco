"""Central configuration. Reads from environment / .env with safe fallbacks."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "generated"
KB_DIR = DATA_DIR / "kb"
MEMORY_DB = ROOT / "data" / "memory.db"

# --- LLM ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-4-8")

# --- Embeddings ---
EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "tfidf")

# --- Voice ---
STT_BACKEND = os.getenv("STT_BACKEND", "text")
TTS_BACKEND = os.getenv("TTS_BACKEND", "print")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")

# --- Real API integrations (optional — falls back to mock services) ---
HUBSPOT_ACCESS_TOKEN = os.getenv("HUBSPOT_ACCESS_TOKEN", "").strip()
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "").strip()
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "").strip()

# Backend selection: "hubspot" | "mock" (auto-detect from key presence)
CRM_BACKEND = os.getenv("CRM_BACKEND", "hubspot" if HUBSPOT_ACCESS_TOKEN else "mock")

# Backend selection: "stripe" | "razorpay" | "mock" (auto-detect from key presence)
def _detect_payments_backend() -> str:
    override = os.getenv("PAYMENTS_BACKEND", "").strip()
    if override in ("stripe", "razorpay", "mock"):
        return override
    if STRIPE_SECRET_KEY:
        return "stripe"
    if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
        return "razorpay"
    return "mock"

PAYMENTS_BACKEND = _detect_payments_backend()

# --- Platform service URLs ---
PLATFORM_URLS = {
    "crm": os.getenv("CRM_URL", "http://127.0.0.1:8101"),
    "payments": os.getenv("PAYMENTS_URL", "http://127.0.0.1:8102"),
    "support": os.getenv("SUPPORT_URL", "http://127.0.0.1:8103"),
    "knowledge": os.getenv("KNOWLEDGE_URL", "http://127.0.0.1:8104"),
    "workflow": os.getenv("WORKFLOW_URL", "http://127.0.0.1:8105"),
}

PLATFORM_PORTS = {
    "crm": 8101,
    "payments": 8102,
    "support": 8103,
    "knowledge": 8104,
    "workflow": 8105,
}

# Deterministic seed so synthetic data + demos are reproducible.
SEED = int(os.getenv("SEED", "42"))

# --- API authentication (portfolio-level safeguards, see README) ---
# Signs borrower session tokens issued by POST /api/auth/login. Empty means a
# per-process random key is used, so no secret is ever committed to source.
API_SECRET = os.getenv("API_SECRET", "").strip()

# Operator key for the demo console endpoints (/api/borrowers, /api/scenario).
# Empty = demo mode: those endpoints are open and a warning is printed.
DEMO_API_KEY = os.getenv("DEMO_API_KEY", "").strip()

# Shared key between the agent and the internal platform services (sent as
# X-Platform-Key). Empty = the services rely on loopback binding alone.
PLATFORM_API_KEY = os.getenv("PLATFORM_API_KEY", "").strip()

# --- Tool / platform action allowlists (enforced by core.tools *and* the
# --- platform services, so a stray LLM tool call cannot do the unexpected) ---
# CRM fields an agent may edit. Identity and loan facts are read-only.
CRM_EDITABLE_FIELDS = ("preferred_language", "email", "contact_note")

# Automations the workflow platform knows how to run.
WORKFLOW_NAMES = ("generic", "reregister_mandate", "send_foreclosure_statement",
                  "initiate_penalty_reversal")

# Ticket priorities understood by the support platform.
TICKET_PRIORITIES = ("low", "medium", "high", "urgent")

# Payment-history status filter values.
PAYMENT_STATUSES = ("success", "failed", "partial", "missed", "auto_debit_failed")

# Values each backend setting accepts. Anything else is a configuration error,
# not something to guess at silently.
VALID_BACKENDS = {
    "EMBEDDINGS_BACKEND": ("tfidf", "sentence-transformers"),
    "STT_BACKEND": ("text", "whisper"),
    "TTS_BACKEND": ("print", "pyttsx3"),
    "CRM_BACKEND": ("hubspot", "mock"),
    "PAYMENTS_BACKEND": ("stripe", "razorpay", "mock"),
}


def validate() -> None:
    """Fail fast with a clear message when configuration is invalid.

    Called at application startup. Missing optional API keys are *not* fatal —
    the app is designed to run offline with its mock platforms — but a backend
    name nobody supports must never be guessed at.
    """
    problems = []
    for name, allowed in VALID_BACKENDS.items():
        value = globals()[name]
        if value not in allowed:
            problems.append(f"{name}={value!r} is not supported; expected one of {list(allowed)}")
    if problems:
        raise RuntimeError("Invalid configuration:\n  - " + "\n  - ".join(problems))


def startup_summary() -> list[str]:
    """Human-readable startup state — modes only, never key values."""
    lines = []
    if not ANTHROPIC_API_KEY:
        lines.append("[config] ANTHROPIC_API_KEY not set - using the rule-based planner.")
    if not API_SECRET:
        lines.append("[config] API_SECRET not set - borrower tokens reset when the API restarts.")
    if not DEMO_API_KEY:
        lines.append("[config] DEMO_API_KEY not set - operator endpoints are open (demo mode).")
    lines.append(f"[config] LLM={LLM_MODEL if ANTHROPIC_API_KEY else 'rule-based'} "
                 f"embeddings={EMBEDDINGS_BACKEND} payments={PAYMENTS_BACKEND} crm={CRM_BACKEND}")
    return lines
