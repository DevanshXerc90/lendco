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
