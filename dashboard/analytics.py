"""Compute analytics from datasets + runtime side-effects + memory + eval results.

Pure-Python (stdlib only) so it has no install risk. Consumed by both the HTML
report generator and the optional Streamlit app.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402


def _load(name):
    p = settings.DATA_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _runtime(name):
    p = settings.ROOT / "data" / "runtime" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def portfolio() -> dict:
    b = _load("borrowers")
    delinq = Counter(x["delinquency_status"] for x in b)
    return {
        "borrowers": len(b),
        "loan_book": sum(x["loan_amount"] for x in b),
        "avg_interest": round(sum(x["interest_rate"] for x in b) / max(len(b), 1), 2),
        "delinquency": dict(delinq),
        "auto_debit_pct": round(100 * sum(1 for x in b if x["auto_debit_enabled"]) / max(len(b), 1), 1),
    }


def payments_summary() -> dict:
    p = _load("payments")
    status = Counter(x["status"] for x in p)
    root = Counter(x["root_cause"] for x in p if x["status"] != "success")
    failure_modes = Counter(x["failure_reason"] for x in p if x["failure_reason"])
    return {
        "total": len(p),
        "by_status": dict(status),
        "failure_root_cause": dict(root),
        "failure_modes": dict(failure_modes),
    }


def operations() -> dict:
    return {
        "tickets_created": len(_runtime("tickets")),
        "payment_links": len(_runtime("payment_links")),
        "callbacks": len(_runtime("callbacks")),
        "escalations": len(_runtime("escalations")),
        "workflow_runs": len(_runtime("workflow_runs")),
        "crm_updates": len(_runtime("crm_updates")),
    }


def agent_learning() -> list[dict]:
    if not settings.MEMORY_DB.exists():
        return []
    conn = sqlite3.connect(settings.MEMORY_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT intent,resolution_path,success_count,fail_count,total_turns,runs FROM agent_memory"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    out = []
    for r in rows:
        runs = r["runs"] or 1
        out.append({
            "intent": r["intent"], "resolution_path": r["resolution_path"],
            "runs": r["runs"], "success_rate": round(r["success_count"] / runs, 2),
            "avg_turns": round(r["total_turns"] / runs, 1),
        })
    return sorted(out, key=lambda x: (x["intent"], -x["success_rate"]))


def conversation_analytics() -> dict:
    """From seeded conversation history — sentiment + intent distribution."""
    c = _load("conversations")
    return {
        "total": len(c),
        "intents": dict(Counter(x["intent"] for x in c)),
        "sentiment": dict(Counter(x["sentiment"] for x in c)),
        "resolved_pct": round(100 * sum(1 for x in c if x["resolved"]) / max(len(c), 1), 1),
        "promises_to_pay": sum(1 for x in c if x.get("promise_to_pay")),
    }


def eval_results() -> dict:
    p = settings.ROOT / "eval" / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def all_analytics() -> dict:
    return {
        "portfolio": portfolio(),
        "payments": payments_summary(),
        "operations": operations(),
        "agent_learning": agent_learning(),
        "conversations": conversation_analytics(),
        "eval": eval_results(),
    }
