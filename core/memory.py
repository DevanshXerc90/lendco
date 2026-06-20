"""Memory & Continuous Learning.

Three tiers persisted in SQLite (stdlib, zero deps):

  Borrower Memory     : preferences, communication style, payment commitments,
                        frequently-asked questions.
  Conversation Memory : per-session questions asked, answers received,
                        outstanding issues, resolution history.
  Agent Memory        : aggregate success/failure of resolution paths per intent
                        (which workflows actually resolve which intents) — this
                        is what makes the agent improve across borrowers.

The orchestrator calls `recall()` at the START of every interaction. Because
recall surfaces preferences, open commitments, prior Q&A, and the best-known
resolution path, the *second* interaction is faster (skip re-discovery), more
personalised (style + language), more accurate (known root causes), and more
outcome-oriented (proven resolution path).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

NOW = lambda: datetime(2026, 6, 9, 10, 0, 0).isoformat()  # deterministic clock for demos

SCHEMA = """
CREATE TABLE IF NOT EXISTS borrower_memory (
  borrower_id TEXT, key TEXT, value TEXT, updated_at TEXT,
  PRIMARY KEY (borrower_id, key)
);
CREATE TABLE IF NOT EXISTS commitments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  borrower_id TEXT, date TEXT, amount REAL, reason TEXT,
  kept INTEGER, created_at TEXT
);
CREATE TABLE IF NOT EXISTS faqs (
  borrower_id TEXT, question TEXT, intent TEXT, count INTEGER, last_asked TEXT,
  PRIMARY KEY (borrower_id, intent)
);
CREATE TABLE IF NOT EXISTS conversations (
  session_id TEXT PRIMARY KEY,
  borrower_id TEXT, intent TEXT, started_at TEXT, ended_at TEXT,
  resolved INTEGER, sentiment TEXT, turns INTEGER,
  transcript TEXT, outstanding TEXT
);
CREATE TABLE IF NOT EXISTS qa_pairs (
  session_id TEXT, borrower_id TEXT, slot TEXT, question TEXT, answer TEXT, asked_at TEXT
);
CREATE TABLE IF NOT EXISTS agent_memory (
  intent TEXT, resolution_path TEXT,
  success_count INTEGER, fail_count INTEGER, total_turns INTEGER, runs INTEGER,
  updated_at TEXT,
  PRIMARY KEY (intent, resolution_path)
);
"""


class MemoryManager:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or settings.MEMORY_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------- Borrower memory
    def set_preference(self, borrower_id: str, key: str, value) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO borrower_memory(borrower_id,key,value,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(borrower_id,key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (borrower_id, key, json.dumps(value), NOW()),
            )

    def get_preferences(self, borrower_id: str) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT key,value FROM borrower_memory WHERE borrower_id=?", (borrower_id,)).fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # ---------------------------------------------------- Commitments (PTP)
    def record_commitment(self, borrower_id: str, date: str, amount: float, reason: str) -> dict:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO commitments(borrower_id,date,amount,reason,kept,created_at) VALUES(?,?,?,?,?,?)",
                (borrower_id, date, amount, reason, None, NOW()),
            )
            cid = cur.lastrowid
        return {"id": cid, "borrower_id": borrower_id, "date": date, "amount": amount, "reason": reason, "kept": None}

    def get_commitments(self, borrower_id: str, only_open: bool = False) -> list[dict]:
        q = "SELECT id,date,amount,reason,kept,created_at FROM commitments WHERE borrower_id=?"
        if only_open:
            q += " AND kept IS NULL"
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY created_at DESC", (borrower_id,)).fetchall()
        return [dict(r) for r in rows]

    def resolve_commitment(self, commitment_id: int, kept: bool) -> None:
        with self._conn() as c:
            c.execute("UPDATE commitments SET kept=? WHERE id=?", (1 if kept else 0, commitment_id))

    # ---------------------------------------------------- FAQs
    def record_faq(self, borrower_id: str, intent: str, question: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO faqs(borrower_id,question,intent,count,last_asked) VALUES(?,?,?,1,?) "
                "ON CONFLICT(borrower_id,intent) DO UPDATE SET count=count+1, last_asked=excluded.last_asked, question=excluded.question",
                (borrower_id, question, intent, NOW()),
            )

    def get_faqs(self, borrower_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT intent,question,count,last_asked FROM faqs WHERE borrower_id=? ORDER BY count DESC", (borrower_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------- Conversation memory
    def record_qa(self, session_id: str, borrower_id: str, slot: str, question: str, answer: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO qa_pairs(session_id,borrower_id,slot,question,answer,asked_at) VALUES(?,?,?,?,?,?)",
                (session_id, borrower_id, slot, question, answer, NOW()),
            )

    def get_asked_slots(self, borrower_id: str) -> set[str]:
        with self._conn() as c:
            rows = c.execute("SELECT DISTINCT slot FROM qa_pairs WHERE borrower_id=? AND slot IS NOT NULL", (borrower_id,)).fetchall()
        return {r["slot"] for r in rows}

    def save_conversation(self, session: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO conversations"
                "(session_id,borrower_id,intent,started_at,ended_at,resolved,sentiment,turns,transcript,outstanding) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    session["session_id"], session["borrower_id"], session.get("intent"),
                    session.get("started_at"), session.get("ended_at", NOW()),
                    1 if session.get("resolved") else 0, session.get("sentiment"),
                    session.get("turns", 0), json.dumps(session.get("transcript", [])),
                    json.dumps(session.get("outstanding", [])),
                ),
            )

    def past_conversations(self, borrower_id: str, limit: int = 5) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT session_id,intent,started_at,resolved,sentiment,turns,outstanding "
                "FROM conversations WHERE borrower_id=? ORDER BY started_at DESC LIMIT ?",
                (borrower_id, limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["outstanding"] = json.loads(d["outstanding"]) if d["outstanding"] else []
            out.append(d)
        return out

    # ---------------------------------------------------- Agent memory (learning)
    def record_resolution(self, intent: str, resolution_path: str, success: bool, turns: int) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO agent_memory(intent,resolution_path,success_count,fail_count,total_turns,runs,updated_at) "
                "VALUES(?,?,?,?,?,1,?) "
                "ON CONFLICT(intent,resolution_path) DO UPDATE SET "
                "success_count=success_count+?, fail_count=fail_count+?, total_turns=total_turns+?, runs=runs+1, updated_at=?",
                (intent, resolution_path, 1 if success else 0, 0 if success else 1, turns, NOW(),
                 1 if success else 0, 0 if success else 1, turns, NOW()),
            )

    def best_resolution_path(self, intent: str) -> dict | None:
        with self._conn() as c:
            rows = c.execute(
                "SELECT resolution_path,success_count,fail_count,total_turns,runs FROM agent_memory WHERE intent=?",
                (intent,),
            ).fetchall()
        best, best_score = None, -1.0
        for r in rows:
            runs = r["runs"] or 1
            success_rate = r["success_count"] / runs
            avg_turns = r["total_turns"] / runs
            score = success_rate - 0.02 * avg_turns  # prefer high success, fewer turns
            if score > best_score:
                best, best_score = r, score
        if not best:
            return None
        runs = best["runs"] or 1
        return {
            "resolution_path": best["resolution_path"],
            "success_rate": round(best["success_count"] / runs, 2),
            "avg_turns": round(best["total_turns"] / runs, 1),
            "runs": runs,
        }

    # ---------------------------------------------------- Unified recall
    def recall(self, borrower_id: str) -> dict:
        """Everything the orchestrator should know before talking to this borrower."""
        return {
            "preferences": self.get_preferences(borrower_id),
            "open_commitments": self.get_commitments(borrower_id, only_open=True),
            "faqs": self.get_faqs(borrower_id),
            "past_conversations": self.past_conversations(borrower_id),
            "previously_asked_slots": sorted(self.get_asked_slots(borrower_id)),
        }


_singleton: MemoryManager | None = None


def get_memory() -> MemoryManager:
    global _singleton
    if _singleton is None:
        _singleton = MemoryManager()
    return _singleton
