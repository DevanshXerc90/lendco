"""Memory Agent — owns recall, capture, and continuous learning.

Capture: commitments, Q&A, FAQs, preferences and sentiment during a session.
Learning: at session end, records which resolution path was used for the intent
and whether it succeeded, so `best_resolution_path` improves over time.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


class MemoryAgent:
    def __init__(self, memory):
        self.memory = memory

    def recall(self, borrower_id: str) -> dict:
        return self.memory.recall(borrower_id)

    def best_path(self, intent: str) -> dict | None:
        return self.memory.best_resolution_path(intent)

    # capture -----------------------------------------------------------------
    def capture_qa(self, session_id, borrower_id, slot, question, answer):
        self.memory.record_qa(session_id, borrower_id, slot, question, answer)

    def capture_faq(self, borrower_id, intent, question):
        self.memory.record_faq(borrower_id, intent, question)

    def capture_commitment(self, borrower_id, date, amount, reason):
        return self.memory.record_commitment(borrower_id, date, amount, reason)

    def set_preference(self, borrower_id, key, value):
        self.memory.set_preference(borrower_id, key, value)

    def resolve_open_commitments(self, borrower_id, kept: bool):
        for c in self.memory.get_commitments(borrower_id, only_open=True):
            self.memory.resolve_commitment(c["id"], kept)

    # learning ----------------------------------------------------------------
    def learn(self, intent: str, tool_trace: list, success: bool, turns: int):
        path = "->".join(dict.fromkeys(t["tool"] for t in tool_trace)) or "conversational"
        self.memory.record_resolution(intent, path, success, turns)
        return path

    def save_session(self, session: dict):
        self.memory.save_conversation(session)
