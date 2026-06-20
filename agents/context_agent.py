"""Context Agent — owns unified-context assembly + memory recall."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.context_engine import ContextEngine, UnifiedContext  # noqa: E402


class ContextAgent:
    def __init__(self, memory):
        self.memory = memory
        self.engine = ContextEngine(memory=memory)

    def build(self, borrower_id: str | None = None, phone: str | None = None) -> UnifiedContext:
        return self.engine.build(borrower_id=borrower_id, phone=phone)

    def recall(self, borrower_id: str) -> dict:
        return self.memory.recall(borrower_id)
