"""Knowledge Agent — owns RAG retrieval + citation-ready policy grounding."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.rag import get_rag  # noqa: E402


class KnowledgeAgent:
    def __init__(self):
        self.rag = get_rag()

    def retrieve(self, query: str, k: int = 3):
        return self.rag.retrieve(query, k=k)

    def grounding_block(self, query: str, k: int = 3) -> str:
        return self.rag.context_block(query, k=k)

    def cite(self, query: str, k: int = 2) -> tuple[str, list[str]]:
        hits = self.rag.retrieve(query, k=k)
        text = " ".join(h.text.replace("\n", " ") for h in hits)
        citations = [f"{h.doc_id} ({h.title})" for h in hits]
        return text, citations
