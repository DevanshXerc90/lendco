"""Knowledge Agent implemented using LangChain Expression Language (LCEL).

Wraps core.rag RAG retrieval and formatting inside an LCEL pipeline, exposing the
exact same public interface as KnowledgeAgent (retrieve, grounding_block, cite).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.rag import get_rag, Retrieved  # noqa: E402

try:
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    RunnableLambda = None
    RunnablePassthrough = None


def _format_grounding_block(hits: list[Retrieved]) -> str:
    if not hits:
        return "(no relevant policy found)"
    return "\n\n".join(f"[{h.doc_id} — {h.title}]\n{h.text}" for h in hits)


def _format_cite(hits: list[Retrieved]) -> tuple[str, list[str]]:
    text = " ".join(h.text.replace("\n", " ") for h in hits)
    citations = [f"{h.doc_id} ({h.title})" for h in hits]
    return text, citations


class KnowledgeAgentLCEL:
    """LCEL-based drop-in replacement for KnowledgeAgent."""

    def __init__(self, rag=None):
        self.rag = rag or get_rag()
        if not _HAS_LANGCHAIN:
            self.retrieve_chain = None
            self.grounding_chain = None
            self.cite_chain = None
            return

        # Runnable wrapper around existing core/rag.py retriever
        def _run_retrieve(inputs: dict[str, Any]) -> list[Retrieved]:
            query = inputs["query"]
            k = inputs.get("k", 3)
            return self.rag.retrieve(query, k=k)

        retriever_runnable = RunnableLambda(_run_retrieve)

        # LCEL pipelines: RunnablePassthrough piped to retriever, then formatter
        self.retrieve_chain = RunnablePassthrough() | retriever_runnable
        self.grounding_chain = (
            RunnablePassthrough()
            | retriever_runnable
            | RunnableLambda(_format_grounding_block)
        )
        self.cite_chain = (
            RunnablePassthrough()
            | retriever_runnable
            | RunnableLambda(_format_cite)
        )

    def retrieve(self, query: str, k: int = 3) -> list[Retrieved]:
        if self.retrieve_chain is not None:
            return self.retrieve_chain.invoke({"query": query, "k": k})
        return self.rag.retrieve(query, k=k)

    def grounding_block(self, query: str, k: int = 3) -> str:
        if self.grounding_chain is not None:
            return self.grounding_chain.invoke({"query": query, "k": k})
        return self.rag.context_block(query, k=k)

    def cite(self, query: str, k: int = 2) -> tuple[str, list[str]]:
        if self.cite_chain is not None:
            return self.cite_chain.invoke({"query": query, "k": k})
        hits = self.rag.retrieve(query, k=k)
        text = " ".join(h.text.replace("\n", " ") for h in hits)
        citations = [f"{h.doc_id} ({h.title})" for h in hits]
        return text, citations


KnowledgeAgent = KnowledgeAgentLCEL
