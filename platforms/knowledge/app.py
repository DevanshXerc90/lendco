"""Knowledge platform (Notion/Confluence-like) — policy & FAQ documents.

The RAG layer can ingest either from this API or directly from the KB files;
this service makes the documents independently addressable like a real KB SaaS.

Run:  uvicorn platforms.knowledge.app:app --port 8104
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings  # noqa: E402

app = FastAPI(title="Knowledge Platform", version="1.0")


def _docs() -> list[dict]:
    out = []
    for p in sorted(settings.KB_DIR.glob("*.md")):
        body = p.read_text(encoding="utf-8")
        title = body.splitlines()[0].lstrip("# ").strip() if body else p.stem
        out.append({"doc_id": p.stem, "title": title, "filename": p.name, "content": body})
    return out


@app.get("/health")
def health():
    return {"status": "ok", "service": "knowledge", "documents": len(_docs())}


@app.get("/documents")
def list_documents():
    return [{"doc_id": d["doc_id"], "title": d["title"], "filename": d["filename"]} for d in _docs()]


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    for d in _docs():
        if d["doc_id"] == doc_id:
            return d
    raise HTTPException(404, "document not found")


@app.get("/search")
def search(q: str, k: int = 3):
    """Naive keyword search — the real retrieval is done by the RAG layer.
    This endpoint exists so the KB is usable as a standalone SaaS."""
    q_terms = {t.lower() for t in q.split() if len(t) > 2}
    scored = []
    for d in _docs():
        text = d["content"].lower()
        score = sum(text.count(t) for t in q_terms)
        if score:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"doc_id": d["doc_id"], "title": d["title"], "score": s} for s, d in scored[:k]]
