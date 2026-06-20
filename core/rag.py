"""Retrieval-Augmented Generation over the knowledge base.

The KB documents are chunked and indexed. Retrieval is pluggable:
  - "tfidf"                : pure-numpy TF-IDF + cosine (zero heavy deps, default)
  - "sentence-transformers": neural embeddings if the package is installed

`retrieve()` returns scored chunks with their source doc, which the agent cites.
A grounded answer must reference retrieved chunks; this is how hallucination is
minimised and decisions are explained from policy.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str


def _chunk_document(doc_id: str, content: str, max_words: int = 90) -> list[Chunk]:
    title = content.splitlines()[0].lstrip("# ").strip() if content else doc_id
    # split on blank lines (paragraphs), then pack into ~max_words chunks
    paras = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    chunks, buf, count = [], [], 0
    for p in paras:
        w = len(p.split())
        if count + w > max_words and buf:
            chunks.append(Chunk(doc_id, title, "\n".join(buf)))
            buf, count = [], 0
        buf.append(p)
        count += w
    if buf:
        chunks.append(Chunk(doc_id, title, "\n".join(buf)))
    return chunks


# --------------------------------------------------------------- TF-IDF backend
class _TfidfIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        docs_tokens = [_tokenize(c.text + " " + c.title) for c in chunks]
        self.vocab = {t: i for i, t in enumerate(sorted({t for toks in docs_tokens for t in toks}))}
        n_docs = len(chunks)
        df = np.zeros(len(self.vocab))
        for toks in docs_tokens:
            for t in set(toks):
                df[self.vocab[t]] += 1
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        self.matrix = np.zeros((n_docs, len(self.vocab)))
        for i, toks in enumerate(docs_tokens):
            self.matrix[i] = self._vectorize(toks)

    def _vectorize(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for t in tokens:
            if t in self.vocab:
                vec[self.vocab[t]] += 1.0
        if vec.sum():
            vec = vec / vec.sum()  # term frequency
        vec = vec * self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def search(self, query: str, k: int) -> list[tuple[float, Chunk]]:
        qv = self._vectorize(_tokenize(query))
        scores = self.matrix @ qv
        order = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.chunks[i]) for i in order if scores[i] > 0]


# ---------------------------------------------- sentence-transformers (optional)
class _NeuralIndex:
    def __init__(self, chunks: list[Chunk]):
        from sentence_transformers import SentenceTransformer  # lazy
        self.chunks = chunks
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embs = self.model.encode([c.text for c in chunks], normalize_embeddings=True)

    def search(self, query: str, k: int) -> list[tuple[float, Chunk]]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embs @ q
        order = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.chunks[i]) for i in order]


@dataclass
class Retrieved:
    doc_id: str
    title: str
    text: str
    score: float


class RAG:
    def __init__(self, backend: str | None = None):
        self.backend_name = backend or settings.EMBEDDINGS_BACKEND
        self.chunks = self._load_chunks()
        if self.backend_name == "sentence-transformers":
            try:
                self.index = _NeuralIndex(self.chunks)
            except Exception:
                self.backend_name = "tfidf"
                self.index = _TfidfIndex(self.chunks)
        else:
            self.index = _TfidfIndex(self.chunks)

    def _load_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for p in sorted(settings.KB_DIR.glob("*.md")):
            chunks.extend(_chunk_document(p.stem, p.read_text(encoding="utf-8")))
        return chunks

    def retrieve(self, query: str, k: int = 3) -> list[Retrieved]:
        hits = self.index.search(query, k)
        return [Retrieved(c.doc_id, c.title, c.text, round(s, 4)) for s, c in hits]

    def context_block(self, query: str, k: int = 3) -> str:
        """A citation-ready block to inject into the LLM prompt."""
        hits = self.retrieve(query, k)
        if not hits:
            return "(no relevant policy found)"
        return "\n\n".join(f"[{h.doc_id} — {h.title}]\n{h.text}" for h in hits)


_rag_singleton: RAG | None = None


def get_rag() -> RAG:
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = RAG()
    return _rag_singleton
