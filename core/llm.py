"""Deterministic reasoning configuration.

The application uses its local, tool-backed planner so the quickstart works
without external credentials or network services.
"""
from __future__ import annotations


class NullLLM:
    available = False
    name = "rule-based-planner"

    def complete(self, system: str, user: str) -> str:  # pragma: no cover
        return ""


_singleton = None


def get_llm() -> NullLLM:
    global _singleton
    if _singleton is None:
        _singleton = NullLLM()
    return _singleton
