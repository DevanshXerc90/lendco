"""Pluggable LLM reasoning engine.

If ANTHROPIC_API_KEY is set, `AnthropicLLM` runs a genuine tool-use loop: the
model reasons, decides which tools to call, we execute them and feed results
back, and it produces a grounded final answer. Prompt caching is applied to the
(stable) system prompt + tool schemas to cut latency/cost on multi-turn calls.

If no key is present, `get_llm()` returns a `NullLLM` (available == False) and
the orchestrator falls back to its deterministic planner — so the whole system
runs offline with identical tool behaviour, just template-phrased responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

import re

CITE_RE = re.compile(r"\d{2}_[a-z_]+")
POLICY_INTENTS = {"penalty_inquiry", "penalty_waiver", "payment_failure", "settlement_request",
                  "foreclosure", "interest_paid"}


@dataclass
class AgentResult:
    text: str
    tool_trace: list = field(default_factory=list)
    iterations: int = 0


class NullLLM:
    available = False
    name = "rule-based-planner"

    def complete(self, system: str, user: str) -> str:  # pragma: no cover
        return ""


class AnthropicLLM:
    available = True

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.LLM_MODEL
        self.name = self.model

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def run_agent(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        tool_executor: Callable[[str, dict], dict],
        max_iters: int = 6,
        max_tokens: int = 900,
        intent: str | None = None,
    ) -> AgentResult:
        # cache the stable system prompt + tool definitions
        system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if tools:
            tools = [dict(t) for t in tools]
            tools[-1]["cache_control"] = {"type": "ephemeral"}

        convo = list(messages)
        trace: list = []
        for i in range(max_iters):
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens,
                system=system_blocks, messages=convo, tools=tools,
            )
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                if intent and intent in POLICY_INTENTS and not CITE_RE.search(text or ""):
                    kb_res = tool_executor("search_knowledge_base", {"query": intent.replace("_", " "), "k": 2})
                    trace.append({"tool": "search_knowledge_base", "input": {"query": intent.replace("_", " "), "k": 2}, "result": kb_res})
                    retry_convo = list(convo) + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": f"Your response is missing a required policy citation. Retrieved policy: {_stringify(kb_res)}. Please rewrite and cite the document ID (e.g. 04_late_payment_policy)."},
                    ]
                    retry_resp = self.client.messages.create(
                        model=self.model, max_tokens=max_tokens, system=system_blocks, messages=retry_convo, tools=tools,
                    )
                    text = "".join(b.text for b in retry_resp.content if b.type == "text").strip()
                    if not CITE_RE.search(text or ""):
                        ticket_res = tool_executor("create_support_ticket", {"category": "policy_inquiry", "subject": f"Policy clarification for {intent}"})
                        trace.append({"tool": "create_support_ticket", "input": {"category": "policy_inquiry", "subject": f"Policy clarification for {intent}"}, "result": ticket_res})
                        text = "I want to make sure I give you the exact policy details. Our policy will be verified by our team, and a support ticket has been raised for your account to ensure this is reviewed accurately."
                return AgentResult(text=text, tool_trace=trace, iterations=i + 1)

            # execute every tool_use block in this turn
            convo.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    result = tool_executor(block.name, block.input or {})
                    trace.append({"tool": block.name, "input": block.input, "result": result})
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": _stringify(result),
                    })
            convo.append({"role": "user", "content": tool_results})

        # ran out of iterations — ask for a final summary without tools
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system_blocks,
            messages=convo + [{"role": "user", "content": "Please give your final response to the borrower now."}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if intent and intent in POLICY_INTENTS and not CITE_RE.search(text or ""):
            kb_res = tool_executor("search_knowledge_base", {"query": intent.replace("_", " "), "k": 2})
            trace.append({"tool": "search_knowledge_base", "input": {"query": intent.replace("_", " "), "k": 2}, "result": kb_res})
            retry_convo = list(convo) + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"Your response is missing a required policy citation. Retrieved policy: {_stringify(kb_res)}. Please rewrite and cite the document ID (e.g. 04_late_payment_policy)."},
            ]
            retry_resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system_blocks, messages=retry_convo, tools=tools,
            )
            text = "".join(b.text for b in retry_resp.content if b.type == "text").strip()
            if not CITE_RE.search(text or ""):
                ticket_res = tool_executor("create_support_ticket", {"category": "policy_inquiry", "subject": f"Policy clarification for {intent}"})
                trace.append({"tool": "create_support_ticket", "input": {"category": "policy_inquiry", "subject": f"Policy clarification for {intent}"}, "result": ticket_res})
                text = "I want to make sure I give you the exact policy details. Our policy will be verified by our team, and a support ticket has been raised for your account to ensure this is reviewed accurately."
        return AgentResult(text=text, tool_trace=trace, iterations=max_iters)



def _stringify(result: dict) -> str:
    import json
    try:
        return json.dumps(result, default=str)[:4000]
    except Exception:
        return str(result)[:4000]


_singleton = None


def get_llm():
    global _singleton
    if _singleton is not None:
        return _singleton
    if settings.ANTHROPIC_API_KEY:
        try:
            _singleton = AnthropicLLM()
            return _singleton
        except Exception:
            pass
    _singleton = NullLLM()
    return _singleton
