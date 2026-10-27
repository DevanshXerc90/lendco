"""Optional hosted intent planner used only when explicitly enabled.

The default application path remains deterministic and offline. This module
uses the Anthropic HTTP API directly so the core requirements remain small.
"""
from __future__ import annotations

import json
import os

import httpx

INTENTS = (
    "remaining_emi", "interest_paid", "penalty_inquiry", "payment_failure",
    "penalty_waiver", "promise_to_pay", "settlement_request", "foreclosure",
    "account_info", "general_faq",
)


class HostedIntentPlanner:
    def __init__(self) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.model = os.getenv("LLM_MODEL", "claude-3-5-haiku-latest")

    @property
    def available(self) -> bool:
        return bool(self.api_key and os.getenv("LENDCO_LLM", "").lower() in {"1", "true", "yes"})

    def classify(self, utterance: str) -> str:
        if not self.available:
            raise RuntimeError("LENDCO_LLM is not enabled with ANTHROPIC_API_KEY")
        prompt = (
            "Classify the borrower utterance into exactly one label: "
            f"{', '.join(INTENTS)}. Return JSON only with an intent field. "
            f"Utterance: {utterance}"
        )
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={"model": self.model, "max_tokens": 40,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        response.raise_for_status()
        text = response.json()["content"][0]["text"]
        intent = json.loads(text)["intent"]
        if intent not in INTENTS:
            raise ValueError(f"LLM returned unsupported intent: {intent!r}")
        return intent
