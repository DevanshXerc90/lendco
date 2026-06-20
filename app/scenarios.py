"""Canned demo scenarios + borrower selection, for reproducible demos/recordings."""
from __future__ import annotations

import json
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402


def _load(name):
    return json.loads((settings.DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _payments_by_borrower():
    byb = defaultdict(list)
    for p in _load("payments"):
        byb[p["borrower_id"]].append(p)
    return byb


def pick_borrower(scenario: str) -> str:
    """Choose a borrower whose data best demonstrates the scenario."""
    byb = _payments_by_borrower()
    if scenario in ("payment_failure", "waiver", "penalty"):
        # latest failure is system-caused -> waiver eligible / clear failure story
        for b, ps in byb.items():
            fails = [p for p in ps if p["status"] != "success"]
            if fails and fails[-1]["root_cause"] == "system":
                return b
    if scenario in ("ptp_memory", "promise_to_pay"):
        # a delinquent borrower (so a promise-to-pay is natural)
        borrowers = _load("borrowers")
        for b in borrowers:
            if b["dpd_days"] > 0:
                return b["borrower_id"]
    if scenario in ("foreclosure", "interest", "remaining", "account"):
        borrowers = _load("borrowers")
        for b in borrowers:
            if b["months_elapsed"] >= 8:
                return b["borrower_id"]
    return "BRW00001"


SCRIPTS: dict[str, list[str]] = {
    "remaining": ["How many EMIs are remaining on my loan?"],
    "interest": ["How much interest have I paid so far?"],
    "penalty": ["Why was a penalty charged on my account?"],
    "payment_failure": ["My payment failed even though I had enough balance."],
    "waiver": [
        "My payment failed even though I had enough balance.",
        "Can my penalty be waived because the failure was caused by the bank?",
    ],
    "foreclosure": ["I want to foreclose and close my loan. What's my payoff amount?"],
    "settlement": ["I'm in financial difficulty and want a one-time settlement.",
                   "I lost my job three months ago and can't keep up with the EMIs."],
    "account": ["Can you give me a summary of my loan account?"],
}

# Two-call memory scenario (scenario 6): (call1 utterances, call2 utterances)
PTP_MEMORY = {
    "call1": [
        "My salary is delayed this month, I will make the payment next Friday.",
        "Friday the 13th of June.",
        "salary delay",
    ],
    "call2": [
        "No, not yet, but I will pay it this week.",
    ],
}
