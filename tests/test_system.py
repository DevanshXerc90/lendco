"""Core test suite. Runs offline (NullLLM) against local-data fallbacks, so it
does NOT require the platform services to be running.

Run:  python -m pytest -q
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from core.context_engine import ContextEngine, _amortize  # noqa: E402
from core.diagnosis import classify_intent, DiagnosisLayer  # noqa: E402
from core.rag import get_rag  # noqa: E402
from core.memory import MemoryManager  # noqa: E402
from core.llm import NullLLM  # noqa: E402
from agents.orchestrator import Orchestrator  # noqa: E402
from app import scenarios  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _data_exists():
    if not (settings.DATA_DIR / "borrowers.json").exists():
        from data.generate import main as gen
        gen()


def _orc():
    return Orchestrator(memory=MemoryManager(db_path=tempfile.mktemp(suffix=".db")), llm=NullLLM())


# --- amortization ----------------------------------------------------------
def test_amortization_split_adds_up():
    prin, intr, out = _amortize(100000, 12.0, 8884.88, 12)  # 1yr loan fully paid
    assert round(prin) == 100000           # principal fully repaid
    assert out == 0
    assert intr > 0


# --- context engine --------------------------------------------------------
def test_context_engine_derives_analytics():
    ctx = ContextEngine().build(borrower_id="BRW00001")
    assert ctx.found
    a = ctx.analytics
    assert a["installments_remaining"] == a["tenure_months"] - a["installments_paid"]
    assert a["outstanding_principal"] >= 0


def test_unknown_borrower():
    ctx = ContextEngine().build(borrower_id="NOPE")
    assert not ctx.found


# --- diagnosis -------------------------------------------------------------
@pytest.mark.parametrize("utt,intent", [
    ("how many emis are remaining", "remaining_emi"),
    ("my payment failed even though i had enough balance", "payment_failure"),
    ("can my penalty be waived because the bank failed", "penalty_waiver"),
    ("my salary is delayed i will pay next friday", "promise_to_pay"),
])
def test_intent_classification(utt, intent):
    assert classify_intent(utt)[0] == intent


def test_diagnosis_no_redundant_questions_when_derivable():
    ctx = ContextEngine().build(borrower_id="BRW00001").to_dict()
    d = DiagnosisLayer().diagnose("how many emis are remaining", ctx)
    assert d.questions == []        # everything is derivable -> never ask


def test_diagnosis_asks_only_unknown():
    ctx = ContextEngine().build(borrower_id="BRW00001").to_dict()
    d = DiagnosisLayer().diagnose("my salary is delayed", ctx, intent="promise_to_pay")
    slots = {q["slot"] for q in d.questions}
    assert "ptp_date" in slots       # ask-only, missing
    assert "ptp_amount" not in slots  # derivable from EMI -> not asked


# --- RAG -------------------------------------------------------------------
def test_rag_retrieves_relevant_policy():
    hits = get_rag().retrieve("can penalty be waived if the bank failed", k=2)
    assert hits and hits[0].doc_id.startswith("05_penalty_waiver")


# --- end-to-end scenarios --------------------------------------------------
def test_remaining_emi_scenario():
    orc = _orc()
    orc.start(borrower_id="BRW00001")
    res = orc.handle("How many EMIs are remaining on my loan?")
    assert res["intent"] == "remaining_emi"
    assert "EMIs remain" in res["text"]


def test_waiver_eligibility_from_root_cause():
    bid = scenarios.pick_borrower("waiver")
    orc = _orc()
    orc.start(borrower_id=bid)
    res = orc.handle("Can my penalty be waived because the bank failed?")
    assert res["intent"] == "penalty_waiver"
    assert res["action"] in ("penalty_reversed", "ticket_created")
    assert "05_penalty_waiver" in res["text"]  # grounded/cited


# --- memory / continuous learning ------------------------------------------
def test_memory_makes_second_call_faster_and_aware():
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    bid = scenarios.pick_borrower("ptp_memory")

    o1 = Orchestrator(memory=mem, llm=NullLLM())
    o1.start(borrower_id=bid)
    for u in scenarios.PTP_MEMORY["call1"]:
        o1.handle(u)
    s1 = o1.end()

    o2 = Orchestrator(memory=mem, llm=NullLLM())
    start2 = o2.start(borrower_id=bid)
    assert start2["memory_aware"] is True            # opener references prior commitment
    for u in scenarios.PTP_MEMORY["call2"]:
        o2.handle(u)
    s2 = o2.end()
    assert s2["turns"] <= s1["turns"]                # repeat call is not slower

    best = mem.best_resolution_path("promise_to_pay")
    assert best is not None                          # agent learned a resolution path
