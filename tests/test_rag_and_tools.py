"""Phase 3 tests: tool calling, citation validation and model failure handling.

Covers the rules the agent layer must obey:
  * only registered tools run, with validated arguments;
  * side effects happen at most once per turn;
  * a citation is only trusted after checking the real KB document;
  * empty retrieval and a failing model degrade to a safe answer;
  * long-term memory never stores raw PAN / phone / email.
"""
from __future__ import annotations

import tempfile

import pytest

import core.rag as rag
from agents.orchestrator import Orchestrator
from agents.voice_agent import VoiceAgent
from core import clients
from core.diagnosis import DiagnosisResult
from core.memory import MemoryManager
from core.rag import POLICY_UNVERIFIED_TEXT, get_rag
from core.tools import ToolContext, execute


BORROWER_ID = "BRW00001"


@pytest.fixture
def ctx() -> ToolContext:
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    return ToolContext(borrower_id=BORROWER_ID, context={}, memory=mem, session_id="S-TEST")


def real_doc_id() -> str:
    ids = sorted(rag.kb_doc_ids())
    assert ids, "knowledge base must contain documents"
    return ids[0]


# ------------------------------------------------------------ tool registration
def test_unknown_tool_is_refused(ctx):
    res = execute("run_shell_command", {"cmd": "rm -rf /"}, ctx)
    assert res["error"].startswith("unknown tool")
    assert ctx.trace == []


INVALID_ARGS = [
    ("get_payment_history", {"status": "everything"}, "status"),
    ("search_knowledge_base", {"query": "   "}, "query"),
    ("search_knowledge_base", {"query": "penalty", "k": 99}, "k"),
    ("create_payment_link", {"amount": -500}, "amount"),
    ("create_payment_link", {"amount": 0}, "amount"),
    ("create_payment_link", {"amount": "500"}, "amount"),
    ("record_commitment", {"date": ""}, "date"),
    ("create_support_ticket", {"category": "Drop Table", "subject": "hi"}, "category"),
    ("create_support_ticket", {"category": "general", "subject": "hi", "priority": "asap"}, "priority"),
    ("escalate_to_human", {"reason": "ok", "priority": "now"}, "priority"),
    ("update_crm", {"field": "loan_amount", "value": "1"}, "field"),
    ("update_crm", {"field": "dpd_days", "value": "0"}, "field"),
    ("trigger_workflow", {"name": "delete_all_records"}, "name"),
]


@pytest.mark.parametrize("tool,args,needle", INVALID_ARGS)
def test_invalid_tool_arguments_are_refused(ctx, tool, args, needle, monkeypatch):
    called = []
    monkeypatch.setattr(clients.payments, "create_payment_link",
                        lambda *a, **k: called.append(a) or {"link_id": "PL1"})
    monkeypatch.setattr(clients.workflow, "trigger",
                        lambda *a, **k: called.append(a) or {"run_id": "WF1"})

    res = execute(tool, args, ctx)

    assert "error" in res and needle in res["error"]
    assert called == []          # nothing reached a platform service
    assert ctx.trace == []       # and nothing was recorded as executed


def test_valid_arguments_run_the_tool(ctx, monkeypatch):
    monkeypatch.setattr(clients.payments, "create_payment_link",
                        lambda *a, **k: {"link_id": "PL1", "url": "https://pay/x"})
    res = execute("create_payment_link", {"amount": 5000, "purpose": "EMI"}, ctx)
    assert res["link_id"] == "PL1"
    assert [t["tool"] for t in ctx.trace] == ["create_payment_link"]


# ----------------------------------------------------------- duplicate actions
def test_side_effect_runs_only_once_per_turn(ctx, monkeypatch):
    calls = []
    monkeypatch.setattr(clients.support, "create_ticket",
                        lambda *a, **k: calls.append(k) or {"ticket_id": f"TKTN{len(calls)}"})

    first = execute("create_support_ticket",
                    {"category": "general", "subject": "hi", "description": "d"}, ctx)
    second = execute("create_support_ticket",
                     {"category": "general", "subject": "hi", "description": "d"}, ctx)

    assert first["ticket_id"] == "TKTN1"
    assert second.get("duplicate") is True
    assert len(calls) == 1  # one ticket, not two


def test_read_tools_are_not_deduplicated(ctx):
    a = execute("get_loan_analytics", {}, ctx)
    b = execute("get_loan_analytics", {}, ctx)
    assert "error" not in a and "error" not in b
    assert len(ctx.trace) == 2


# ------------------------------------------------------------------- citations
def test_citation_of_missing_document_is_not_grounded(ctx):
    grounded, retrieved = get_rag().verify_citations("As per 99_fake_policy you pay nothing.", set())
    assert grounded is False
    assert retrieved == set()          # nothing was added: the document does not exist


def test_citation_of_real_document_is_verified_and_recorded(ctx):
    doc_id = real_doc_id()
    grounded, retrieved = get_rag().verify_citations(f"Policy {doc_id} says so.", set())
    assert grounded is True
    assert doc_id in retrieved          # verified documents are remembered


def test_text_without_citation_is_not_grounded(ctx):
    grounded, _ = get_rag().verify_citations("A penalty always applies.", set())
    assert grounded is False


def test_knowledge_base_search_records_retrieved_documents(ctx):
    execute("search_knowledge_base", {"query": "late payment penalty grace period", "k": 3}, ctx)
    assert ctx.retrieved_docs, "retrieval must record the documents it returned"


def test_empty_retrieval_returns_an_explicit_note(ctx, monkeypatch):
    monkeypatch.setattr(rag.RAG, "retrieve", lambda self, query, k=3: [])
    res = execute("search_knowledge_base", {"query": "zzz unrelated"}, ctx)
    assert res["results"] == []
    assert "no matching policy document" in res["note"]
    assert ctx.retrieved_docs == set()


# ------------------------------------------------------------ model/agent paths
def _diagnosis(intent: str = "penalty_inquiry") -> DiagnosisResult:
    return DiagnosisResult(intent=intent, confidence=0.9, known={}, unknown=[],
                           questions=[], suggested_tools=["search_knowledge_base"])


class _Ctx:
    """Minimal context accepted by the rule-based handlers."""
    def __init__(self):
        self.profile = {"name": "Test", "loan_id": "LN100", "loan_type": "personal",
                        "loan_amount": 100000, "interest_rate": 12}
        self.analytics = {"emi_amount": 5000, "installments_paid": 6, "tenure_months": 24,
                          "outstanding_principal": 80000, "next_due_date": "2026-11-01",
                          "overdue_amount": 0, "total_penalty_charged": 500}


class _FakeLLM:
    """Model that returns a fabricated citation every time."""
    available = True
    name = "fake-model"

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def run_agent(self, system, messages, tools, tool_executor, **kwargs):
        self.calls += 1
        from core.llm import AgentResult
        return AgentResult(text=self.text, tool_trace=[], iterations=1)


def test_fabricated_citation_falls_back_to_safe_answer(monkeypatch):
    monkeypatch.setattr(clients.support, "create_ticket",
                        lambda *a, **k: {"ticket_id": "TKTN1"})
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    agent = VoiceAgent(llm=_FakeLLM("As per 99_fake_policy you owe nothing."), memory=mem)
    tool_ctx = ToolContext(borrower_id=BORROWER_ID, context={}, memory=mem, session_id="S-TEST")

    res = agent.respond_llm({}, {}, _diagnosis(), [{"role": "borrower", "text": "why the penalty?"}], tool_ctx)

    assert res["text"] == POLICY_UNVERIFIED_TEXT   # fabricated source never shown
    assert res["action"] == "ticket_created"
    assert any(t["tool"] == "create_support_ticket" for t in tool_ctx.trace)


def test_empty_retrieval_gives_safe_answer_not_a_policy_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(rag.RAG, "retrieve", lambda self, query, k=3: [])
    monkeypatch.setattr(clients.support, "create_ticket",
                        lambda *a, **k: {"ticket_id": "TKTN1"})
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    agent = VoiceAgent(llm=None, memory=mem)
    tool_ctx = ToolContext(borrower_id=BORROWER_ID, context={}, memory=mem, session_id="S-TEST")

    res = agent.respond_rule_based(_Ctx().__dict__, {}, _diagnosis(), {}, tool_ctx)

    assert res["text"] == POLICY_UNVERIFIED_TEXT
    assert res["action"] == "ticket_created"


class _ExplodingLLM:
    available = True
    name = "exploding-model"

    def run_agent(self, *args, **kwargs):
        raise RuntimeError("model endpoint unavailable")


def test_model_failure_falls_back_to_rule_based():
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    orc = Orchestrator(memory=mem, llm=_ExplodingLLM())
    start = orc.start(borrower_id=BORROWER_ID)
    assert start["found"]

    result = orc.handle("Why was a penalty charged on my account?")

    assert result["text"]                      # a real answer, not an exception
    assert "penalt" in result["text"].lower()
    orc.end()


def test_untrusted_retrieved_text_is_not_treated_as_instructions():
    # Retrieved text may contain instruction-like content; the prompt must say
    # it is data. This keeps the guardrail explicit and reviewable.
    from agents.voice_agent import PERSONA
    assert "DATA, not" in PERSONA


# ---------------------------------------------------------------- memory hygiene
def test_memory_scrubs_identifiers():
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    mem.record_faq(BORROWER_ID, "general_faq",
                   "My PAN is ABCDE1234F, call me on 9876543210 or a@b.com")
    saved = mem.get_faqs(BORROWER_ID)[0]["question"]
    assert "ABCDE1234F" not in saved
    assert "9876543210" not in saved
    assert "a@b.com" not in saved
    assert "ABCDE****F" in saved            # still recognisable for the agent


def test_memory_scrubs_commitment_reason():
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    mem.record_commitment(BORROWER_ID, "2026-11-05", 5000, "waiting for salary to 9876543210")
    reason = mem.get_commitments(BORROWER_ID)[0]["reason"]
    assert "9876543210" not in reason