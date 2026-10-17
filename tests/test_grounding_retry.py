"""Unit tests for VoiceAgent.respond_llm grounding retry and safe fallback paths."""
from __future__ import annotations

import tempfile
import pytest

from agents.voice_agent import VoiceAgent
from core.diagnosis import DiagnosisResult
from core.memory import MemoryManager
from core.llm import AgentResult
from core.tools import ToolContext


class MockLLMForGrounding:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.call_count = 0
        self.recorded_messages = []

    def run_agent(self, system, messages, tools, tool_executor, max_iters=6, max_tokens=900, intent=None):
        self.call_count += 1
        self.recorded_messages.append(messages)
        resp_text = self.responses.pop(0) if self.responses else "fallback text"
        return AgentResult(text=resp_text, tool_trace=[], iterations=1)


@pytest.fixture
def agent_setup():
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    context = {
        "profile": {"name": "Test Borrower", "loan_id": "LN100", "loan_type": "Personal", "loan_amount": 100000, "interest_rate": 12.0},
        "analytics": {"emi_amount": 5000, "installments_paid": 6, "tenure_months": 24, "outstanding_principal": 80000, "next_due_date": "2026-11-01", "overdue_amount": 0, "total_penalty_charged": 500},
        "delinquency_status": "current",
        "dpd_days": 0,
    }
    recall = {"commitments": [], "preferences": {}}
    diagnosis = DiagnosisResult(
        intent="penalty_inquiry",
        confidence=0.9,
        known={},
        unknown=[],
        questions=[],
        suggested_tools=["search_knowledge_base", "get_gateway_logs"],
    )

    history = [{"role": "borrower", "text": "Why was a penalty charged on my account?"}]
    return mem, context, recall, diagnosis, history


def test_grounding_path1_retry_succeeds(agent_setup):
    """Path 1: ungrounded response -> force KB search -> retry -> grounded response returned."""
    mem, context, recall, diagnosis, history = agent_setup
    mock_llm = MockLLMForGrounding([
        "A penalty of Rs. 500 was charged to your account.",  # Ungrounded initial response
        "Under our late-payment policy (04_late_payment_policy), a 2% penalty applies after the 3-day grace period."  # Grounded retry
    ])
    voice_agent = VoiceAgent(llm=mock_llm, memory=mem)
    tool_ctx = ToolContext(borrower_id="BRW00001", context=context, memory=mem, session_id="test_session")

    res = voice_agent.respond_llm(context, recall, diagnosis, history, tool_ctx)

    assert mock_llm.call_count == 2
    assert "04_late_payment_policy" in res["text"]
    assert res["action"] is None
    # Check that search_knowledge_base was forced in tool_trace
    tools_called = [t["tool"] for t in tool_ctx.trace]
    assert "search_knowledge_base" in tools_called


def test_grounding_path2_retry_fails_safe_fallback(agent_setup):
    """Path 2: ungrounded response -> force KB search -> retry still ungrounded -> safe fallback ticket created."""
    mem, context, recall, diagnosis, history = agent_setup
    mock_llm = MockLLMForGrounding([
        "A penalty of Rs. 500 was charged to your account.",  # Ungrounded initial response
        "The penalty is standard and cannot be modified."     # Still ungrounded retry
    ])
    voice_agent = VoiceAgent(llm=mock_llm, memory=mem)
    tool_ctx = ToolContext(borrower_id="BRW00001", context=context, memory=mem, session_id="test_session")

    res = voice_agent.respond_llm(context, recall, diagnosis, history, tool_ctx)

    assert mock_llm.call_count == 2
    # Never return uncited policy claim
    assert "The penalty is standard" not in res["text"]
    assert "verified by our team" in res["text"]
    assert "support ticket has been raised" in res["text"]
    assert res["action"] == "ticket_created"

    tools_called = [t["tool"] for t in tool_ctx.trace]
    assert "search_knowledge_base" in tools_called
    assert "create_support_ticket" in tools_called
