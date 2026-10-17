"""Tests for Anthropic tool-use loop with a fake Anthropic client.

Covers:
1. tool_use then final text
2. several tool_use blocks in one turn
3. max_iters reached then forced final answer
4. tool result truncation at 4000 chars
"""
from __future__ import annotations

import pytest
from core.llm import AnthropicLLM, _stringify


class FakeBlock:
    def __init__(self, block_type: str, text: str = "", tool_id: str = "", name: str = "", tool_input: dict = None):
        self.type = block_type
        self.text = text
        self.id = tool_id
        self.name = name
        self.input = tool_input or {}

    def model_dump(self):
        if self.type == "text":
            return {"type": "text", "text": self.text}
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


class FakeResponse:
    def __init__(self, content: list[FakeBlock], stop_reason: str = "end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class FakeAnthropicMessages:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse([FakeBlock("text", text="Default fake response")], stop_reason="end_turn")


class FakeAnthropicClient:
    def __init__(self, responses: list[FakeResponse]):
        self.messages = FakeAnthropicMessages(responses)


def _create_fake_llm(responses: list[FakeResponse]) -> AnthropicLLM:
    client = FakeAnthropicClient(responses)
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.client = client
    llm.model = "claude-test"
    llm.name = "claude-test"
    return llm


def test_tool_use_then_final_text():
    """1. Tool use block executed, followed by final text response."""
    resp1 = FakeResponse(
        content=[FakeBlock("tool_use", tool_id="call_1", name="search_knowledge_base", tool_input={"query": "late fee"})],
        stop_reason="tool_use",
    )
    resp2 = FakeResponse(
        content=[FakeBlock("text", text="Based on policy 04_late_payment_policy, late fee is 2%.")],
        stop_reason="end_turn",
    )
    llm = _create_fake_llm([resp1, resp2])

    tools_called = []
    def executor(name, args):
        tools_called.append((name, args))
        return {"results": [{"doc_id": "04_late_payment_policy", "text": "late fee is 2%"}]}

    result = llm.run_agent(
        system="You are a loan assistant.",
        messages=[{"role": "user", "content": "What is the late fee?"}],
        tools=[{"name": "search_knowledge_base"}],
        tool_executor=executor,
    )

    assert result.text == "Based on policy 04_late_payment_policy, late fee is 2%."
    assert result.iterations == 2
    assert len(tools_called) == 1
    assert tools_called[0][0] == "search_knowledge_base"


def test_several_tool_use_blocks_in_one_turn():
    """2. Multiple tool use blocks in a single assistant turn."""
    resp1 = FakeResponse(
        content=[
            FakeBlock("tool_use", tool_id="call_1", name="get_loan_analytics", tool_input={}),
            FakeBlock("tool_use", tool_id="call_2", name="search_knowledge_base", tool_input={"query": "repayment"}),
        ],
        stop_reason="tool_use",
    )
    resp2 = FakeResponse(
        content=[FakeBlock("text", text="Your remaining EMIs are 5 per policy 01_repayment_terms.")],
        stop_reason="end_turn",
    )
    llm = _create_fake_llm([resp1, resp2])

    tools_called = []
    def executor(name, args):
        tools_called.append(name)
        if name == "get_loan_analytics":
            return {"remaining_emis": 5}
        return {"results": [{"doc_id": "01_repayment_terms"}]}

    result = llm.run_agent(
        system="System prompt",
        messages=[{"role": "user", "content": "Tell me my loan details"}],
        tools=[{"name": "get_loan_analytics"}, {"name": "search_knowledge_base"}],
        tool_executor=executor,
    )

    assert result.text == "Your remaining EMIs are 5 per policy 01_repayment_terms."
    assert tools_called == ["get_loan_analytics", "search_knowledge_base"]
    assert len(result.tool_trace) == 2


def test_max_iters_reached_forced_final_answer():
    """3. Max iterations reached, forcing a final summary answer."""
    max_iters = 3
    resps = [
        FakeResponse(
            [FakeBlock("tool_use", tool_id=f"call_{i}", name="search_knowledge_base", tool_input={})],
            stop_reason="tool_use",
        )
        for i in range(max_iters)
    ]
    final_resp = FakeResponse(
        [FakeBlock("text", text="Forced final answer after reaching iteration limit.")],
        stop_reason="end_turn",
    )
    resps.append(final_resp)

    llm = _create_fake_llm(resps)

    result = llm.run_agent(
        system="System prompt",
        messages=[{"role": "user", "content": "Help me with my loan"}],
        tools=[{"name": "search_knowledge_base"}],
        tool_executor=lambda n, a: {"status": "ok"},
        max_iters=max_iters,
    )

    assert result.text == "Forced final answer after reaching iteration limit."
    assert result.iterations == max_iters
    assert len(llm.client.messages.calls) == max_iters + 1
    last_call = llm.client.messages.calls[-1]
    assert last_call["messages"][-1]["content"] == "Please give your final response to the borrower now."


def test_tool_result_truncation_at_4000_chars():
    """4. Tool results are truncated at exactly 4000 characters."""
    large_payload = {"huge_log": "x" * 6000}
    truncated = _stringify(large_payload)
    assert len(truncated) == 4000

    resp1 = FakeResponse(
        content=[FakeBlock("tool_use", tool_id="call_big", name="get_payment_history", tool_input={})],
        stop_reason="tool_use",
    )
    resp2 = FakeResponse(
        content=[FakeBlock("text", text="Processed large log.")],
        stop_reason="end_turn",
    )
    llm = _create_fake_llm([resp1, resp2])

    result = llm.run_agent(
        system="System prompt",
        messages=[{"role": "user", "content": "Show my full payment logs"}],
        tools=[{"name": "get_payment_history"}],
        tool_executor=lambda n, a: large_payload,
    )

    assert result.text == "Processed large log."
    # Check that tool_result sent to API in turn 2 was truncated to 4000 chars
    call2 = llm.client.messages.calls[1]
    tool_result_block = call2["messages"][-1]["content"][0]
    assert tool_result_block["type"] == "tool_result"
    assert len(tool_result_block["content"]) == 4000


def test_evaluator_refuses_without_key(monkeypatch):
    """evaluator with use_llm=True refuses to run when ANTHROPIC_API_KEY is not set."""
    from eval.evaluator import run_all
    from config import settings
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")
    with pytest.raises(SystemExit) as exc:
        run_all(use_llm=True)
    assert exc.value.code == 1

