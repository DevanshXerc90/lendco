"""Tests for LangChain LCEL Knowledge Agent parity and interface."""
from __future__ import annotations

import pytest
from eval.lcel_parity import check_lcel_parity, FIXED_QUERIES, extract_doc_ids
from agents.knowledge_agent import KnowledgeAgent as OriginalKnowledgeAgent
from agents.knowledge_agent_lcel import KnowledgeAgentLCEL


def test_lcel_parity():
    pytest.importorskip("langchain_core")
    assert check_lcel_parity() is True


def test_lcel_interface_and_order():
    pytest.importorskip("langchain_core")
    orig = OriginalKnowledgeAgent()
    lcel = KnowledgeAgentLCEL()

    for q in FIXED_QUERIES:
        # retrieve
        orig_hits = orig.retrieve(q, k=2)
        lcel_hits = lcel.retrieve(q, k=2)
        assert [h.doc_id for h in orig_hits] == [h.doc_id for h in lcel_hits]

        # grounding_block
        assert orig.grounding_block(q, k=2) == lcel.grounding_block(q, k=2)

        # cite
        orig_text, orig_cites = orig.cite(q, k=2)
        lcel_text, lcel_cites = lcel.cite(q, k=2)
        assert orig_text == lcel_text
        assert extract_doc_ids(orig_cites) == extract_doc_ids(lcel_cites)
