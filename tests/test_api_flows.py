"""End-to-end API tests for the main borrower flows.

These exercise the documented happy paths the demo UI uses — login, borrower
record, analytics, grounded chat and the scenario runner — against the real
data and the deterministic (offline) planner, so they need no API keys and no
running platform services.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import config.settings as settings
import api.index as api_index

BORROWER_ID = "BRW00001"


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(settings, "DEMO_API_KEY", "")  # demo mode, like the quickstart
    return TestClient(api_index.app)


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_login_then_read_own_records(client):
    login = client.post("/api/auth/login",
                        json={"borrower_id": BORROWER_ID, "phone": "+919876543210"})
    if login.status_code != 200:
        # phone numbers come from the generated data — use the real one
        borrower = next(b for b in api_index._load("borrowers")
                        if b["borrower_id"] == BORROWER_ID)
        login = client.post("/api/auth/login",
                            json={"borrower_id": BORROWER_ID, "phone": borrower["phone"]})
    assert login.status_code == 200
    token = login.json()["access_token"]

    record = client.get(f"/api/borrowers/{BORROWER_ID}",
                        headers={"Authorization": f"Bearer {token}"})
    assert record.status_code == 200
    assert "****" in record.json()["kyc"]["pan"]  # PAN stays masked


def test_analytics_returns_derived_loan_facts(client):
    body = client.get(f"/api/analytics/{BORROWER_ID}").json()
    a = body["analytics"]
    assert a["installments_remaining"] == a["tenure_months"] - a["installments_paid"]
    assert a["overdue_amount"] >= 0


def test_chat_returns_a_grounded_answer(client):
    res = client.post("/api/chat", json={
        "borrower_id": BORROWER_ID,
        "message": "How many EMIs are remaining on my loan?",
    })
    assert res.status_code == 200
    body = res.json()
    assert body["intent"] == "remaining_emi"
    assert "EMIs remain" in body["text"]
    assert body["tools_used"] or body["grounding"] is not None


def test_chat_with_a_policy_question_cites_a_real_document(client):
    res = client.post("/api/chat", json={
        "borrower_id": BORROWER_ID,
        "message": "Can my penalty be waived because the bank failed?",
    })
    assert res.status_code == 200
    body = res.json()
    cited = body["grounding"] or []
    assert cited, "a policy answer must carry citations"
    from core.rag import kb_doc_ids
    assert all(c["doc_id"] in kb_doc_ids() for c in cited)


def test_scenario_runner_executes_a_full_script(client):
    res = client.post("/api/scenario", json={"scenario": "penalty"})
    assert res.status_code == 200
    body = res.json()
    assert body["turns"], "scenario should produce turns"
    assert all(t["agent"] for t in body["turns"])


def test_documents_endpoints_serve_the_kb(client):
    docs = client.get("/api/documents").json()
    assert docs and all(d["doc_id"] for d in docs)
    one = client.get(f"/api/documents/{docs[0]['doc_id']}")
    assert one.status_code == 200
    assert one.json()["content"].startswith("#")