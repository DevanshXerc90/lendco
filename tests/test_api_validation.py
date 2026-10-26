"""Phase 2 tests: authentication, authorization and request validation.

Covers the public API (api/index.py) and the internal platform services
(platforms/*): wrong/missing credentials, cross-borrower access, malformed
identifiers, invalid amounts, non-editable CRM fields and unknown workflows.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import config.settings as settings
import api.index as api_index
from api import auth
from core import clients


BORROWER_ID = "BRW00001"


# --------------------------------------------------------------------- helpers
@pytest.fixture
def client() -> TestClient:
    return TestClient(api_index.app)


@pytest.fixture
def operator_key(monkeypatch) -> str:
    """Simulate a configured deployment: DEMO_API_KEY is required."""
    monkeypatch.setattr(settings, "DEMO_API_KEY", "test-operator-key")
    return "test-operator-key"


@pytest.fixture(autouse=True)
def _known_borrower(monkeypatch):
    """Borrower BRW00001 with a fixed phone, independent of generated data."""
    fake = {"borrower_id": BORROWER_ID, "phone": "+919876543210",
            "name": "Test Borrower", "kyc": {"pan": "ABCDE1234F"}}

    def lookup(bid: str) -> dict:
        if bid == BORROWER_ID:
            return fake
        raise HTTPException(404, "not found")

    monkeypatch.setattr(api_index, "_borrower_or_404", lookup)


def login(client: TestClient, phone: str = "+919876543210") -> str:
    r = client.post("/api/auth/login", json={"borrower_id": BORROWER_ID, "phone": phone})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ----------------------------------------------------------------------- login
def test_login_rejects_wrong_phone(client):
    r = client.post("/api/auth/login", json={"borrower_id": BORROWER_ID, "phone": "+911111111111"})
    assert r.status_code == 401


def test_login_rejects_malformed_borrower_id(client):
    r = client.post("/api/auth/login", json={"borrower_id": "../../etc", "phone": "+919876543210"})
    assert r.status_code == 400


def test_login_token_scopes_borrower(client):
    token = login(client)
    r = client.get(f"/api/borrowers/{BORROWER_ID}", headers=bearer(token))
    assert r.status_code == 200


def test_cross_borrower_token_forbidden(client):
    token = login(client)
    # A valid token, but pointed at a different borrower.
    r = client.get("/api/borrowers/BRW00002", headers=bearer(token))
    assert r.status_code == 403


def test_garbage_token_rejected(client):
    r = client.get(f"/api/borrowers/{BORROWER_ID}", headers=bearer("BRW00001:9999999999:notasignature"))
    assert r.status_code == 401


def test_token_signature_cannot_be_forged(client):
    token = login(client)
    forged = token[:-4] + "0000"  # tamper with the signature
    r = client.get(f"/api/borrowers/{BORROWER_ID}", headers=bearer(forged))
    assert r.status_code == 401


# --------------------------------------------------- operator key enforcement
def test_borrower_endpoints_require_credentials_when_key_configured(client, operator_key):
    assert client.get(f"/api/borrowers/{BORROWER_ID}").status_code == 401
    assert client.get(f"/api/payments", params={"borrower_id": BORROWER_ID}).status_code == 401


def test_operator_key_grants_access(client, operator_key):
    r = client.get(f"/api/borrowers/{BORROWER_ID}", headers={"X-API-Key": operator_key})
    assert r.status_code == 200


def test_wrong_operator_key_rejected(client, operator_key):
    r = client.get(f"/api/borrowers/{BORROWER_ID}", headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_borrower_list_requires_operator_key(client, operator_key):
    assert client.get("/api/borrowers").status_code == 403
    r = client.get("/api/borrowers", headers={"X-API-Key": operator_key})
    assert r.status_code == 200
    # full phone numbers must never appear in the list response
    assert all("*" in b["phone"] for b in r.json())


def test_scenario_requires_operator_key(client, operator_key):
    assert client.post("/api/scenario", json={"scenario": "remaining"}).status_code == 403
    r = client.post("/api/scenario", json={"scenario": "remaining"},
                    headers={"X-API-Key": operator_key})
    assert r.status_code == 200


def test_operator_key_not_configured_means_open_demo_mode(client, monkeypatch):
    # Documented demo mode: no key configured -> endpoints are open.
    monkeypatch.setattr(settings, "DEMO_API_KEY", "")
    assert client.get("/api/borrowers").status_code == 200


# ------------------------------------------------------------------ validation
def test_invalid_borrower_id_is_400(client):
    assert client.get("/api/borrowers/not-an-id").status_code == 400
    assert client.get("/api/payments", params={"borrower_id": "x"}).status_code == 400


def test_unknown_borrower_is_404(client, monkeypatch):
    monkeypatch.setattr(settings, "DEMO_API_KEY", "")  # demo mode: request reaches the handler
    assert client.get("/api/borrowers/BRW99999").status_code == 404


def test_document_id_path_traversal_blocked(client):
    assert client.get("/api/documents/..").status_code in (400, 404)
    r = client.get("/api/documents/..%2F..%2Fconfig%2Fsettings")
    assert r.status_code in (400, 404)
    # a non-existent but well-formed id is a plain 404
    assert client.get("/api/documents/no_such_doc").status_code == 404


def test_chat_rejects_unknown_scenario(client):
    r = client.post("/api/scenario", json={"scenario": "definitely_not_a_scenario"})
    assert r.status_code == 400


def test_chat_rejects_oversized_message(client):
    r = client.post("/api/chat", json={"borrower_id": BORROWER_ID, "message": "x" * 5000})
    assert r.status_code == 422


def test_chat_rejects_malformed_borrower_id(client):
    r = client.post("/api/chat", json={"borrower_id": "nope", "message": "hi"})
    assert r.status_code == 422


# ----------------------------------------------------------- platform services
def test_payment_link_rejects_invalid_amounts():
    c = TestClient(__import__("platforms.payments.app", fromlist=["app"]).app)
    base = {"borrower_id": BORROWER_ID, "purpose": "EMI"}
    for amount in (0, -100, 10_000_001):
        r = c.post("/payment-links", json={**base, "amount": amount})
        assert r.status_code == 422, f"amount {amount} should be rejected"
    r = c.post("/payment-links", json={**base, "amount": 5000})
    assert r.status_code == 200


def test_payment_link_rejects_malformed_borrower_id():
    c = TestClient(__import__("platforms.payments.app", fromlist=["app"]).app)
    r = c.post("/payment-links", json={"borrower_id": "DROP TABLE", "amount": 100})
    assert r.status_code == 422


def test_crm_rejects_non_editable_field():
    c = TestClient(__import__("platforms.crm.app", fromlist=["app"]).app)
    r = c.patch(f"/borrowers/{BORROWER_ID}", json={"field": "loan_amount", "value": "1"})
    assert r.status_code == 422
    r = c.patch(f"/borrowers/{BORROWER_ID}", json={"field": "dpd_days", "value": "0"})
    assert r.status_code == 422
    r = c.patch(f"/borrowers/{BORROWER_ID}", json={"field": "preferred_language", "value": "hindi"})
    assert r.status_code == 200
    r = c.patch("/borrowers/BRW99999", json={"field": "preferred_language", "value": "hindi"})
    assert r.status_code == 404


def test_support_rejects_bad_priority_and_id():
    c = TestClient(__import__("platforms.support.app", fromlist=["app"]).app)
    r = c.post("/tickets", json={"borrower_id": BORROWER_ID, "category": "general",
                                 "subject": "hi", "priority": "asap"})
    assert r.status_code == 422
    r = c.post("/tickets", json={"borrower_id": "nope", "category": "general", "subject": "hi"})
    assert r.status_code == 422
    r = c.post("/tickets", json={"borrower_id": BORROWER_ID, "category": "general",
                                 "subject": "hi", "priority": "high"})
    assert r.status_code == 200


def test_unknown_workflow_is_rejected():
    c = TestClient(__import__("platforms.workflow.app", fromlist=["app"]).app)
    r = c.post(f"/workflows/definitely_unknown/trigger",
               json={"borrower_id": BORROWER_ID, "params": {}})
    assert r.status_code == 404
    r = c.post(f"/workflows/reregister_mandate/trigger",
               json={"borrower_id": BORROWER_ID, "params": {}})
    assert r.status_code == 200


def test_platform_key_required_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "PLATFORM_API_KEY", "internal-secret")
    c = TestClient(__import__("platforms.workflow.app", fromlist=["app"]).app)
    payload = {"borrower_id": BORROWER_ID, "reason": "follow up"}
    assert c.post("/callbacks", json=payload).status_code == 403
    r = c.post("/callbacks", json=payload, headers={"X-Platform-Key": "internal-secret"})
    assert r.status_code == 200
    # the agent client must send the same key
    assert clients._headers() == {"X-Platform-Key": "internal-secret"}
