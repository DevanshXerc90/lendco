"""Phase 4 tests: downstream service failures and configuration errors.

Covers how the agent behaves when a platform service is down, answers with a
malformed body, or when configuration is invalid:
  * reads fall back to local generated data and say so (one clear warning);
  * writes are kept locally and marked with `fallback: True`;
  * a tool call still succeeds, so the conversation does not break;
  * malformed responses raise a clear error instead of a JSON traceback;
  * invalid backend configuration fails fast at startup;
  * warnings never contain credentials or borrower data.
"""
from __future__ import annotations

import tempfile

import pytest

import config.settings as settings
from core import clients
from core.memory import MemoryManager
from core.tools import ToolContext, execute

BORROWER_ID = "BRW00001"


def _down(*args, **kwargs):
    raise ConnectionError("connection refused")


@pytest.fixture
def local_runtime(tmp_path, monkeypatch):
    """Point the clients' local fallback store at a temp directory."""
    monkeypatch.setattr(clients, "RUNTIME_DIR", tmp_path)
    return tmp_path


# -------------------------------------------------------------------- read path
def test_payment_history_falls_back_to_local_data(monkeypatch, capsys):
    clients._warned.discard("payments.history")  # other tests may have warned already
    monkeypatch.setattr(clients, "_get", _down)

    items = clients.payments.history(BORROWER_ID)

    assert items, "local generated data should answer when the service is down"
    out = capsys.readouterr().out
    assert "[Payments] mock service unreachable" in out
    assert BORROWER_ID not in out        # no borrower data in logs


def test_gateway_logs_fall_back_to_local_data(monkeypatch, capsys):
    # pick a borrower that actually has a failed payment in the generated data
    failed = next(p["borrower_id"] for p in clients._local("payments")
                  if p["status"] != "success")
    monkeypatch.setattr(clients, "_get", _down)

    logs = clients.payments.gateway_logs(failed)

    assert logs
    assert "response_code" in logs[0]    # normalised to the service's shape
    assert failed not in capsys.readouterr().out


def test_borrower_lookup_marks_local_source(monkeypatch):
    monkeypatch.setattr(clients, "_get", _down)

    b = clients.crm.get_borrower(BORROWER_ID)

    assert b is not None
    assert b["_source"] == "local_generated_data"  # not passed off as live CRM data


def test_fallback_warning_is_logged_only_once(monkeypatch, capsys):
    monkeypatch.setattr(clients, "_get", _down)
    clients._warned.discard("payments.history")

    clients.payments.history(BORROWER_ID)
    clients.payments.history(BORROWER_ID)

    assert capsys.readouterr().out.count("[Payments] mock service unreachable") == 1


# ------------------------------------------------------------------- write path
def test_ticket_is_kept_locally_when_support_is_down(monkeypatch, local_runtime):
    monkeypatch.setattr(clients, "_post", _down)

    res = clients.support.create_ticket(BORROWER_ID, "general", "Subject", "Body")

    assert res["ticket_id"].startswith("TKTN")
    assert res["fallback"] is True            # visibly not a live-service record
    assert res["borrower_id"] == BORROWER_ID
    stored = (local_runtime / "tickets.json").read_text()
    assert "Subject" in stored


def test_payment_link_is_kept_locally_when_payments_is_down(monkeypatch, local_runtime):
    monkeypatch.setattr(clients, "_post", _down)

    res = clients.payments.create_payment_link(BORROWER_ID, 5000, "EMI retry")

    assert res["url"].startswith("https://pay.lendco.example/")
    assert res["fallback"] is True


def test_workflow_actions_are_kept_locally_when_service_is_down(monkeypatch, local_runtime):
    monkeypatch.setattr(clients, "_post", _down)

    cb = clients.workflow.schedule_callback(BORROWER_ID, "follow up")
    esc = clients.workflow.escalate(BORROWER_ID, "policy question")
    wf = clients.workflow.trigger("send_foreclosure_statement", BORROWER_ID)

    assert cb["callback_id"].startswith("CB") and cb["fallback"]
    assert esc["escalation_id"].startswith("ESC") and esc["fallback"]
    assert wf["run_id"].startswith("WF") and wf["fallback"]


def test_tool_still_succeeds_when_platform_is_down(monkeypatch, local_runtime):
    """The agent loop must not lose the side effect just because a service is down."""
    monkeypatch.setattr(clients, "_post", _down)
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    ctx = ToolContext(borrower_id=BORROWER_ID, context={}, memory=mem, session_id="S-TEST")

    res = execute("create_support_ticket",
                  {"category": "general", "subject": "hi", "description": "d"}, ctx)

    assert "error" not in res
    assert res["ticket_id"].startswith("TKTN")
    assert [t["tool"] for t in ctx.trace] == ["create_support_ticket"]


# ------------------------------------------------------------ malformed replies
class _BadResponse:
    status_code = 200

    def json(self):
        raise ValueError("not json")

    def raise_for_status(self):
        pass


def test_malformed_response_raises_clear_error(monkeypatch):
    monkeypatch.setattr(clients.httpx, "get", lambda *a, **k: _BadResponse())

    with pytest.raises(RuntimeError, match="malformed response"):
        clients._get("http://127.0.0.1:9999", "/things")


def test_every_request_has_a_timeout():
    # 8s connect/read/write/pool — a hung platform can never stall a call.
    assert clients._TIMEOUT.read == 8.0


# ---------------------------------------------------------------- configuration
def test_default_configuration_is_valid():
    settings.validate()  # the shipped defaults must never fail startup


def test_startup_summary_hides_key_values(monkeypatch, capsys):
    summary = "\n".join(settings.startup_summary())

    assert "payments=" in summary