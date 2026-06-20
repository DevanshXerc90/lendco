"""Typed clients for the five mock platforms.

Each client wraps HTTP calls to one independent service. If a service is
unreachable, the client falls back to reading the generated data directly so the
agent degrades gracefully (a production-readiness concern) instead of crashing.
"""
from __future__ import annotations

import json
from functools import lru_cache

import httpx

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

_TIMEOUT = httpx.Timeout(8.0)


@lru_cache(maxsize=1)
def _local(name: str) -> list[dict]:
    path = settings.DATA_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _get(base: str, path: str, **params):
    r = httpx.get(f"{base}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(base: str, path: str, payload: dict):
    r = httpx.post(f"{base}{path}", json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- CRM
class CRMClient:
    base = settings.PLATFORM_URLS["crm"]

    def get_borrower(self, borrower_id: str) -> dict | None:
        try:
            return _get(self.base, f"/borrowers/{borrower_id}")
        except Exception:
            return next((b for b in _local("borrowers") if b["borrower_id"] == borrower_id), None)

    def find_by_phone(self, phone: str) -> dict | None:
        try:
            return _get(self.base, "/borrowers", phone=phone)
        except Exception:
            p = phone if phone.startswith("+91") else f"+91{phone}"
            return next((b for b in _local("borrowers") if b["phone"] == p), None)

    def get_kyc(self, borrower_id: str) -> dict | None:
        try:
            return _get(self.base, f"/borrowers/{borrower_id}/kyc")
        except Exception:
            b = self.get_borrower(borrower_id)
            return {"borrower_id": borrower_id, **b["kyc"]} if b else None

    def update(self, borrower_id: str, field: str, value: str, note: str | None = None) -> dict:
        return _post_patch(self.base, f"/borrowers/{borrower_id}", {"field": field, "value": value, "note": note})


def _post_patch(base: str, path: str, payload: dict):
    r = httpx.patch(f"{base}{path}", json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ----------------------------------------------------------------------- Payments
class PaymentsClient:
    base = settings.PLATFORM_URLS["payments"]

    def history(self, borrower_id: str, status: str | None = None) -> list[dict]:
        try:
            return _get(self.base, "/payments", borrower_id=borrower_id, status=status)
        except Exception:
            items = [p for p in _local("payments") if p["borrower_id"] == borrower_id]
            return [p for p in items if not status or p["status"] == status]

    def gateway_logs(self, borrower_id: str) -> list[dict]:
        try:
            return _get(self.base, "/gateway-logs", borrower_id=borrower_id)
        except Exception:
            return [p for p in _local("payments") if p["borrower_id"] == borrower_id and p["status"] != "success"]

    def create_payment_link(self, borrower_id: str, amount: float, purpose: str = "EMI payment") -> dict:
        return _post(self.base, "/payment-links", {"borrower_id": borrower_id, "amount": amount, "purpose": purpose})


# ------------------------------------------------------------------------ Support
class SupportClient:
    base = settings.PLATFORM_URLS["support"]

    def tickets(self, borrower_id: str | None = None, status: str | None = None) -> list[dict]:
        try:
            return _get(self.base, "/tickets", borrower_id=borrower_id, status=status)
        except Exception:
            items = [t for t in _local("tickets") if not borrower_id or t["borrower_id"] == borrower_id]
            return [t for t in items if not status or t["status"] == status]

    def create_ticket(self, borrower_id: str, category: str, subject: str, description: str,
                      loan_id: str | None = None, priority: str = "medium") -> dict:
        return _post(self.base, "/tickets", {
            "borrower_id": borrower_id, "loan_id": loan_id, "category": category,
            "subject": subject, "description": description, "priority": priority,
        })


# ---------------------------------------------------------------------- Knowledge
class KnowledgeClient:
    base = settings.PLATFORM_URLS["knowledge"]

    def documents(self) -> list[dict]:
        try:
            return _get(self.base, "/documents")
        except Exception:
            return [{"doc_id": p.stem, "filename": p.name} for p in settings.KB_DIR.glob("*.md")]

    def get_document(self, doc_id: str) -> dict | None:
        try:
            return _get(self.base, f"/documents/{doc_id}")
        except Exception:
            p = settings.KB_DIR / f"{doc_id}.md"
            return {"doc_id": doc_id, "content": p.read_text(encoding="utf-8")} if p.exists() else None


# ----------------------------------------------------------------------- Workflow
class WorkflowClient:
    base = settings.PLATFORM_URLS["workflow"]

    def schedule_callback(self, borrower_id: str, reason: str, preferred_window: str = "tomorrow 10:00-12:00") -> dict:
        return _post(self.base, "/callbacks", {"borrower_id": borrower_id, "reason": reason, "preferred_window": preferred_window})

    def escalate(self, borrower_id: str, reason: str, priority: str = "high", context: str | None = None) -> dict:
        return _post(self.base, "/escalations", {"borrower_id": borrower_id, "reason": reason, "priority": priority, "context": context})

    def trigger(self, name: str, borrower_id: str, params: dict | None = None) -> dict:
        return _post(self.base, f"/workflows/{name}/trigger", {"borrower_id": borrower_id, "params": params or {}})


crm = CRMClient()
payments = PaymentsClient()
support = SupportClient()
knowledge = KnowledgeClient()
workflow = WorkflowClient()
