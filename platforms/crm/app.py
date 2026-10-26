"""CRM platform (Zoho/HubSpot-like) — borrower master record + KYC + CRM notes.

Run:  uvicorn platforms.crm.app:app --port 8101
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings  # noqa: E402
from platforms.common import load, append_runtime, read_runtime, require_platform_key  # noqa: E402

app = FastAPI(title="CRM Platform", version="1.0")
_BORROWERS = {b["borrower_id"]: b for b in load("borrowers")}
_BY_PHONE = {b["phone"]: b for b in _BORROWERS.values()}


# Only these borrower fields may be updated through the API. Identity and loan
# facts (phone, loan_amount, dpd_days, KYC ...) are read-only and must never be
# editable from a chat tool call.
UPDATABLE_FIELDS = set(settings.CRM_EDITABLE_FIELDS)


class CRMUpdate(BaseModel):
    field: str = Field(min_length=1, max_length=40)
    value: str = Field(default="", max_length=200)
    note: str | None = Field(default=None, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok", "service": "crm", "borrowers": len(_BORROWERS)}


@app.get("/borrowers/{borrower_id}")
def get_borrower(borrower_id: str):
    b = _BORROWERS.get(borrower_id)
    if not b:
        raise HTTPException(404, "borrower not found")
    return b


@app.get("/borrowers")
def find_borrower(phone: str | None = None, loan_id: str | None = None):
    if phone:
        b = _BY_PHONE.get(phone) or _BY_PHONE.get(phone if phone.startswith("+91") else f"+91{phone}")
        if not b:
            raise HTTPException(404, "no borrower with that phone")
        return b
    if loan_id:
        for b in _BORROWERS.values():
            if b["loan_id"] == loan_id:
                return b
        raise HTTPException(404, "no borrower with that loan id")
    return list(_BORROWERS.values())[:50]


@app.get("/borrowers/{borrower_id}/kyc")
def get_kyc(borrower_id: str):
    b = _BORROWERS.get(borrower_id)
    if not b:
        raise HTTPException(404, "borrower not found")
    return {"borrower_id": borrower_id, **b["kyc"]}


@app.patch("/borrowers/{borrower_id}", dependencies=[Depends(require_platform_key)])
def update_borrower(borrower_id: str, upd: CRMUpdate):
    if upd.field not in UPDATABLE_FIELDS:
        raise HTTPException(422, f"Field '{upd.field}' is not editable. Allowed: {sorted(UPDATABLE_FIELDS)}")
    b = _BORROWERS.get(borrower_id)
    if not b:
        raise HTTPException(404, "borrower not found")
    record = {"borrower_id": borrower_id, "field": upd.field, "value": upd.value, "note": upd.note}
    append_runtime("crm_updates", record)
    # reflect simple top-level field updates in the in-memory copy
    if upd.field in b:
        b[upd.field] = upd.value
    return {"updated": True, **record}


@app.get("/borrowers/{borrower_id}/notes")
def get_notes(borrower_id: str):
    return [r for r in read_runtime("crm_updates") if r["borrower_id"] == borrower_id]
