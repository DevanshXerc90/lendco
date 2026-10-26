"""Authentication helpers for the public API.

Two credentials, kept deliberately small (standard library only):

  * **Borrower token** — issued by `POST /api/auth/login` once the caller proves
    they know the borrower's registered phone number. The token is an
    HMAC-signed `borrower_id:expiry` pair, so every borrower-scoped endpoint can
    check *which* borrower a request is for and refuse another borrower's
    records (403).

  * **Operator key** — `X-API-Key` header for the demo console endpoints
    (borrower list, scenario runner). Only enforced when `DEMO_API_KEY` is set;
    with it unset the API runs in demo mode and says so at startup.

This is a portfolio-grade safeguard, not a banking login: there is no user
table, no refresh tokens and no rate limiting. See README "Known limitations".
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

from fastapi import Header, HTTPException

from config import settings

TOKEN_TTL_SECONDS = 8 * 60 * 60  # 8 hours

_DEV_SECRET: str | None = None


def _secret() -> str:
    """Signing key: the configured API_SECRET, else a per-process random key.

    The dev fallback exists so nothing secret is ever written into the repo,
    while the demo still works out of the box (tokens reset on restart).
    """
    if settings.API_SECRET:
        return settings.API_SECRET
    global _DEV_SECRET
    if _DEV_SECRET is None:
        _DEV_SECRET = os.urandom(32).hex()
        print("[auth] API_SECRET not set - using a per-process signing key; tokens reset on restart.")
    return _DEV_SECRET


def _sign(payload: str) -> str:
    return hmac.new(_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()


def issue_token(borrower_id: str) -> tuple[str, int]:
    """Return (token, expires_at_unix_seconds) for `borrower_id`."""
    expires = int(time.time()) + TOKEN_TTL_SECONDS
    payload = f"{borrower_id}:{expires}"
    return f"{payload}:{_sign(payload)}", expires


def verify_token(token: str) -> str | None:
    """Return the borrower_id a token is scoped to, or None if it is invalid."""
    parts = token.split(":")
    if len(parts) != 3:
        return None
    borrower_id, expires, signature = parts
    payload = f"{borrower_id}:{expires}"
    if not hmac.compare_digest(signature, _sign(payload)):
        return None
    try:
        if int(expires) < int(time.time()):
            return None
    except ValueError:
        return None
    return borrower_id or None


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.startswith("Bearer "):
        return authorization.removeprefix("Bearer ").strip() or None
    return None


def _operator_ok(x_api_key: str | None) -> bool:
    """Operator access: only enforced when an operator key is configured."""
    if not settings.DEMO_API_KEY:
        return True  # demo mode
    return bool(x_api_key) and hmac.compare_digest(x_api_key, settings.DEMO_API_KEY)


def require_operator(x_api_key: str | None = Header(None)) -> None:
    """Protect demo-console endpoints (borrower list, scenario runner)."""
    if not _operator_ok(x_api_key):
        raise HTTPException(403, "Operator API key required (X-API-Key).")


def check_borrower_access(
    borrower_id: str,
    authorization: str | None,
    x_api_key: str | None = None,
) -> None:
    """Allow the operator, or only the borrower who owns `borrower_id`.

    Order of checks:

    1. A matching operator key (`X-API-Key`) may act for any borrower.
    2. If a bearer token was supplied it is *always* verified and its scope is
       enforced — even in demo mode — so one borrower's session can never read
       another borrower's records (403).
    3. No credentials at all: rejected when an operator key is configured
       (401). With `DEMO_API_KEY` unset the API is in demo mode (open, warned
       at startup) — see README "Known limitations".
    """
    if settings.DEMO_API_KEY and x_api_key and hmac.compare_digest(x_api_key, settings.DEMO_API_KEY):
        return  # operator acting for any borrower
    token = _bearer(authorization)
    if token:
        owner = verify_token(token)
        if owner is None:
            raise HTTPException(401, "Invalid or expired token.")
        if owner != borrower_id:
            raise HTTPException(403, "This session does not allow access to that borrower's records.")
        return
    if settings.DEMO_API_KEY:
        raise HTTPException(401, "Sign in with a borrower token (POST /api/auth/login) or the operator API key (X-API-Key).")
    # demo mode: no key configured, no credential presented — open as documented
