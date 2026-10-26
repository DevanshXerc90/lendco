"""Phase 1 tests: dates/clock and loan calculations.

Covers:
  * the controllable clock (core.dates.set_now / LENDCO_NOW)
  * calendar-month arithmetic (no 30-day months assumed)
  * amortization totals in Decimal
  * overdue amounts, partial payments, duplicate records, due dates
  * seeded data agreeing with the derived analytics
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from config import settings
from core import dates
from core.context_engine import _amortize, _derive_analytics


@pytest.fixture(autouse=True)
def _unfreeze_clock():
    """Never leak a frozen clock into other tests."""
    yield
    dates.set_now(None)


# --------------------------------------------------------------------- clock
def test_set_now_freezes_the_clock():
    dates.set_now(datetime(2026, 6, 9, 10, 0, 0))
    assert dates.today() == date(2026, 6, 9)
    assert dates.now() == datetime(2026, 6, 9, 10, 0, 0, tzinfo=timezone.utc)
    assert dates.iso_now() == "2026-06-09T10:00:00+00:00"


def test_clock_returns_to_real_time_after_unfreeze():
    dates.set_now(datetime(2026, 6, 9, 10, 0, 0))
    dates.set_now(None)
    assert abs(dates.now() - datetime.now(timezone.utc)) < timedelta(seconds=60)


def test_lenco_now_env_override(monkeypatch):
    monkeypatch.setenv("LENDCO_NOW", "2026-01-31T08:30:00")
    assert dates.today() == date(2026, 1, 31)


def test_all_timestamps_are_utc():
    assert dates.now().tzinfo is not None
    assert dates.iso_now().endswith("+00:00")


# ------------------------------------------------------------- month arithmetic
@pytest.mark.parametrize(
    "start,months,expected",
    [
        (date(2026, 1, 31), 1, date(2026, 2, 28)),   # Jan 31 has no Feb 31
        (date(2024, 1, 31), 1, date(2024, 2, 29)),   # leap year
        (date(2026, 3, 31), 1, date(2026, 4, 30)),   # Apr has 30 days
        (date(2026, 1, 31), 13, date(2027, 2, 28)),  # year roll-over
        (date(2026, 12, 15), 2, date(2027, 2, 15)),  # Dec -> Feb across years
        (date(2026, 5, 10), -2, date(2026, 3, 10)),  # backwards
        (date(2026, 5, 10), 0, date(2026, 5, 10)),   # no-op
    ],
)
def test_add_months_is_calendar_accurate(start, months, expected):
    assert dates.add_months(start, months) == expected


def test_monthly_schedule_never_assumes_30_day_months():
    """A 30-day assumption drifts ~5 days every 2 months; calendar months do not."""
    start = date(2026, 1, 31)
    due_6 = dates.add_months(start, 6)
    assert due_6 == date(2026, 7, 31)
    # 6 x 30 days would land on 2026-07-30 — one day off.
    assert start + timedelta(days=6 * 30) != due_6


# ---------------------------------------------------------------- amortization
def test_amortization_totals_balance():
    principal, rate, emi = 100_000, 12.0, 3_321.43
    prin, interest, outstanding = _amortize(principal, rate, emi, 6)
    # Every rupee is accounted for: paid principal + outstanding == principal.
    assert round(prin + outstanding, 2) == round(principal, 2)
    assert interest > 0
    assert outstanding < principal


def test_first_installment_interest_is_principal_times_rate():
    prin, interest, _ = _amortize(120_000, 18.0, 4_000, 1)
    assert round(interest, 2) == round(120_000 * 0.18 / 12, 2)
    assert round(prin, 2) == round(4_000 - interest, 2)


def test_amortization_never_goes_negative_on_long_schedules():
    # 120 installments on a loan that finishes earlier must clamp at zero.
    _, _, outstanding = _amortize(50_000, 24.0, 5_000, 120)
    assert outstanding == 0.0


# ------------------------------------------------------------- analytics fixtures
def _profile(**overrides) -> dict:
    base = {
        "borrower_id": "BRWTEST",
        "tenure_months": 12,
        "emi_amount": 1000.0,
        "loan_amount": 10_000.0,
        "interest_rate": 12.0,
        "loan_start_date": "2026-01-15",
    }
    base.update(overrides)
    return base


def _payment(**overrides) -> dict:
    base = {
        "payment_id": "PAY0000001",
        "borrower_id": "BRWTEST",
        "installment_no": 1,
        "due_date": "2026-02-15",
        "amount_due": 1000.0,
        "amount_paid": 1000.0,
        "status": "success",
        "penalty_charged": 0.0,
    }
    base.update(overrides)
    return base


def test_fully_paid_installments_are_counted():
    payments = [_payment(payment_id=f"PAY{i}", installment_no=i) for i in range(1, 4)]
    a = _derive_analytics(_profile(), payments)
    assert a.installments_paid == 3
    assert a.installments_remaining == 9  # 12 - 3


def test_overdue_only_counts_installments_already_due():
    """An unpaid installment dated in the future is not overdue."""
    dates.set_now(datetime(2026, 3, 1, 10, 0, 0))
    payments = [
        _payment(payment_id="PAY1", installment_no=1, due_date="2026-02-15",
                 status="failed", amount_paid=0.0),
        _payment(payment_id="PAY2", installment_no=2, due_date="2026-04-15",
                 status="failed", amount_paid=0.0),
    ]
    a = _derive_analytics(_profile(), payments)
    assert a.overdue_amount == 1000.0  # only February's EMI is actually late
    assert a.next_due_date == "2026-02-15"  # earliest unpaid installment


def test_partial_payment_reduces_the_overdue_amount():
    dates.set_now(datetime(2026, 3, 1, 10, 0, 0))
    payments = [_payment(status="partial", amount_paid=400.0)]
    a = _derive_analytics(_profile(), payments)
    assert a.overdue_amount == 600.0  # 1000 due - 400 paid
    assert a.installments_paid == 0   # partial is not a full installment


def test_duplicate_records_are_not_counted_twice():
    """Same payment_id twice (a retried upload) must not double the penalty."""
    row = _payment(status="failed", amount_paid=0.0, penalty_charged=50.0)
    a = _derive_analytics(_profile(), [row, dict(row)])
    assert a.overdue_amount == 1000.0  # not 2000
    assert a.total_penalty_charged == 50.0  # not 100


def test_failed_attempt_followed_by_success_counts_as_paid():
    dates.set_now(datetime(2026, 3, 1, 10, 0, 0))
    payments = [
        _payment(payment_id="PAY1", status="failed", amount_paid=0.0,
                 penalty_charged=50.0),
        _payment(payment_id="PAY2", status="success", amount_paid=1000.0),
    ]
    a = _derive_analytics(_profile(), payments)
    assert a.installments_paid == 1
    assert a.overdue_amount == 0.0  # the retry settled the installment
    assert a.total_penalty_charged == 50.0  # the failed attempt's charge stands


def test_next_due_is_the_next_scheduled_installment_on_the_calendar():
    """With everything recorded paid, the next due date is the next installment
    on the calendar-month schedule. A date that has already gone by is reported
    as-is instead of being pushed forward, so a lagging ledger stays visible."""
    dates.set_now(datetime(2026, 6, 20, 10, 0, 0))
    # First EMI (due 15 Feb) paid; nothing else recorded, though 15 Mar, 15 Apr
    # and 15 May have gone by.
    a = _derive_analytics(_profile(), [_payment(payment_id="PAY1")])
    assert a.installments_paid == 1
    # 15 Mar 2026 = start (15 Jan) + 2 calendar months — not start + 60 days.
    assert a.next_due_date == "2026-03-15"


def test_next_due_is_none_when_loan_is_fully_repaid():
    dates.set_now(datetime(2027, 2, 1, 10, 0, 0))
    payments = [_payment(payment_id=f"PAY{i}", installment_no=i) for i in range(1, 13)]
    a = _derive_analytics(_profile(), payments)
    assert a.installments_paid == 12
    assert a.installments_remaining == 0
    assert a.next_due_date is None
    assert a.overdue_amount == 0.0


def test_installs_remaining_never_negative():
    payments = [_payment(payment_id=f"PAY{i}", installment_no=i) for i in range(1, 20)]
    a = _derive_analytics(_profile(), payments)
    assert a.installments_remaining == 0


def test_no_payments_means_nothing_is_overdue():
    dates.set_now(datetime(2026, 6, 20, 10, 0, 0))
    a = _derive_analytics(_profile(), [])
    assert a.installments_paid == 0
    assert a.overdue_amount == 0.0  # no unpaid record exists, so nothing to add up
    # Nothing recorded: the schedule still reports the first EMI (start + 1 month).
    assert a.next_due_date == "2026-02-15"


# ------------------------------------------------------------- seeded demo data
def test_seeded_profiles_agree_with_the_ledger():
    """dpd_days > 0 must be exactly the borrowers with an overdue amount."""
    borrowers = json.loads((settings.DATA_DIR / "borrowers.json").read_text())
    payments = json.loads((settings.DATA_DIR / "payments.json").read_text())
    by_borrower: dict[str, list[dict]] = {}
    for p in payments:
        by_borrower.setdefault(p["borrower_id"], []).append(p)

    mismatched = [
        b["borrower_id"] for b in borrowers
        if (b["dpd_days"] > 0) != (_derive_analytics(b, by_borrower.get(b["borrower_id"], [])).overdue_amount > 0)
    ]
    assert mismatched == [], f"profile and ledger disagree for {mismatched[:5]}"


def test_seeded_due_dates_follow_the_calendar():
    """Generated due dates are real calendar months apart, never start + n*30 days."""
    payments = json.loads((settings.DATA_DIR / "payments.json").read_text())
    first_due = next(p for p in payments if p["installment_no"] == 1)
    start = date.fromisoformat(
        next(b["loan_start_date"] for b in
             json.loads((settings.DATA_DIR / "borrowers.json").read_text())
             if b["borrower_id"] == first_due["borrower_id"])
    )
    due = date.fromisoformat(first_due["due_date"])
    assert (due.year, due.month) == (dates.add_months(start, 1).year, dates.add_months(start, 1).month)
