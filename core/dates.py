"""Date & time utilities — the single source of "now" for the whole project.

Two jobs:

1. A controllable clock. Nothing in the app hardcodes a date anymore; everyone
   calls `today()` / `now()`. Tests and demos can freeze time with `set_now()`
   (or the `LENDCO_NOW` environment variable) instead of the code baking in a
   fixed timestamp.

2. Calendar-correct month arithmetic. `add_months()` adds real calendar months
   and clamps the day (31 Jan + 1 month = 28 Feb), so we never assume every
   month has 30 days.

Timezone strategy: every timestamp is UTC, returned as a timezone-aware
datetime. ISO-8601 strings produced here always end with "+00:00".
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone

# Frozen clock (set by tests/demos). None => use the real system clock.
_FIXED_NOW: datetime | None = None


def set_now(value: datetime | None) -> None:
    """Freeze the project clock at `value` (or pass None to unfreeze).

    Naive datetimes are interpreted as UTC so there is only one timezone in
    the system. Tests use this: dates.set_now(datetime(2026, 6, 9, 10, 0, 0)).
    """
    global _FIXED_NOW
    if value is not None and value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    _FIXED_NOW = value


def now() -> datetime:
    """Current UTC time (frozen value while `set_now` is active)."""
    if _FIXED_NOW is not None:
        return _FIXED_NOW
    override = os.getenv("LENDCO_NOW", "").strip()
    if override:
        parsed = datetime.fromisoformat(override)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    return datetime.now(timezone.utc)


def today() -> date:
    """Current UTC date."""
    return now().date()


def iso_now() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return now().replace(microsecond=0).isoformat()


def add_months(d: date, months: int) -> date:
    """Add `months` calendar months to `d`, clamping the day of month.

    add_months(date(2026, 1, 31), 1) -> date(2026, 2, 28)
    add_months(date(2026, 1, 31), 13) -> date(2027, 2, 28)
    Negative values go backwards. Used for EMI due-date schedules so we never
    assume 30-day months.
    """
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, _days_in_month(year, month))
    return d.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return (next_month - date(year, month, 1)).days