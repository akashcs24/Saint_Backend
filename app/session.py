"""NSE cash-market session clock (Asia/Kolkata).

Hours: Mon–Fri 09:15–15:30, skipping weekends and official holidays via
``nse-calendar``. Special sessions (e.g. Muhurat) are treated as trading days
when the calendar reports them.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
# Live quotes / Fyers / Nifty board refresh window (slightly before cash open).
LIVE_DATA_OPEN = time(9, 14)
LIVE_DATA_CLOSE = time(15, 30)
# First verification checkpoint after open for overnight / next-session calls.
OPEN_PLUS_MINUTES = 30

SessionPhase = Literal["before_open", "during_market", "after_close", "closed_day"]
ReactionBucket = Literal["next_session", "live_session", "already_reacted"]


def now_ist(when: datetime | None = None) -> datetime:
    if when is None:
        return datetime.now(IST)
    if when.tzinfo is None:
        # Naive timestamps from RSS are treated as UTC, then converted.
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(IST)


def to_ist(when: datetime | date | str | None) -> datetime | None:
    if when is None:
        return None
    if isinstance(when, date) and not isinstance(when, datetime):
        return datetime.combine(when, time(0, 0), tzinfo=IST)
    if isinstance(when, str):
        try:
            when = datetime.fromisoformat(when.replace("Z", "+00:00"))
        except ValueError:
            return None
    return now_ist(when)


def _is_holiday(d: date) -> bool:
    try:
        import nse_calendar as nse

        return bool(nse.is_nse_holiday(d.isoformat()))
    except Exception:
        # Offline / package failure → weekends only.
        return d.weekday() >= 5


def is_trading_day(d: date | datetime | None = None) -> bool:
    if d is None:
        d = now_ist().date()
    elif isinstance(d, datetime):
        d = now_ist(d).date()
    return not _is_holiday(d)


def next_trading_day(d: date | datetime | None = None) -> date:
    if d is None:
        base = now_ist().date()
    elif isinstance(d, datetime):
        base = now_ist(d).date()
    else:
        base = d
    try:
        import nse_calendar as nse

        nxt = nse.next_trading_day(base.isoformat())
        if hasattr(nxt, "date"):
            return nxt.date() if isinstance(nxt, datetime) else nxt
        return date.fromisoformat(str(nxt)[:10])
    except Exception:
        cur = base + timedelta(days=1)
        while cur.weekday() >= 5:
            cur += timedelta(days=1)
        return cur


def prev_trading_day(d: date | datetime | None = None) -> date:
    if d is None:
        base = now_ist().date()
    elif isinstance(d, datetime):
        base = now_ist(d).date()
    else:
        base = d
    try:
        import nse_calendar as nse

        prv = nse.prev_trading_day(base.isoformat())
        if hasattr(prv, "date"):
            return prv.date() if isinstance(prv, datetime) else prv
        return date.fromisoformat(str(prv)[:10])
    except Exception:
        cur = base - timedelta(days=1)
        while cur.weekday() >= 5:
            cur -= timedelta(days=1)
        return cur


def session_bounds(d: date) -> tuple[datetime, datetime]:
    """Open and close datetimes (IST-aware) for a calendar date."""
    open_dt = datetime.combine(d, SESSION_OPEN, tzinfo=IST)
    close_dt = datetime.combine(d, SESSION_CLOSE, tzinfo=IST)
    return open_dt, close_dt


def is_cash_session_open(when: datetime | None = None) -> bool:
    now = now_ist(when)
    if not is_trading_day(now.date()):
        return False
    open_dt, close_dt = session_bounds(now.date())
    return open_dt <= now <= close_dt


def is_live_data_window(when: datetime | None = None) -> bool:
    """True Mon–Fri (trading days) 09:14–15:30 IST.

    Use this to pause Fyers probes, Nifty board live refresh, and Gemini OI
    insights overnight so we do not burn tokens after the cash close.
    """
    now = now_ist(when)
    if not is_trading_day(now.date()):
        return False
    start = datetime.combine(now.date(), LIVE_DATA_OPEN, tzinfo=IST)
    end = datetime.combine(now.date(), LIVE_DATA_CLOSE, tzinfo=IST)
    return start <= now <= end


def live_data_window_label() -> str:
    return "09:14–15:30 IST · trading days"


# First half-hour after open: gap filter + early thesis validation window.
OPEN_WINDOW_MINUTES = 30


def is_open_window(when: datetime | None = None) -> bool:
    """True during 09:15–09:45 IST on a trading day.

    This is when overnight calls need fast quotes for the first-5-minute gap
    decision and the first completed 15m bar.
    """
    now = now_ist(when)
    if not is_trading_day(now.date()):
        return False
    open_dt, _ = session_bounds(now.date())
    end = open_dt + timedelta(minutes=OPEN_WINDOW_MINUTES)
    return open_dt <= now <= end


def effective_quote_ttl_s(when: datetime | None = None) -> int:
    from .config import settings

    if is_open_window(when):
        return int(getattr(settings, "quote_ttl_open_s", 30))
    return int(settings.quote_ttl_s)


def effective_intraday_ttl_s(when: datetime | None = None) -> int:
    from .config import settings

    if is_open_window(when):
        return int(getattr(settings, "intraday_ttl_open_s", 60))
    return int(settings.intraday_ttl_s)


def classify_published_at(when: datetime | str | None) -> SessionPhase:
    """Where in the NSE day a headline landed."""
    ts = to_ist(when)
    if ts is None:
        return "closed_day"
    d = ts.date()
    if not is_trading_day(d):
        return "closed_day"
    open_dt, close_dt = session_bounds(d)
    if ts < open_dt:
        return "before_open"
    if ts <= close_dt:
        return "during_market"
    return "after_close"


def next_session_open(when: datetime | str | None = None) -> datetime:
    """The next cash-session open that has not yet started relative to ``when``."""
    ts = to_ist(when) or now_ist()
    d = ts.date()
    if is_trading_day(d):
        open_dt, _ = session_bounds(d)
        if ts < open_dt:
            return open_dt
    nxt = next_trading_day(d)
    open_dt, _ = session_bounds(nxt)
    return open_dt


def prior_session_close(when: datetime | str | None = None) -> datetime:
    """Most recent cash-session close at or before ``when``."""
    ts = to_ist(when) or now_ist()
    d = ts.date()
    if is_trading_day(d):
        _, close_dt = session_bounds(d)
        if ts >= close_dt:
            return close_dt
        # Still in/before today's session → previous trading day's close.
    prv = prev_trading_day(d)
    _, close_dt = session_bounds(prv)
    return close_dt


def checkpoint_times(session_date: date) -> dict[str, datetime]:
    """Open / +30m / close checkpoints for a trading day."""
    open_dt, close_dt = session_bounds(session_date)
    return {
        "open": open_dt,
        "open_plus": open_dt + timedelta(minutes=OPEN_PLUS_MINUTES),
        "close": close_dt,
    }


def target_session_date(published_at: datetime | str | None) -> date:
    """Trading day the story is expected to affect."""
    phase = classify_published_at(published_at)
    ts = to_ist(published_at) or now_ist()
    if phase == "during_market":
        return ts.date()
    if phase == "before_open" and is_trading_day(ts.date()):
        return ts.date()
    return next_trading_day(ts.date())


def session_snapshot(when: datetime | None = None) -> dict:
    now = now_ist(when)
    open_now = is_cash_session_open(now)
    phase: SessionPhase
    if not is_trading_day(now.date()):
        phase = "closed_day"
    else:
        open_dt, close_dt = session_bounds(now.date())
        if now < open_dt:
            phase = "before_open"
        elif now <= close_dt:
            phase = "during_market"
        else:
            phase = "after_close"
    nxt = next_session_open(now)
    open_window = is_open_window(now)
    return {
        "tz": "Asia/Kolkata",
        "now": now.isoformat(),
        "open": open_now,
        "openWindow": open_window,
        "quoteTtlS": effective_quote_ttl_s(now),
        "refreshHintMs": 30_000 if open_window else 300_000,
        "phase": phase,
        "nextOpen": nxt.isoformat(),
        "priorClose": prior_session_close(now).isoformat(),
        "tradingDay": is_trading_day(now.date()),
    }
