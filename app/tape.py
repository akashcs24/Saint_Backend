"""Prior-session volume / VWAP tape for overnight news (Phase D).

Backtest finding (60d): these gates help **closed-session** bullish calls and
do **not** transfer to live/open-session news (flat ~78%, prior-day tape
irrelevant). Saint therefore applies them only when the evidence set is
overnight-dominated.

Graded long-side gate (shorts unchanged):

* **Veto / demote** — prior close ≥ 1% above session VWAP (5% hit historically)
* **Boost** — prior day volume > 1.5× SMA20
* **Mild boost** — prior close still below session VWAP (dip + news)

Session VWAP is reconstructed from cached 15m bars when available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .prices import fetch_intraday_15m, load_ohlcv
from .session import classify_published_at, is_cash_session_open, now_ist, prev_trading_day

EXTENDED_PCT = 1.0
VOL_SURGE_K = 1.5
CLOSED_PHASES = frozenset({"after_close", "closed_day", "before_open"})


def _session_vwap(day_bars: pd.DataFrame) -> float | None:
    if day_bars.empty or "Volume" not in day_bars.columns:
        return None
    tp = (day_bars["High"] + day_bars["Low"] + day_bars["Close"]) / 3.0
    vol = day_bars["Volume"].fillna(0.0).clip(lower=0.0)
    cv = float(vol.sum())
    if cv <= 0:
        return None
    return float((tp * vol).sum() / cv)


def _prior_completed_session_day() -> pd.Timestamp:
    """Most recent fully finished cash session (excludes today's open bar)."""
    now = now_ist()
    d = now.date()
    if is_cash_session_open(now):
        # Today still forming — use previous trading day.
        prior = prev_trading_day(d)
    else:
        # After close / weekend / holiday: if today traded and is closed, today
        # is the completed session; else walk back.
        from .session import is_trading_day

        if is_trading_day(d) and classify_published_at(now) == "after_close":
            prior = d
        else:
            prior = prev_trading_day(d)
    return pd.Timestamp(prior)


def prior_tape(symbol: str) -> dict:
    """Volume / VWAP posture of the last completed session."""
    daily = load_ohlcv(symbol)
    if daily.empty or len(daily) < 25:
        return {}

    session_day = _prior_completed_session_day()
    # Daily index may be tz-naive dates.
    day_rows = daily.loc[daily.index.normalize() == session_day.normalize()]
    if day_rows.empty:
        # Fall back to last daily bar strictly on/before session_day.
        prior = daily.loc[daily.index.normalize() <= session_day.normalize()]
        if prior.empty:
            return {}
        row = prior.iloc[-1]
        session_day = pd.Timestamp(prior.index[-1]).normalize()
    else:
        row = day_rows.iloc[-1]

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])
    volume = float(row["Volume"]) if "Volume" in row.index else 0.0

    hist = daily.loc[daily.index.normalize() <= session_day.normalize()].tail(20)
    vol_ratio = None
    if len(hist) >= 10 and "Volume" in hist.columns:
        sma = float(hist["Volume"].mean())
        vol_ratio = (volume / sma) if sma else None

    # Prefer 15m-reconstructed session VWAP; fall back to typical price.
    session_vwap = None
    try:
        bars = fetch_intraday_15m(symbol)
        if not bars.empty:
            day_bars = bars.loc[bars.index.normalize() == session_day.normalize()]
            session_vwap = _session_vwap(day_bars)
    except Exception:  # noqa: BLE001
        session_vwap = None
    if session_vwap is None:
        session_vwap = (high + low + close) / 3.0

    dist_vwap_pct = round((close - session_vwap) / session_vwap * 100.0, 3)
    vol_surge = bool(vol_ratio is not None and vol_ratio > VOL_SURGE_K)
    below = close < session_vwap
    extended = dist_vwap_pct >= EXTENDED_PCT

    return {
        "tapeAsOf": session_day.strftime("%Y-%m-%d"),
        "sessionVwap": round(session_vwap, 2),
        "distVwapPct": dist_vwap_pct,
        "closeBelowVwap": below,
        "extendedAboveVwap": extended,
        "volRatio20": round(vol_ratio, 3) if vol_ratio is not None else None,
        "volSurge": vol_surge,
    }


def news_is_closed_session(related_news: list[dict]) -> bool:
    """True when evidence weight is majority overnight / next-open."""
    closed_w = 0.0
    live_w = 0.0
    for n in related_news:
        impact = float(n.get("impact") or 1) / 10.0
        relevance = float(n.get("relevance", 1.0))
        credibility = float(n.get("credibility") or 0.6)
        w = impact * relevance * credibility
        if w <= 0:
            continue
        phase = classify_published_at(n.get("publishedAt"))
        if phase in CLOSED_PHASES:
            closed_w += w
        elif phase == "during_market":
            live_w += w
    total = closed_w + live_w
    if total <= 0:
        # No timing → treat as closed only if market is currently closed
        # (overnight board); otherwise neutral/live.
        return not is_cash_session_open()
    return closed_w >= live_w


def tape_conviction_factor(sentiment: str, tape: dict, *, closed_session: bool) -> float:
    """Long-side graded factor; neutral on live news and on shorts."""
    if not closed_session or not tape or sentiment != "Positive":
        return 1.0
    if tape.get("extendedAboveVwap"):
        return 0.45  # hard demote — historically ~5% hit
    factor = 1.0
    if tape.get("volSurge"):
        factor *= 1.2
    if tape.get("closeBelowVwap"):
        factor *= 1.1
    return round(factor, 3)


def tape_blocks_buy(sentiment: str, tape: dict, *, closed_session: bool) -> bool:
    """Overnight bullish call stretched above VWAP must not become ``buy``."""
    if not closed_session or sentiment != "Positive" or not tape:
        return False
    return bool(tape.get("extendedAboveVwap"))


def tape_supports_buy(sentiment: str, tape: dict, *, closed_session: bool) -> bool:
    """Soft confirmation: vol surge or still below VWAP (and not extended)."""
    if not closed_session or sentiment != "Positive" or not tape:
        return False
    if tape.get("extendedAboveVwap"):
        return False
    return bool(tape.get("volSurge") or tape.get("closeBelowVwap"))
