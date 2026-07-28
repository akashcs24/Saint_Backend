"""Overnight thesis live health — behaviour check during cash hours.

Not a watchlist or order book. For an overnight directional call we only ask:
is the stock still acting like the prediction?

States (time-updating):

* ``confirming``  — open/path with the thesis
* ``fading``      — had the thesis then losing it (incl. giveback from the high)
* ``invalidated`` — opened against the thesis
* ``cooling``     — flat / no follow-through yet
* ``pending``     — market not open yet / no open print

During the first ~5 minutes only the gap vs baseline is available. After the
first completed 15m bar (~09:30) and open+30m, the state can tighten or flip.

Gap / path are measured vs the **prior cash-session close** (the print the
overnight call is betting on), not vs a days-old news baseline. Otherwise
"open move" disagrees with the chart gap.

Peak giveback: once the trade ran a meaningful favorable move (default ≥2%),
fading triggers when ≥50% of that run is given back — even if price is still
green vs baseline. Waiting for a cross below 0 is too late after a +5% spike.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from .prices import fetch_intraday_15m, price_at_or_before
from .session import is_cash_session_open, now_ist, prior_session_close, session_bounds

GAP_THR = 0.3
HOLD_THR = 0.3
# Meaningful favorable run before peak-giveback can fire (avoids noise).
PEAK_MIN_RUN_PCT = 2.0
# Fraction of the peak run that must be given back → fading / watch.
PEAK_GIVEBACK_FRAC = 0.50
# Absolute trailing stop in percentage-points from peak (after arm).
# Matches backtest: trail ~2% off high once trade is at least +0.5%.
TRAIL_ARM_PCT = 0.5
TRAIL_DROP_PCT = 2.0
CLOSED_PHASES = frozenset({"after_close", "closed_day", "before_open"})


def _side(move: float | None, expected: int, thr: float) -> str:
    if move is None or expected == 0:
        return "na"
    if abs(move) < thr:
        return "flat"
    return "with" if (move > 0) == (expected > 0) else "against"


def _move_pct(base: float | None, px: float | None) -> float | None:
    if base is None or px is None or base <= 0:
        return None
    return round((px - base) / base * 100.0, 3)


def _session_gap_baseline(symbol: str, when: datetime | None = None) -> tuple[float | None, str | None]:
    """Prior cash close before this session — what the overnight gap is against.

    Prefers daily cache, but if that print is older than the true prior session
    (stale parquet), falls back to Yahoo's previous close so the gap matches
    the chart / day-change % the user sees.
    """
    from .quotes import get_quote
    from .session import prev_trading_day

    now = now_ist(when)
    close_at = prior_session_close(now)
    expected_day = prev_trading_day(now.date())
    label, price = price_at_or_before(symbol, close_at, prefer_intraday=False)

    cache_ok = False
    if price is not None and price > 0 and label and label.startswith("close@"):
        try:
            cache_day = date.fromisoformat(label.split("@", 1)[1][:10])
            cache_ok = cache_day >= expected_day
        except ValueError:
            cache_ok = False

    if cache_ok:
        return float(price), label

    q = get_quote(symbol)
    if q is not None:
        prev = q.previous_close
        if prev is None and q.ltp and q.change_pct is not None:
            denom = 1.0 + (q.change_pct / 100.0)
            if denom > 0:
                prev = q.ltp / denom
        if prev is not None and prev > 0:
            return float(prev), f"prevClose@{expected_day.isoformat()}"

    if price is not None and price > 0:
        return float(price), label
    return None, None


def _today_path(symbol: str, baseline: float, when: datetime | None = None) -> dict:
    """Open / +15m / +30m / last / session high-low vs prior-session close."""
    now = now_ist(when)
    day = now.date()
    open_dt, _ = session_bounds(day)
    out = {
        "openMovePct": None,
        "plus15MovePct": None,
        "plus30MovePct": None,
        "lastMovePct": None,
        "sessionHighPct": None,
        "sessionLowPct": None,
        "barsSeen": 0,
        "minutesSinceOpen": max(0, int((now - open_dt).total_seconds() // 60))
        if now >= open_dt
        else 0,
    }
    try:
        bars = fetch_intraday_15m(symbol)
    except Exception:  # noqa: BLE001
        return out
    if bars.empty:
        return out
    day_ts = pd.Timestamp(day)
    day_bars = bars.loc[bars.index.normalize() == day_ts]
    if day_bars.empty:
        return out
    out["barsSeen"] = int(len(day_bars))
    open_px = float(day_bars.iloc[0]["Open"])
    out["openMovePct"] = _move_pct(baseline, open_px)
    now_naive = now.replace(tzinfo=None) if getattr(now, "tzinfo", None) else now
    # First bar close ≈ open+15m once that bar has finished (or a newer bar exists).
    c15 = float(day_bars.iloc[0]["Close"])
    bar0_end = (pd.Timestamp(day_bars.index[0]) + pd.Timedelta(minutes=15)).to_pydatetime()
    if getattr(bar0_end, "tzinfo", None) is not None:
        bar0_end = bar0_end.replace(tzinfo=None)
    if now_naive >= bar0_end or len(day_bars) >= 2:
        out["plus15MovePct"] = _move_pct(baseline, c15)
    if len(day_bars) >= 2:
        out["plus30MovePct"] = _move_pct(baseline, float(day_bars.iloc[1]["Close"]))
    out["lastMovePct"] = _move_pct(baseline, float(day_bars.iloc[-1]["Close"]))
    if "High" in day_bars.columns:
        out["sessionHighPct"] = _move_pct(baseline, float(day_bars["High"].max()))
    if "Low" in day_bars.columns:
        out["sessionLowPct"] = _move_pct(baseline, float(day_bars["Low"].min()))
    return out


def _favorable_pct(move: float | None, expected: int) -> float | None:
    """Signed move flipped so + always means with the thesis."""
    if move is None or expected == 0:
        return None
    return float(move) if expected > 0 else float(-move)


def peak_giveback(
    *,
    expected_direction: int,
    last_move_pct: float | None,
    session_high_pct: float | None = None,
    session_low_pct: float | None = None,
    min_run_pct: float = PEAK_MIN_RUN_PCT,
    giveback_frac: float = PEAK_GIVEBACK_FRAC,
    trail_arm_pct: float = TRAIL_ARM_PCT,
    trail_drop_pct: float = TRAIL_DROP_PCT,
) -> dict:
    """Detect giveback / trailing stop from session high (long) / low (short).

    Two exit triggers (either fires → fading):
    1) Fractional giveback: peak ≥ min_run and gave back ≥ giveback_frac of the run
    2) Absolute trail: peak ≥ trail_arm and drop from peak ≥ trail_drop (pp)
    """
    empty = {
        "peakFavPct": None,
        "currentFavPct": None,
        "givebackFrac": None,
        "trailDropPct": None,
        "triggered": False,
        "triggerReason": None,
    }
    if expected_direction == 0:
        return empty

    if expected_direction > 0:
        peak_raw = session_high_pct
    else:
        peak_raw = session_low_pct

    peak_fav = _favorable_pct(peak_raw, expected_direction)
    cur_fav = _favorable_pct(last_move_pct, expected_direction)
    if peak_fav is None or cur_fav is None:
        return empty

    # Peak must be the best favorable print seen; never below current.
    peak_fav = max(peak_fav, cur_fav)
    drop = peak_fav - cur_fav
    out = {
        "peakFavPct": round(peak_fav, 3),
        "currentFavPct": round(cur_fav, 3),
        "givebackFrac": None,
        "trailDropPct": round(drop, 3),
        "triggered": False,
        "triggerReason": None,
    }
    if peak_fav <= 1e-9:
        return out

    gb = drop / peak_fav
    out["givebackFrac"] = round(gb, 3)

    # 1) Classic fractional giveback on a meaningful run.
    if peak_fav >= float(min_run_pct) and gb >= float(giveback_frac):
        out["triggered"] = True
        out["triggerReason"] = "giveback"
        return out

    # 2) Absolute trailing stop (earlier / smaller runs).
    if peak_fav >= float(trail_arm_pct) and drop >= float(trail_drop_pct):
        out["triggered"] = True
        out["triggerReason"] = "trail"
        return out

    return out


def classify_health(
    *,
    expected_direction: int,
    open_move_pct: float | None,
    plus15_move_pct: float | None = None,
    plus30_move_pct: float | None = None,
    last_move_pct: float | None = None,
    session_high_pct: float | None = None,
    session_low_pct: float | None = None,
    market_open: bool | None = None,
) -> dict:
    """Pure state machine — easy to unit test without Yahoo."""
    if market_open is None:
        market_open = is_cash_session_open()
    if expected_direction == 0:
        return {
            "thesisHealth": "na",
            "gapState": "na",
            "holdState": "na",
            "label": "No directional overnight call",
            "peakFavPct": None,
            "givebackFrac": None,
        }
    if not market_open and open_move_pct is None:
        return {
            "thesisHealth": "pending",
            "gapState": "na",
            "holdState": "na",
            "label": "Waiting for cash open",
            "peakFavPct": None,
            "givebackFrac": None,
        }

    gap = _side(open_move_pct, expected_direction, GAP_THR)
    # Prefer the most recent completed checkpoint for hold/fade.
    hold_move = plus30_move_pct if plus30_move_pct is not None else plus15_move_pct
    if hold_move is None:
        hold_move = last_move_pct
    hold = _side(hold_move, expected_direction, HOLD_THR)

    # Live LTP / last print is what the trader sees right now.
    live_move = last_move_pct if last_move_pct is not None else hold_move
    peak = peak_giveback(
        expected_direction=expected_direction,
        last_move_pct=live_move,
        session_high_pct=session_high_pct,
        session_low_pct=session_low_pct,
    )

    if gap == "against":
        health = "invalidated"
        label = "Opened against the overnight call"
    elif peak["triggered"]:
        # Fire before waiting for a cross below baseline (0).
        health = "fading"
        peak_v = peak["peakFavPct"]
        cur_v = peak["currentFavPct"]
        if peak.get("triggerReason") == "trail":
            drop = peak.get("trailDropPct")
            label = (
                f"Trail stop — dropped {drop:.1f}pp from peak "
                f"({peak_v:+.1f}% → {cur_v:+.1f}%) — exit/watch"
            )
        else:
            gb_pct = int(round((peak["givebackFrac"] or 0) * 100))
            label = (
                f"Gave back {gb_pct}% of the run "
                f"(peak {peak_v:+.1f}% → now {cur_v:+.1f}%) — exit/watch"
            )
    elif gap == "with" and hold == "with":
        health = "confirming"
        label = "Behaving with the overnight call"
    elif gap == "with" and hold == "against":
        health = "fading"
        label = "Opened with the call, now reversing"
    elif gap == "with" and hold in {"flat", "na"}:
        health = "confirming"
        label = "Gap with thesis — early confirmation"
    elif gap == "flat" and hold == "with":
        health = "confirming"
        label = "Follow-through building after a flat open"
    elif gap == "flat" and hold == "against":
        health = "fading"
        label = "No gap, then moved against the call"
    elif gap == "flat":
        health = "cooling"
        label = "Flat open — no follow-through yet"
    else:
        health = "cooling"
        label = "Waiting for a clearer open path"

    return {
        "thesisHealth": health,
        "gapState": gap,
        "holdState": hold,
        "label": label,
        "peakFavPct": peak.get("peakFavPct"),
        "currentFavPct": peak.get("currentFavPct"),
        "givebackFrac": peak.get("givebackFrac"),
        "trailDropPct": peak.get("trailDropPct"),
        "exitTrigger": peak.get("triggerReason"),
    }


def apply_thesis_exit(
    action: str,
    thesis_health: str | None,
    *,
    action_note: str | None = None,
) -> tuple[str, str | None]:
    """Hold/exit overlay — live path can kill a published buy after open.

    * invalidated → watch (exit — opened against the call)
    * fading → watch (hold/exit — thesis losing grip)
    * confirming / cooling / pending / na → leave action alone
    """
    if action in {"already priced", "already fallen", "watch"}:
        return action, action_note
    if action not in {"buy long", "buy short", "buy", "short", "avoid"}:
        return action, action_note

    if thesis_health == "invalidated":
        tag = "exit — opened against the call"
        note = f"{action_note} · {tag}" if action_note else tag
        return "watch", note
    if thesis_health == "fading":
        tag = "hold/exit — thesis fading (giveback or reverse)"
        note = f"{action_note} · {tag}" if action_note else tag
        return "watch", note
    return action, action_note


def thesis_health_for_stock(
    symbol: str,
    *,
    expected_direction: int,
    baseline_price: float | None,
    session_phase: str | None,
    current_ltp: float | None = None,
) -> dict | None:
    """Attach live thesis health for overnight calls once cash is open (or pending)."""
    if expected_direction == 0 or not session_phase or session_phase not in CLOSED_PHASES:
        return None

    market_open = is_cash_session_open()
    # Prefer prior session close for THIS open. News baseline can be days old
    # (e.g. close@Jul-21 while validating on Jul-27) and will not match the chart gap.
    gap_base, gap_label = _session_gap_baseline(symbol)
    if gap_base is None:
        if baseline_price is None or baseline_price <= 0:
            return None
        gap_base = float(baseline_price)
        gap_label = "news-baseline"

    path = (
        _today_path(symbol, float(gap_base))
        if market_open
        else {
            "openMovePct": None,
            "plus15MovePct": None,
            "plus30MovePct": None,
            "lastMovePct": None,
            "sessionHighPct": None,
            "sessionLowPct": None,
            "barsSeen": 0,
            "minutesSinceOpen": 0,
        }
    )
    # Prefer live LTP as the freshest "last" when quotes are available.
    if market_open and current_ltp and current_ltp > 0:
        path["lastMovePct"] = _move_pct(float(gap_base), float(current_ltp))
        # Extend session extremes with live print (15m high can lag).
        last_pct = path["lastMovePct"]
        if last_pct is not None:
            hi = path.get("sessionHighPct")
            lo = path.get("sessionLowPct")
            path["sessionHighPct"] = last_pct if hi is None else max(float(hi), last_pct)
            path["sessionLowPct"] = last_pct if lo is None else min(float(lo), last_pct)
        # Before any 15m bar exists, LTP is also the best open proxy.
        if path["openMovePct"] is None:
            path["openMovePct"] = path["lastMovePct"]

    state = classify_health(
        expected_direction=expected_direction,
        open_move_pct=path.get("openMovePct"),
        plus15_move_pct=path.get("plus15MovePct"),
        plus30_move_pct=path.get("plus30MovePct"),
        last_move_pct=path.get("lastMovePct"),
        session_high_pct=path.get("sessionHighPct"),
        session_low_pct=path.get("sessionLowPct"),
        market_open=market_open,
    )
    return {
        **state,
        "openMovePct": path.get("openMovePct"),
        "plus15MovePct": path.get("plus15MovePct"),
        "plus30MovePct": path.get("plus30MovePct"),
        "lastMovePct": path.get("lastMovePct"),
        "sessionHighPct": path.get("sessionHighPct"),
        "sessionLowPct": path.get("sessionLowPct"),
        "barsSeen": path.get("barsSeen"),
        "minutesSinceOpen": path.get("minutesSinceOpen"),
        "gapBaselinePrice": round(float(gap_base), 2),
        "gapBaselineLabel": gap_label,
    }
