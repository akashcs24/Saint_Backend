"""Action confirmation layer — volume/price gates on executable labels.

Bias/conviction stay news-driven. This module only demotes ``buy long`` /
``buy short`` → ``watch`` when tape confirmation fails (backtest-backed):

Overnight / closed-session news
* Longs: veto prior-day *no demand* (quiet vol + up day)
* Longs: require prior-day RVOL ≥ 1.25 *or* quality tape (below VWAP / surge)
* After cash open: prefer first 15m open vol elevated + price with news

Live / open-session news
* Require 15m window (1 before + news + up to 2 after) TOD RVOL elevate
  *and* window price direction agreeing with the news

Failures never flip bias — only the actionable label.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .prices import fetch_intraday_15m
from .session import is_cash_session_open, now_ist, prev_trading_day

OPEN_TOD_MIN = 9 * 60 + 15
LOOKBACK_DAYS = 20
RVOL_ELEVATED = 1.25
RVOL_STRONG = 1.5
QUIET_RVOL = 0.8
PRICE_MOVE_MIN = 0.15  # % — tiny moves don't count as alignment


def _ist_naive(when) -> pd.Timestamp | None:
    if when is None:
        return None
    ts = pd.Timestamp(when)
    if getattr(ts, "tz", None) is not None:
        return ts.tz_convert("Asia/Kolkata").tz_localize(None)
    return ts.tz_localize(None)


def _load_15m(symbol: str) -> pd.DataFrame:
    try:
        df = fetch_intraday_15m(symbol, force=False)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df is None or df.empty or "Volume" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["tod"] = out.index.hour * 60 + out.index.minute
    out["day"] = out.index.normalize()
    return out


def enrich_tape_scenarios(tape: dict | None) -> dict:
    """Add no-demand / elevated flags onto prior_tape snapshot."""
    t = dict(tape or {})
    if not t:
        return t
    vol = t.get("volRatio20")
    chg = t.get("priorDayChangePct")
    quiet = vol is not None and float(vol) < QUIET_RVOL
    elevated = vol is not None and float(vol) >= RVOL_ELEVATED
    surge = bool(t.get("volSurge"))
    t["volQuiet"] = quiet
    t["volElevated"] = elevated or surge
    chg_f = float(chg) if chg is not None else None
    t["noDemand"] = bool(quiet and chg_f is not None and chg_f > 0.3)
    t["noSupply"] = bool(quiet and chg_f is not None and chg_f < -0.3)
    return t


def _first_candle(day_bars: pd.DataFrame) -> pd.Series | None:
    if day_bars.empty:
        return None
    exact = day_bars.loc[day_bars["tod"] == OPEN_TOD_MIN]
    if not exact.empty:
        return exact.iloc[0]
    return day_bars.iloc[0]


def open_first_candle_confirm(symbol: str) -> dict[str, Any]:
    """Today's 09:15 vol vs prior day's 09:15 (overnight news → open flow)."""
    out: dict[str, Any] = {
        "ready": False,
        "elevated": False,
        "strong": False,
        "ratioPrior": None,
        "priceChangePct": None,
        "priceAlignedLong": None,
        "priceAlignedShort": None,
    }
    if not is_cash_session_open():
        return out
    df = _load_15m(symbol)
    if df.empty:
        return out
    today = pd.Timestamp(now_ist().date())
    prior = pd.Timestamp(prev_trading_day(now_ist().date()))
    tgt = _first_candle(df.loc[df["day"] == today.normalize()])
    pri = _first_candle(df.loc[df["day"] == prior.normalize()])
    if tgt is None or pri is None:
        return out
    vol_t = float(tgt.get("Volume") or 0)
    vol_p = float(pri.get("Volume") or 0)
    if vol_t <= 0 or vol_p <= 0:
        return out
    ratio = vol_t / vol_p
    px = None
    o = float(tgt.get("Open") or 0)
    c = float(tgt.get("Close") or 0)
    if o > 0:
        px = (c - o) / o * 100.0
    out.update(
        {
            "ready": True,
            "elevated": ratio >= RVOL_ELEVATED,
            "strong": ratio >= RVOL_STRONG,
            "ratioPrior": round(ratio, 3),
            "priceChangePct": round(px, 3) if px is not None else None,
            "priceAlignedLong": px is not None and px > PRICE_MOVE_MIN,
            "priceAlignedShort": px is not None and px < -PRICE_MOVE_MIN,
            "volTarget": int(vol_t),
            "volPrior": int(vol_p),
        }
    )
    return out


def live_window_confirm(symbol: str, published_at) -> dict[str, Any]:
    """1 before + news + up to 2 after, vs same clock-time average."""
    out: dict[str, Any] = {
        "ready": False,
        "confirm": False,
        "confirmRealtime": False,
        "priceAlignedLong": None,
        "priceAlignedShort": None,
        "rvolWindow": None,
        "rvolMax": None,
    }
    ts = _ist_naive(published_at)
    if ts is None:
        return out
    df = _load_15m(symbol)
    if df.empty:
        return out

    minute = (ts.minute // 15) * 15
    news_start = ts.normalize() + pd.Timedelta(hours=int(ts.hour), minutes=int(minute))
    started = df.loc[df.index <= news_start]
    if started.empty:
        return out
    news_idx = started.index[-1]
    loc = int(df.index.get_indexer([news_idx])[0])
    if loc < 1 or loc + 1 >= len(df):
        return out
    # Prefer full window when after bars exist; else realtime (before+news).
    after = 2 if loc + 2 < len(df) else 1 if loc + 1 < len(df) else 0
    i0, i1 = loc - 1, loc + after
    window = df.iloc[i0 : i1 + 1]
    if window["day"].nunique() != 1:
        return out

    news_day = pd.Timestamp(df.index[loc]).normalize()
    hist = df.loc[df["day"] < news_day]
    if hist.empty:
        return out
    days = sorted(hist["day"].unique())[-LOOKBACK_DAYS:]
    hist = hist.loc[hist["day"].isin(days)]
    tod_means = hist.groupby("tod")["Volume"].mean().to_dict()

    vols = [float(x) for x in window["Volume"].fillna(0.0).tolist()]
    rvols: list[float] = []
    bases: list[float] = []
    for v, t in zip(vols, window.index, strict=True):
        tod = int(pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute)
        avg = float(tod_means.get(tod) or 0)
        if avg > 0:
            rvols.append(v / avg)
            bases.append(avg)
        else:
            rvols.append(float("nan"))

    if not bases:
        return out
    window_rvol = sum(vols) / sum(bases)
    valid = [r for r in rvols if r == r]
    r_max = max(valid) if valid else None
    confirm = window_rvol >= RVOL_ELEVATED or (r_max is not None and r_max >= RVOL_STRONG)

    # Realtime: before + news only
    rt_vols = vols[:2]
    rt_bases = []
    for t in list(window.index)[:2]:
        tod = int(pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute)
        avg = float(tod_means.get(tod) or 0)
        if avg > 0:
            rt_bases.append(avg)
    rt_confirm = False
    if rt_bases:
        rt_rvol = sum(rt_vols) / sum(rt_bases)
        rt_valid = [r for r in rvols[:2] if r == r]
        rt_confirm = rt_rvol >= RVOL_ELEVATED or (rt_valid and max(rt_valid) >= RVOL_STRONG)

    px = None
    if "Close" in window.columns and len(window) >= 2:
        c0 = float(window.iloc[0]["Close"])
        c1 = float(window.iloc[-1]["Close"])
        if c0 > 0:
            px = (c1 - c0) / c0 * 100.0

    out.update(
        {
            "ready": True,
            "confirm": bool(confirm),
            "confirmRealtime": bool(rt_confirm),
            "rvolWindow": round(window_rvol, 3),
            "rvolMax": round(r_max, 3) if r_max is not None else None,
            "priceChangePct": round(px, 3) if px is not None else None,
            "priceAlignedLong": px is not None and px > PRICE_MOVE_MIN,
            "priceAlignedShort": px is not None and px < -PRICE_MOVE_MIN,
        }
    )
    return out


def apply_action_confirmation(
    symbol: str,
    *,
    action: str,
    action_note: str | None,
    closed_session_news: bool,
    tape: dict | None,
    anchor: dict | None,
) -> dict[str, Any]:
    """Demote executable actions when volume/price confirmation fails."""
    act = (action or "watch").strip().lower()
    note = action_note
    reasons: list[str] = []
    status = "n/a"
    open_snap: dict = {}
    live_snap: dict = {}
    tape_x = enrich_tape_scenarios(tape)

    if act not in {"buy long", "buy", "buy short", "short"}:
        return {
            "action": action,
            "actionNote": note,
            "actionConfirm": status,
            "actionConfirmReasons": reasons,
            "actionConfirmOk": True,
            "tapeConfirm": tape_x or None,
            "openConfirm": None,
            "liveConfirm": None,
        }

    is_long = act in {"buy long", "buy"}

    if closed_session_news:
        status = "confirmed"
        # P0 — no-demand veto for longs
        if is_long and tape_x.get("noDemand"):
            act = "watch"
            reasons.append("no demand on prior tape")
            status = "demoted"
        # P0 — overnight longs need elevated prior RVOL or quality tape
        elif is_long:
            quality = bool(
                tape_x.get("volElevated")
                or tape_x.get("volSurge")
                or tape_x.get("closeBelowVwap")
            )
            extended = bool(tape_x.get("extendedAboveVwap"))
            if extended:
                act = "watch"
                reasons.append("extended above VWAP — fade risk")
                status = "demoted"
            elif not quality and tape_x:
                # Weak/quiet prior day — hold as watch until open confirms
                if is_cash_session_open():
                    open_snap = open_first_candle_confirm(symbol)
                    if open_snap.get("ready") and open_snap.get("elevated"):
                        aligned = open_snap.get("priceAlignedLong") if is_long else open_snap.get(
                            "priceAlignedShort"
                        )
                        if aligned:
                            status = "confirmed"
                            reasons.append("open flow confirmed (weak prior tape)")
                        else:
                            act = "watch"
                            reasons.append("open volume without price confirm")
                            status = "awaiting"
                    else:
                        act = "watch"
                        reasons.append("awaiting open volume confirm")
                        status = "awaiting"
                else:
                    act = "watch"
                    reasons.append("prior tape not elevated — wait for open")
                    status = "awaiting"
            else:
                # Strong prior tape — still refine after open when available
                if is_cash_session_open():
                    open_snap = open_first_candle_confirm(symbol)
                    if open_snap.get("ready"):
                        aligned = (
                            open_snap.get("priceAlignedLong")
                            if is_long
                            else open_snap.get("priceAlignedShort")
                        )
                        if open_snap.get("elevated") and aligned:
                            reasons.append("open1 vol+price confirmed")
                            status = "confirmed"
                        elif open_snap.get("elevated") and aligned is False:
                            act = "watch"
                            reasons.append("open volume vs news price diverge")
                            status = "demoted"
                        # if open not elevated yet but prior was hot — keep buy
                        elif not open_snap.get("elevated"):
                            reasons.append("prior tape ok; open vol still quiet")
                            status = "confirmed"
        else:
            # Shorts: optional open confirm when available (soft demote only on clear diverge)
            if is_cash_session_open():
                open_snap = open_first_candle_confirm(symbol)
                if open_snap.get("ready") and open_snap.get("elevated"):
                    if open_snap.get("priceAlignedShort"):
                        reasons.append("open1 confirms short")
                        status = "confirmed"
                    elif open_snap.get("priceAlignedLong"):
                        act = "watch"
                        reasons.append("open flow against short thesis")
                        status = "demoted"
    else:
        # Live session news — need 15m window confirm + price align
        pub = (anchor or {}).get("publishedAt")
        live_snap = live_window_confirm(symbol, pub)
        vol_ok = bool(live_snap.get("confirm") or live_snap.get("confirmRealtime"))
        aligned = (
            live_snap.get("priceAlignedLong")
            if is_long
            else live_snap.get("priceAlignedShort")
        )
        if not live_snap.get("ready"):
            act = "watch"
            reasons.append("awaiting live volume window")
            status = "awaiting"
        elif vol_ok and aligned:
            status = "confirmed"
            reasons.append("live vol+price window confirmed")
        elif vol_ok and aligned is False:
            act = "watch"
            reasons.append("live volume without price confirm")
            status = "demoted"
        else:
            act = "watch"
            reasons.append("live tape quiet — wait for volume")
            status = "awaiting"

    if reasons:
        tag = reasons[0]
        note = f"{note} · {tag}" if note else tag

    # Normalize action labels
    if act == "buy":
        act = "buy long"
    if act == "short":
        act = "buy short"

    return {
        "action": act,
        "actionNote": note,
        "actionConfirm": status,
        "actionConfirmReasons": reasons[:4],
        "actionConfirmOk": status in {"confirmed", "n/a"},
        "tapeConfirm": tape_x or None,
        "openConfirm": open_snap or None,
        "liveConfirm": live_snap or None,
    }
