"""Support / resistance levels for live prediction (Phase C).

Backtest finding (60d, Nifty 50, closed-session directional calls): the textbook
"resistance blocks a rally" is *inverted* for news-driven overnight moves. A
bullish call fires far better when price is pressed against resistance (a coiled
breakout) than when it has open room in mid-range:

    long within ~1% of resistance : 55.7% hit  (n=413)
    long with room to run         : 36.0% hit  (n=506)

So structure is used as a **confirmation gate for bullish calls**: a long only
earns a confident action + conviction boost when it sits at a breakout barrier.
Shorts already carry a strong standalone edge, so they are left largely alone.

Levels are swing pivots (fractal highs/lows) clustered together, plus the 20-day
range extremes. The fractal method inherently ignores the most recent bars, so
today's in-progress candle never becomes its own "level".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .prices import load_ohlcv

# Within this % of a level counts as "at" it.
AT_LEVEL_PCT = 1.0


def _swing_levels(df: pd.DataFrame, k: int = 3, lookback: int = 120) -> tuple[list[float], list[float]]:
    if df.empty or len(df) < (2 * k + 1):
        return [], []
    recent = df.tail(lookback)
    highs = recent["High"].to_numpy()
    lows = recent["Low"].to_numpy()
    n = len(recent)
    res: list[float] = []
    sup: list[float] = []
    for i in range(k, n - k):
        wh = highs[i - k : i + k + 1]
        wl = lows[i - k : i + k + 1]
        if highs[i] == wh.max() and (highs[i] > wh).sum() >= 1:
            res.append(float(highs[i]))
        if lows[i] == wl.min() and (lows[i] < wl).sum() >= 1:
            sup.append(float(lows[i]))
    return res, sup


def _cluster(levels: list[float], tol_pct: float = 0.6) -> list[float]:
    if not levels:
        return []
    ordered = sorted(levels)
    clusters: list[list[float]] = [[ordered[0]]]
    for lv in ordered[1:]:
        if abs(lv - clusters[-1][-1]) / clusters[-1][-1] * 100.0 <= tol_pct:
            clusters[-1].append(lv)
        else:
            clusters.append([lv])
    return [float(np.mean(c)) for c in clusters]


def _nearest_above(levels: list[float], price: float) -> float | None:
    above = [lv for lv in levels if lv > price]
    return min(above) if above else None


def _nearest_below(levels: list[float], price: float) -> float | None:
    below = [lv for lv in levels if lv < price]
    return max(below) if below else None


def sr_position(symbol: str, price: float | None) -> dict:
    """Where ``price`` sits relative to the nearest chart barriers.

    Returns an empty dict when there isn't enough daily history, so callers can
    treat "no structure" as neutral (no gate, no boost).
    """
    if not price or price <= 0:
        return {}
    df = load_ohlcv(symbol)
    if df.empty or len(df) < 25:
        return {}

    res_raw, sup_raw = _swing_levels(df)
    window = df.tail(20)
    resistances = _cluster(res_raw) + [float(window["High"].max())]
    supports = _cluster(sup_raw) + [float(window["Low"].min())]

    resistance = _nearest_above(resistances, price)
    support = _nearest_below(supports, price)
    dist_res = round((resistance - price) / price * 100.0, 3) if resistance else None
    dist_sup = round((price - support) / price * 100.0, 3) if support else None

    band = None
    if resistance and support and resistance > support:
        band = round((price - support) / (resistance - support), 3)

    at_res = dist_res is not None and dist_res <= AT_LEVEL_PCT
    at_sup = dist_sup is not None and dist_sup <= AT_LEVEL_PCT
    return {
        "nearestResistance": round(resistance, 2) if resistance else None,
        "nearestSupport": round(support, 2) if support else None,
        "distResistPct": dist_res,
        "distSupportPct": dist_sup,
        "bandPos": band,
        "atResistance": at_res,
        "atSupport": at_sup,
        "breakout20": bool(price >= float(window["High"].max())),
        "breakdown20": bool(price <= float(window["Low"].min())),
    }


def sr_conviction_factor(sentiment: str, pos: dict) -> float:
    """Structure multiplier for conviction (Phase C).

    Bullish: reward a breakout setup (at resistance / already breaking out),
    penalise a mid-range long with no barrier to press against — historically
    the weakest bullish surface. Bearish: shorts are strong regardless, with a
    small nod to breakdown / at-support continuation.
    """
    if not pos:
        return 1.0
    if sentiment == "Positive":
        if pos.get("breakout20") or pos.get("atResistance"):
            return 1.25
        return 0.6
    if sentiment == "Negative":
        if pos.get("breakdown20") or pos.get("atSupport"):
            return 1.1
        return 1.0
    return 1.0


def is_breakout_long(sentiment: str, pos: dict) -> bool:
    """True when a bullish call has structural confirmation (breakout setup)."""
    if sentiment != "Positive" or not pos:
        return False
    return bool(pos.get("breakout20") or pos.get("atResistance"))
