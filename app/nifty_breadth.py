"""Nifty 50 advance/decline breadth + weight-weighted lean.

Weights auto-refresh from smart-investing.in (cached ~12h), with a baked-in
fallback snapshot. Renormalized at runtime over successful quotes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .fyers_auth import get_access_token
from .fyers_quotes import fetch_fyers_quotes
from .nifty_pcr import fetch_nifty_pcr
from .nifty_weights import get_nifty_weights
from .quotes import get_quote
from .trend_history import record as record_trend
from .trend_history import trend_pair
from .universe import UNIVERSE

# Flat band — ignore noise in A/D count.
_FLAT_PCT = 0.05
# Weight-weighted contribution thresholds for a directional lean.
_LEAN_PCT = 0.12
_WEIGHT_TREND_EPS = 1.0  # percentage points of decline-weight
_BREADTH_CACHE: dict = {"ts": 0.0, "payload": None}
_BREADTH_TTL_S = 90.0  # reuse breadth board across dashboard rebuilds (Yahoo)
_BREADTH_TTL_FYERS_S = 1.0  # live poller refreshes quotes every ~1s per batch


def _breadth_ttl_s() -> float:
    if get_access_token():
        return _BREADTH_TTL_FYERS_S
    return _BREADTH_TTL_S


def _side(change_pct: float) -> str:
    if change_pct > _FLAT_PCT:
        return "up"
    if change_pct < -_FLAT_PCT:
        return "down"
    return "flat"


def build_nifty_breadth(*, max_workers: int = 12, force: bool = False) -> dict:
    """Advance/decline + weight-weighted Nifty lean from constituent quotes."""
    import time

    now = time.time()
    ttl = _breadth_ttl_s()
    if (
        not force
        and _BREADTH_CACHE["payload"] is not None
        and now - float(_BREADTH_CACHE["ts"]) < ttl
    ):
        out = dict(_BREADTH_CACHE["payload"])
        out["cached"] = True
        return out

    weight_pack = get_nifty_weights()
    nifty_weights: dict[str, float] = weight_pack.get("weights") or {}
    symbols = [s for s in nifty_weights if s in UNIVERSE]
    quotes: dict[str, object] = {}
    quote_source = "yahoo"

    # Prefer Fyers realtime when the user has connected from the app header.
    if get_access_token():
        fyers_map = fetch_fyers_quotes(symbols)
        if fyers_map:
            quotes.update(fyers_map)
            quote_source = "fyers"

    # Fill any gaps with Yahoo (or all-Yahoo when Fyers not connected).
    missing = [s for s in symbols if s not in quotes]

    def _one(sym: str):
        return sym, get_quote(sym)

    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_one, s) for s in missing]
            for fut in as_completed(futs):
                try:
                    sym, q = fut.result()
                except Exception:  # noqa: BLE001
                    continue
                if q is not None:
                    quotes[sym] = q
        if quote_source == "fyers" and missing:
            quote_source = "fyers+yahoo"

    pcr = fetch_nifty_pcr()
    weights_meta = {
        "source": weight_pack.get("source"),
        "count": weight_pack.get("count"),
        "fetchedAt": weight_pack.get("fetchedAt"),
        "unmapped": weight_pack.get("unmapped") or [],
        "quoteSource": quote_source,
    }

    if not quotes:
        return {
            "ready": False,
            "advances": 0,
            "declines": 0,
            "unchanged": 0,
            "quoted": 0,
            "universe": len(symbols),
            "weightUp": 0.0,
            "weightDown": 0.0,
            "weightFlat": 0.0,
            "contributionPct": None,
            "lean": "unclear",
            "action": "watch",
            "label": "Waiting for constituent quotes",
            "topUp": [],
            "topDown": [],
            "segments": [],
            "pcr": pcr,
            "weightsMeta": weights_meta,
        }

    raw_w = {s: nifty_weights[s] for s in quotes}
    total_w = sum(raw_w.values()) or 1.0
    # Renormalize so available names sum to 100.
    weights = {s: (w / total_w) * 100.0 for s, w in raw_w.items()}

    advances = declines = unchanged = 0
    weight_up = weight_down = weight_flat = 0.0
    contribution = 0.0
    movers: list[tuple[str, float, float, float, str]] = []

    for sym, q in quotes.items():
        chg = float(getattr(q, "change_pct", 0.0) or 0.0)
        w = weights[sym]
        side = _side(chg)
        contrib = (w / 100.0) * chg
        contribution += contrib
        movers.append((sym, chg, w, contrib, side))
        if side == "up":
            advances += 1
            weight_up += w
        elif side == "down":
            declines += 1
            weight_down += w
        else:
            unchanged += 1
            weight_flat += w

    contrib_r = round(contribution, 3)
    pcr_lean = (pcr or {}).get("lean") if isinstance(pcr, dict) else None
    pcr_oi = (pcr or {}).get("oiPcr") if isinstance(pcr, dict) else None
    weight_gap = weight_down - weight_up  # + = more declining weight
    # "Whipsaw": heavy names lean one way, but weight×move (and/or PCR) keeps index flat.
    whipsaw_down = weight_gap >= 12 and abs(contrib_r) < _LEAN_PCT
    whipsaw_up = (weight_up - weight_down) >= 12 and abs(contrib_r) < _LEAN_PCT
    pcr_supports_up = pcr_lean == "bullish" or (
        isinstance(pcr_oi, (int, float)) and pcr_oi > 1.0
    )
    pcr_supports_down = pcr_lean == "bearish" or (
        isinstance(pcr_oi, (int, float)) and pcr_oi < 0.8
    )

    if whipsaw_down:
        lean, action = "mixed", "watch"
        if pcr_supports_up:
            label = "Heavy weights soft, but lift + PCR put tilt — whipsaw"
        else:
            label = "More declining weight, but movers offset — whipsaw"
    elif whipsaw_up:
        lean, action = "mixed", "watch"
        if pcr_supports_down:
            label = "Heavy weights firm, but drag + PCR call tilt — whipsaw"
        else:
            label = "More advancing weight, but drags offset — whipsaw"
    elif contrib_r >= _LEAN_PCT and weight_up >= weight_down:
        lean, action = "bullish", "buy long"
        label = "Weight-weighted breadth leans up"
    elif contrib_r <= -_LEAN_PCT and weight_down >= weight_up:
        lean, action = "bearish", "buy short"
        label = "Weight-weighted breadth leans down"
    elif advances > declines and weight_up > weight_down:
        lean, action = "bullish", "watch"
        label = "More advances by weight — soft up lean"
    elif declines > advances and weight_down > weight_up:
        lean, action = "bearish", "watch"
        label = "More declines by weight — soft down lean"
    else:
        lean, action = "mixed", "watch"
        label = "Breadth mixed — no clear Nifty lean"

    top_up = [
        {"symbol": s, "changePct": round(c, 2), "weight": round(w, 2), "contributionPct": round(x, 3)}
        for s, c, w, x, _side_name in sorted(
            [m for m in movers if m[4] == "up"], key=lambda m: -m[2]
        )[:3]
    ]
    top_down = [
        {"symbol": s, "changePct": round(c, 2), "weight": round(w, 2), "contributionPct": round(x, 3)}
        for s, c, w, x, _side_name in sorted(
            [m for m in movers if m[4] == "down"], key=lambda m: -m[2]
        )[:3]
    ]

    ups = sorted([m for m in movers if m[4] == "up"], key=lambda m: -m[2])
    flats = sorted([m for m in movers if m[4] == "flat"], key=lambda m: -m[2])
    downs = sorted([m for m in movers if m[4] == "down"], key=lambda m: -m[2])
    segments = [
        {
            "symbol": s,
            "changePct": round(c, 2),
            "weight": round(w, 2),
            "contributionPct": round(x, 3),
            "side": side,
        }
        for s, c, w, x, side in [*ups, *flats, *downs]
    ]

    # Decline-weight trend: arrow up = more weight declining (pressure rising).
    w_down_r = round(weight_down, 1)
    w_up_r = round(weight_up, 1)
    record_trend("nifty_weight_down", w_down_r)
    record_trend("nifty_weight_up", w_up_r)
    weight_trend = {
        "down": trend_pair("nifty_weight_down", current=w_down_r, eps=_WEIGHT_TREND_EPS),
        "up": trend_pair("nifty_weight_up", current=w_up_r, eps=_WEIGHT_TREND_EPS),
    }

    result = {
        "ready": True,
        "advances": advances,
        "declines": declines,
        "unchanged": unchanged,
        "quoted": len(quotes),
        "universe": len(symbols),
        "weightUp": w_up_r,
        "weightDown": w_down_r,
        "weightFlat": round(weight_flat, 1),
        "contributionPct": contrib_r,
        "lean": lean,
        "action": action,
        "label": label,
        "topUp": top_up,
        "topDown": top_down,
        "segments": segments,
        "pcr": pcr,
        "weightsMeta": weights_meta,
        "quoteSource": quote_source,
        "weightTrend": weight_trend,
        "cached": False,
    }
    _BREADTH_CACHE["ts"] = time.time()
    _BREADTH_CACHE["payload"] = result
    return result
