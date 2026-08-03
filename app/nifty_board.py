"""Nifty 50 dedicated board payload for /nifty page."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from .fyers_auth import get_access_token, fyers_status
from .fyers_quotes import ensure_fyers_poller, fetch_fyers_quotes
from .nifty_breadth import build_nifty_breadth
from .nifty_breadth_history import breadth_history, record_breadth_snapshot
from .nifty_chain import atm_wing_board, enrich_oi_plot, fetch_nifty_option_chain, oi_insights
from .nifty_futures import combine_sync_insight, fetch_nifty_futures
from .nifty_oi_ai import maybe_ai_oi_insight
from .nifty_pcr import _bias, fetch_nifty_pcr
from .nifty_pcr_history import pcr_history, record_pcr_snapshot
from .nifty_stock_fut import build_market_sync_card, build_stock_fut_basket
from .nifty_weights import get_nifty_weights
from .quotes import get_quote
from .session import is_live_data_window, live_data_window_label
from .universe import UNIVERSE

_BOARD_CACHE: dict[str, Any] = {
    "ts": 0.0,
    "payload": None,
    "prevWing": None,
    "prevSpot": None,
    "building": False,
    "lastError": None,
}
_BOARD_LOCK = threading.Lock()
_BOARD_DONE = threading.Event()
_BOARD_TTL_S = 30.0
_BOARD_LIVE_TTL_S = 1.0  # Fyers breadth/spot slice when poller is running
_BOARD_TTL_CLOSED_S = 15 * 60.0  # after close, reuse last board — no live churn


def _empty_nifty_board(*, error: str | None = None) -> dict[str, Any]:
    """Paint-able shell while the first Yahoo/Fyers/Gemini build runs on Render."""
    in_hours = is_live_data_window()
    msg = error or "Market board warming up — first load on free hosting can take 1–2 min."
    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "fyersConnected": bool(fyers_status(verify=False).get("connected")),
        "marketHours": in_hours,
        "marketHoursLabel": live_data_window_label(),
        "liveDataPaused": not in_hours,
        "index": {"ready": False, "ltp": None, "changePct": None, "source": None},
        "breadth": {"ready": False, "segments": []},
        "breadthHistory": breadth_history(5),
        "drivers": {"topUp": [], "topDown": []},
        "leadLag": {"ready": False, "label": "Waiting for quotes", "verdict": "—"},
        "futures": {"ready": False},
        "stockFutBasket": {"ready": False},
        "marketSync": {"ready": False},
        "pcr": {"ready": False},
        "pcrHistory": pcr_history(10),
        "optionOi": {"ready": False, "plot": [], "rows": []},
        "oiInsight": {"headline": msg, "bullets": [], "source": "system"},
        "insights": [msg],
        "paperTrades": None,
        "cached": False,
        "stale": True,
        "building": True,
        "error": error,
    }


def _store_nifty_board(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["cached"] = False
    payload["stale"] = False
    payload["building"] = False
    payload.pop("error", None)
    _BOARD_CACHE["ts"] = time.time()
    _BOARD_CACHE["heavy_ts"] = time.time()
    _BOARD_CACHE["payload"] = payload
    _BOARD_CACHE["lastError"] = None
    _BOARD_DONE.set()
    return payload


def _rebuild_nifty_board_bg(*, force: bool = False) -> None:
    with _BOARD_LOCK:
        if _BOARD_CACHE["building"]:
            return
        _BOARD_CACHE["building"] = True
        _BOARD_DONE.clear()
    try:
        payload = _build_nifty_board_full(force=force)
        _store_nifty_board(payload)
    except Exception as exc:  # noqa: BLE001
        _BOARD_CACHE["lastError"] = f"{type(exc).__name__}: {exc}"
        _BOARD_DONE.set()
    finally:
        with _BOARD_LOCK:
            _BOARD_CACHE["building"] = False
        _BOARD_DONE.set()


def _kick_rebuild(*, force: bool = False) -> None:
    threading.Thread(
        target=_rebuild_nifty_board_bg,
        kwargs={"force": force},
        daemon=True,
        name="saint-nifty-rebuild",
    ).start()


def _refresh_live_slice(payload: dict[str, Any]) -> dict[str, Any]:
    """Fast path: refresh Fyers-driven breadth/spot/sync without OI/AI/chain."""
    weight_pack = get_nifty_weights()
    symbols = [s for s in (weight_pack.get("weights") or {}) if s in UNIVERSE]
    if get_access_token() and symbols:
        ensure_fyers_poller(symbols)

    breadth = build_nifty_breadth(force=True) or {}
    index = _index_quote()
    spot = payload.get("optionOi", {}).get("spot") or index.get("ltp")
    if spot is None:
        spot = payload.get("index", {}).get("ltp")

    lead = _lead_lag(breadth, index, spot=float(spot) if spot is not None else None)
    futures = payload.get("futures") if isinstance(payload.get("futures"), dict) else {}
    stock_fut = (
        payload.get("stockFutBasket")
        if isinstance(payload.get("stockFutBasket"), dict)
        else {}
    )
    market_sync = build_market_sync_card(
        cash_basket_pct=lead.get("basketMovePct"),
        nifty_spot_pct=lead.get("indexMovePct"),
        stock_fut_basket_pct=stock_fut.get("basketMovePct") if stock_fut.get("ready") else None,
        nifty_fut_pct=futures.get("changePct") if futures.get("ready") else None,
        spot=float(spot) if spot is not None else None,
        nifty_fut_ltp=float(futures["ltp"]) if futures.get("ltp") is not None else None,
        stock_fut_meta=stock_fut,
    )
    sync_insight = combine_sync_insight(
        index_vs_basket_pp=lead.get("indexVsBasketPp"),
        cash_stance=str(lead.get("stance") or "unclear"),
        futures=futures if futures.get("ready") else None,
    )
    lead["syncInsight"] = market_sync.get("insight") or sync_insight
    lead["marketSync"] = market_sync

    segments = list(breadth.get("segments") or [])
    ups = sorted(
        [s for s in segments if s.get("side") == "up"],
        key=lambda x: -float(x.get("weight") or 0),
    )
    downs = sorted(
        [s for s in segments if s.get("side") == "down"],
        key=lambda x: -float(x.get("weight") or 0),
    )

    out = dict(payload)
    out["asOf"] = datetime.now(timezone.utc).isoformat()
    out["fyersConnected"] = bool(fyers_status(verify=False).get("connected"))
    out["index"] = index
    out["breadth"] = {
        "ready": bool(breadth.get("ready")),
        "advances": breadth.get("advances"),
        "declines": breadth.get("declines"),
        "unchanged": breadth.get("unchanged"),
        "quoted": breadth.get("quoted"),
        "universe": breadth.get("universe"),
        "weightUp": breadth.get("weightUp"),
        "weightDown": breadth.get("weightDown"),
        "weightFlat": breadth.get("weightFlat"),
        "contributionPct": breadth.get("contributionPct"),
        "lean": breadth.get("lean"),
        "action": breadth.get("action"),
        "label": breadth.get("label"),
        "quoteSource": breadth.get("quoteSource")
        or (breadth.get("weightsMeta") or {}).get("quoteSource"),
        "segments": segments,
        "weightTrend": breadth.get("weightTrend"),
    }
    out["drivers"] = {"topUp": ups[:10], "topDown": downs[:10]}
    out["leadLag"] = lead
    out["marketSync"] = market_sync
    out["liveRefresh"] = True
    out["cached"] = True
    out["stale"] = False
    _BOARD_CACHE["payload"] = out
    _BOARD_CACHE["ts"] = time.time()
    return out


def get_nifty_board(*, force: bool = False) -> dict[str, Any]:
    """Serve cached board instantly; never block HTTP on a cold Yahoo/Fyers rebuild."""
    now = time.time()
    in_hours = is_live_data_window()
    ttl = _BOARD_TTL_S if in_hours else _BOARD_TTL_CLOSED_S
    cached = _BOARD_CACHE.get("payload")
    age = now - float(_BOARD_CACHE["ts"]) if cached is not None else None

    if cached is not None:
        fyers_live = bool(get_access_token()) and in_hours
        heavy_ts = float(_BOARD_CACHE.get("heavy_ts") or _BOARD_CACHE["ts"] or now)
        heavy_age = now - heavy_ts

        if fyers_live and age is not None and age >= _BOARD_LIVE_TTL_S and not force:
            out = _refresh_live_slice(dict(cached))
            out["cacheAgeS"] = 0.0
            out["building"] = bool(_BOARD_CACHE.get("building"))
            out["marketHours"] = in_hours
            out["marketHoursLabel"] = live_data_window_label()
            out["liveDataPaused"] = False
            if heavy_age >= _BOARD_TTL_S:
                _kick_rebuild(force=False)
            return out

        out = dict(cached)
        out["cached"] = True
        out["cacheAgeS"] = round(float(age or 0), 1)
        stale = bool(age is not None and age >= ttl)
        out["stale"] = stale or force
        out["building"] = bool(_BOARD_CACHE.get("building"))
        out["marketHours"] = in_hours
        out["marketHoursLabel"] = live_data_window_label()
        out["liveDataPaused"] = not in_hours
        if out.get("error"):
            out["error"] = _BOARD_CACHE.get("lastError")
        if force or stale:
            _kick_rebuild(force=force)
        return out

    # After close: reuse last snapshot if we have one.
    if not in_hours and cached is not None and not force:
        out = dict(cached)
        out["cached"] = True
        out["marketHours"] = False
        out["marketHoursLabel"] = live_data_window_label()
        out["liveDataPaused"] = True
        out["building"] = False
        out["stale"] = False
        return out

    _kick_rebuild(force=force)
    empty = _empty_nifty_board(error=_BOARD_CACHE.get("lastError"))
    empty["building"] = bool(_BOARD_CACHE.get("building"))
    return empty


def _index_quote() -> dict[str, Any]:
    q = get_quote("NIFTY")
    if not q:
        return {"ready": False, "ltp": None, "changePct": None, "source": None}
    return {
        "ready": True,
        "key": "NIFTY",
        "name": "NIFTY 50",
        "ltp": q.ltp,
        "change": q.change,
        "changePct": q.change_pct,
        "previousClose": q.previous_close,
        "volume": q.volume,
        "source": q.source,
    }


def _lead_lag(
    breadth: dict[str, Any],
    index: dict[str, Any],
    *,
    spot: float | None = None,
) -> dict[str, Any]:
    """Cash sync: basket (weight×stock day%) is baseline; is printed Nifty ahead or lagging?"""
    # Sync band: |index − basket| < 0.08pp → in sync; beyond → ahead / lagging.
    sync_band_pp = 0.08
    basket = breadth.get("contributionPct")
    idx = index.get("changePct")
    if basket is None or idx is None:
        return {
            "ready": False,
            "baseline": "basket",
            "basketMovePct": basket,
            "indexMovePct": idx,
            "indexVsBasketPp": None,
            "indexVsBasketPts": None,
            "syncBandPp": sync_band_pp,
            "diffPp": None,
            "stance": "unclear",
            "label": "Waiting for quotes",
            "verdict": "—",
            "howToRead": (
                "Basket = Σ(weight × stock % from prev close). That is the baseline. "
                f"In sync if |Nifty − basket| < {sync_band_pp:.2f}pp; else ahead/lagging."
            ),
            "note": None,
        }
    basket_f = float(basket)
    idx_f = float(idx)
    # Positive => printed Nifty ahead of cash basket
    index_vs_basket = idx_f - basket_f
    pts = None
    if spot is not None and float(spot) > 0:
        pts = round(float(spot) * index_vs_basket / 100.0, 1)

    if index_vs_basket >= sync_band_pp:
        stance, verdict = "nifty_ahead", "Nifty ahead"
        label = f"Nifty running ahead of basket by {index_vs_basket:.3f}pp"
        if pts is not None:
            label += f" (~{pts:+.1f} pts)"
    elif index_vs_basket <= -sync_band_pp:
        stance, verdict = "nifty_lagging", "Nifty lagging"
        label = f"Nifty lagging basket by {abs(index_vs_basket):.3f}pp"
        if pts is not None:
            label += f" (~{pts:+.1f} pts)"
    else:
        stance, verdict = "in_sync", "In sync"
        label = "Nifty and cash basket roughly in sync"
        if pts is not None:
            label += f" ({index_vs_basket:+.3f}pp · ~{pts:+.1f} pts)"

    return {
        "ready": True,
        "baseline": "basket",
        "basketMovePct": round(basket_f, 3),
        "indexMovePct": round(idx_f, 3),
        "indexVsBasketPp": round(index_vs_basket, 3),
        "indexVsBasketPts": pts,
        "syncBandPp": sync_band_pp,
        # Keep old key for any leftover clients (basket − index)
        "diffPp": round(basket_f - idx_f, 3),
        "stance": stance,
        "verdict": verdict,
        "label": label,
        "howToRead": (
            "Both % are vs previous close. Basket is the weight-rebuilt cash baseline. "
            f"In sync if gap < ±{sync_band_pp:.2f}pp "
            f"(~±{round(float(spot) * sync_band_pp / 100.0, 0) if spot else 20:.0f} index pts at current spot); "
            "beyond that = Nifty ahead (green) or lagging (red)."
        ),
        "note": (
            "Not classic futures arb. Futures basis (below) answers premium/discount to spot."
        ),
    }


def build_nifty_board(*, force: bool = False) -> dict[str, Any]:
    """Backward-compatible alias — prefer get_nifty_board for HTTP handlers."""
    return get_nifty_board(force=force)


def _build_nifty_board_full(*, force: bool = False) -> dict[str, Any]:
    in_hours = is_live_data_window()

    weight_pack = get_nifty_weights()
    symbols = [s for s in (weight_pack.get("weights") or {}) if s in UNIVERSE]
    # Fyers quotes only during live window (fetch_fyers_quotes also guards).
    if in_hours and get_access_token() and symbols:
        fetch_fyers_quotes(symbols)

    breadth = build_nifty_breadth(force=force) or {}
    index = _index_quote()
    chain_raw = fetch_nifty_option_chain(force=force) or {}
    if chain_raw.get("spot") and not index.get("ltp"):
        index = {
            "ready": True,
            "key": "NIFTY",
            "name": "NIFTY 50",
            "ltp": chain_raw["spot"],
            "change": None,
            "changePct": None,
            "previousClose": None,
            "volume": None,
            "source": chain_raw.get("spotSource") or chain_raw.get("source"),
        }

    wing = atm_wing_board(chain_raw, wing=15) if chain_raw else {"ready": False, "rows": [], "plot": []}
    spot = chain_raw.get("spot") or index.get("ltp")
    plot = enrich_oi_plot(list(wing.get("plot") or []))

    pcr = fetch_nifty_pcr(force=force) or {}
    if (not pcr.get("ready")) and chain_raw.get("oiPcr"):
        oi_pcr = float(chain_raw["oiPcr"])
        lean, label = _bias(oi_pcr)
        pcr = {
            "ready": True,
            "oiPcr": oi_pcr,
            "volumePcr": None,
            "putOi": int(chain_raw.get("putOi") or 0),
            "callOi": int(chain_raw.get("callOi") or 0),
            "lean": lean,
            "label": label,
            "expiry": chain_raw.get("expiry"),
            "asOf": chain_raw.get("asOf"),
            "source": chain_raw.get("source"),
        }

    prev = _BOARD_CACHE.get("prevWing")
    prev_spot = _BOARD_CACHE.get("prevSpot")
    insight_pack = oi_insights(
        {"wingBoard": prev, "spot": prev_spot} if prev else None,
        {"wingBoard": wing},
        float(spot) if spot is not None else None,
    )

    ai_packet = {
        "spot": spot,
        "atmStrike": wing.get("atmStrike"),
        "ceOiWing": wing.get("ceOiWing"),
        "peOiWing": wing.get("peOiWing"),
        "ceOiChgWing": wing.get("ceOiChgWing"),
        "peOiChgWing": wing.get("peOiChgWing"),
        "oiPcr": pcr.get("oiPcr"),
        "ruleHeadline": insight_pack.get("headline"),
        "ruleSentiment": insight_pack.get("sentiment"),
        "metrics": insight_pack.get("metrics"),
    }
    ai_insight = maybe_ai_oi_insight(ai_packet, force=force) if in_hours else (
        maybe_ai_oi_insight(ai_packet, force=False)  # returns cache only off-hours
    )

    headline = (ai_insight or {}).get("insight") or insight_pack.get("headline")
    sentiment = (ai_insight or {}).get("sentiment") or insight_pack.get("sentiment")
    insight_source = (ai_insight or {}).get("source") or "rules"

    # Flat list for older UI fields + structured pack
    insights_list = [headline] + list(insight_pack.get("bullets") or [])
    insights_list = [x for x in insights_list if x][:5]

    hist = record_pcr_snapshot(
        {
            "oiPcr": pcr.get("oiPcr"),
            "volumePcr": pcr.get("volumePcr"),
            "putOi": pcr.get("putOi"),
            "callOi": pcr.get("callOi"),
            "spot": spot,
            "lean": pcr.get("lean"),
            "ceOiWing": wing.get("ceOiWing"),
            "peOiWing": wing.get("peOiWing"),
            "ceOiChgWing": wing.get("ceOiChgWing"),
            "peOiChgWing": wing.get("peOiChgWing"),
            "insight": headline,
        }
    )

    br_hist = record_breadth_snapshot(
        {
            "weightUp": breadth.get("weightUp"),
            "weightDown": breadth.get("weightDown"),
            "weightFlat": breadth.get("weightFlat"),
            "contributionPct": breadth.get("contributionPct"),
            "advances": breadth.get("advances"),
            "declines": breadth.get("declines"),
            "lean": breadth.get("lean"),
        }
    )

    lead = _lead_lag(breadth, index, spot=float(spot) if spot is not None else None)
    futures = fetch_nifty_futures(spot=float(spot) if spot is not None else None, force=force)

    weights = (weight_pack.get("weights") or {}) if isinstance(weight_pack, dict) else {}
    # Prefer weights already used in breadth path
    if not weights:
        weights = get_nifty_weights().get("weights") or {}
    stock_fut = build_stock_fut_basket(weights, force=force)

    market_sync = build_market_sync_card(
        cash_basket_pct=lead.get("basketMovePct"),
        nifty_spot_pct=lead.get("indexMovePct"),
        stock_fut_basket_pct=stock_fut.get("basketMovePct") if stock_fut.get("ready") else None,
        nifty_fut_pct=futures.get("changePct") if futures.get("ready") else None,
        spot=float(spot) if spot is not None else None,
        nifty_fut_ltp=float(futures["ltp"]) if futures.get("ltp") is not None else None,
        stock_fut_meta=stock_fut,
    )

    sync_insight = combine_sync_insight(
        index_vs_basket_pp=lead.get("indexVsBasketPp"),
        cash_stance=str(lead.get("stance") or "unclear"),
        futures=futures if futures.get("ready") else None,
    )
    lead["syncInsight"] = market_sync.get("insight") or sync_insight
    lead["marketSync"] = market_sync

    # Paper-trade engine: breadth + sync-cross strategies vs Fyers ATM CE (best-effort).
    paper_tick: dict[str, Any] | None = None
    try:
        from .nifty_paper_trades import tick_paper_trades

        cross_pp = None
        try:
            raw_cross = (market_sync.get("cross") or {}).get("diffPp")
            if raw_cross is not None:
                cross_pp = float(raw_cross)
        except (TypeError, ValueError):
            cross_pp = None
        paper_tick = tick_paper_trades(force=False, cross_diff_pp=cross_pp)
    except Exception:  # noqa: BLE001
        paper_tick = None

    segments = list(breadth.get("segments") or [])
    ups = sorted(
        [s for s in segments if s.get("side") == "up"],
        key=lambda x: -float(x.get("weight") or 0),
    )
    downs = sorted(
        [s for s in segments if s.get("side") == "down"],
        key=lambda x: -float(x.get("weight") or 0),
    )

    payload = {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "fyersConnected": bool(fyers_status(verify=False).get("connected")),
        "marketHours": in_hours,
        "marketHoursLabel": live_data_window_label(),
        "liveDataPaused": not in_hours,
        "index": index,
        "breadth": {
            "ready": bool(breadth.get("ready")),
            "advances": breadth.get("advances"),
            "declines": breadth.get("declines"),
            "unchanged": breadth.get("unchanged"),
            "quoted": breadth.get("quoted"),
            "universe": breadth.get("universe"),
            "weightUp": breadth.get("weightUp"),
            "weightDown": breadth.get("weightDown"),
            "weightFlat": breadth.get("weightFlat"),
            "contributionPct": breadth.get("contributionPct"),
            "lean": breadth.get("lean"),
            "action": breadth.get("action"),
            "label": breadth.get("label"),
            "quoteSource": breadth.get("quoteSource")
            or (breadth.get("weightsMeta") or {}).get("quoteSource"),
            "segments": segments,
            "weightTrend": breadth.get("weightTrend"),
        },
        "breadthHistory": br_hist or breadth_history(5),
        "drivers": {
            "topUp": ups[:10],
            "topDown": downs[:10],
        },
        "leadLag": lead,
        "futures": futures,
        "stockFutBasket": stock_fut,
        "marketSync": market_sync,
        "pcr": pcr,
        "pcrHistory": hist or pcr_history(10),
        "optionOi": {
            "ready": bool(wing.get("ready")),
            "source": chain_raw.get("source"),
            "expiry": chain_raw.get("expiry"),
            "spot": spot,
            "atmStrike": wing.get("atmStrike"),
            "ceOiWing": wing.get("ceOiWing"),
            "peOiWing": wing.get("peOiWing"),
            "ceOiChgWing": wing.get("ceOiChgWing"),
            "peOiChgWing": wing.get("peOiChgWing"),
            "plot": plot,
            "rows": wing.get("rows") or [],
        },
        "oiInsight": {
            "headline": headline,
            "sentiment": sentiment,
            "bullets": insight_pack.get("bullets") or [],
            "source": insight_source,
            "metrics": insight_pack.get("metrics") or {},
        },
        "insights": insights_list,
        "paperTrades": paper_tick,
        "cached": False,
    }

    _BOARD_CACHE["prevWing"] = wing
    _BOARD_CACHE["prevSpot"] = spot
    return payload
