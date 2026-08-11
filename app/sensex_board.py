"""Sensex 30 board — index, breadth, drivers, basket lead/lag (no Nifty OI/PCR)."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from .fyers_auth import fyers_status, get_access_token
from .fyers_quotes import ensure_fyers_poller
from .market_minute_store import record_index_quote
from .nifty_breadth import build_sensex_breadth
from .nifty_breadth_history import breadth_history, record_breadth_snapshot
from .nifty_chain import atm_wing_board, enrich_oi_plot
from .nifty_pcr import _bias
from .nifty_pcr_history import pcr_history, record_pcr_snapshot
from .quotes import get_quote
from .sensex_weights import get_sensex_weights
from .session import is_cash_session_open, is_live_data_window, live_data_window_label
from .universe import UNIVERSE

_BOARD_CACHE: dict[str, Any] = {
    "ts": 0.0,
    "payload": None,
    "building": False,
    "lastError": None,
}
_BOARD_LOCK = threading.Lock()
_BOARD_DONE = threading.Event()
_BOARD_TTL_S = 30.0
_BOARD_LIVE_TTL_S = 1.0
_BOARD_TTL_CLOSED_S = 15 * 60.0


def _sanitize_interval(interval: int | None) -> int:
    return 1 if int(interval or 5) == 1 else 5


def _sanitize_rows(rows: int | None) -> int:
    return max(3, min(int(rows or 5), 500))


def _fetch_sensex_option_chain(*, force: bool = False) -> dict[str, Any] | None:
    """Best-effort Sensex option chain from NiftyTrader's generic endpoint."""
    import requests

    url = "https://webapi.niftytrader.in/webapi/option/option-chain-data?symbol=sensex&expiryDate="
    try:
        r = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Saint/1.0)",
                "Accept": "application/json",
            },
        )
        r.raise_for_status()
        body = r.json()
        rd = body.get("resultData") or {}
        rows = rd.get("opDatas") or []
        if not rows:
            return None
        spot = None
        for row in rows:
            if row.get("index_close"):
                spot = float(row["index_close"])
                break
        strikes: list[dict[str, Any]] = []
        for row in rows:
            try:
                strike = float(row.get("strike_price") or 0)
            except (TypeError, ValueError):
                continue
            if strike <= 0:
                continue
            strikes.append(
                {
                    "strike": strike,
                    "ceOi": int(float(row.get("calls_oi") or 0)),
                    "peOi": int(float(row.get("puts_oi") or 0)),
                    "ceOiChg": int(float(row.get("calls_change_oi") or 0)),
                    "peOiChg": int(float(row.get("puts_change_oi") or 0)),
                    "ceVol": int(float(row.get("calls_volume") or 0)),
                    "peVol": int(float(row.get("puts_volume") or 0)),
                }
            )
        if not strikes:
            return None
        ce_oi = sum(s["ceOi"] for s in strikes)
        pe_oi = sum(s["peOi"] for s in strikes)
        return {
            "source": "niftytrader",
            "spot": spot,
            "expiry": (rows[0].get("expiry_date") or "")[:10] or None,
            "asOf": rows[0].get("time") or rows[0].get("created_at"),
            "strikes": strikes,
            "callOi": ce_oi,
            "putOi": pe_oi,
            "oiPcr": round(pe_oi / ce_oi, 3) if ce_oi > 0 else None,
        }
    except Exception:  # noqa: BLE001
        return None


def _empty_sensex_board(*, error: str | None = None) -> dict[str, Any]:
    in_hours = is_live_data_window()
    msg = error or "Sensex board warming up — first load can take a minute."
    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "indexKey": "SENSEX",
        "fyersConnected": bool(fyers_status(verify=False).get("connected")),
        "marketHours": in_hours,
        "marketHoursLabel": live_data_window_label(),
        "liveDataPaused": not in_hours,
        "index": {"ready": False, "key": "SENSEX", "ltp": None, "changePct": None, "source": None},
        "breadth": {"ready": False, "segments": []},
        "breadthHistory": breadth_history(5, index_key="sensex", interval_minutes=5),
        "drivers": {"topUp": [], "topDown": []},
        "leadLag": {"ready": False, "label": "Waiting for quotes", "verdict": "—"},
        "pcr": {"ready": False},
        "pcrHistory": pcr_history(10, index_key="sensex"),
        "optionOi": {"ready": False, "plot": [], "rows": []},
        "oiInsight": {"headline": msg, "bullets": [], "source": "system"},
        "insights": [msg],
        "cached": False,
        "stale": True,
        "building": True,
        "error": error or msg,
    }


def _index_quote() -> dict[str, Any]:
    q = get_quote("SENSEX")
    if not q:
        return {"ready": False, "key": "SENSEX", "ltp": None, "changePct": None, "source": None}
    try:
        record_index_quote("sensex", q)
    except Exception:  # noqa: BLE001
        pass
    return {
        "ready": True,
        "key": "SENSEX",
        "name": "SENSEX",
        "ltp": q.ltp,
        "change": q.change,
        "changePct": q.change_pct,
        "previousClose": q.previous_close,
        "volume": q.volume,
        "source": q.source,
    }


def _lead_lag(breadth: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    sync_band_pp = 0.08
    basket = breadth.get("contributionPct")
    idx = index.get("changePct")
    spot = index.get("ltp")
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
        }
    basket_f = float(basket)
    idx_f = float(idx)
    index_vs_basket = idx_f - basket_f
    pts = None
    if spot is not None and float(spot) > 0:
        pts = round(float(spot) * index_vs_basket / 100.0, 1)

    if index_vs_basket >= sync_band_pp:
        stance, verdict = "index_ahead", "Sensex ahead"
        label = f"Sensex running ahead of basket by {index_vs_basket:.3f}pp"
    elif index_vs_basket <= -sync_band_pp:
        stance, verdict = "index_lagging", "Sensex lagging"
        label = f"Sensex lagging basket by {abs(index_vs_basket):.3f}pp"
    else:
        stance, verdict = "in_sync", "In sync"
        label = "Sensex and cash basket roughly in sync"

    if pts is not None:
        label += f" (~{pts:+.1f} pts)"

    return {
        "ready": True,
        "baseline": "basket",
        "basketMovePct": round(basket_f, 3),
        "indexMovePct": round(idx_f, 3),
        "indexVsBasketPp": round(index_vs_basket, 3),
        "indexVsBasketPts": pts,
        "syncBandPp": sync_band_pp,
        "diffPp": round(basket_f - idx_f, 3),
        "stance": stance,
        "verdict": verdict,
        "label": label,
    }


def _build_full(*, force: bool = False, breadth_interval: int = 5, breadth_rows: int = 5) -> dict[str, Any]:
    weight_pack = get_sensex_weights(force=force)
    symbols = [s for s in (weight_pack.get("weights") or {}) if s in UNIVERSE]
    if get_access_token() and symbols:
        ensure_fyers_poller(symbols)

    breadth = build_sensex_breadth(force=force) or {}
    index = _index_quote()
    lead = _lead_lag(breadth, index)
    if is_cash_session_open():
        br_hist = record_breadth_snapshot(
            {
                "weightUp": breadth.get("weightUp"),
                "weightDown": breadth.get("weightDown"),
                "weightFlat": breadth.get("weightFlat"),
                "contributionPct": breadth.get("contributionPct"),
                "advances": breadth.get("advances"),
                "declines": breadth.get("declines"),
                "lean": breadth.get("lean"),
            },
            index_key="sensex",
            limit=breadth_rows,
            interval_minutes=breadth_interval,
        )
    else:
        br_hist = breadth_history(
            breadth_rows,
            index_key="sensex",
            interval_minutes=breadth_interval,
        )

    chain = _fetch_sensex_option_chain(force=force) or {}
    wing = atm_wing_board(chain, wing=15) if chain else {"ready": False, "rows": [], "plot": []}
    plot = enrich_oi_plot(list(wing.get("plot") or [])) if wing.get("ready") else []
    oi_pcr = chain.get("oiPcr")
    pcr_lean, pcr_label = _bias(float(oi_pcr)) if isinstance(oi_pcr, (int, float)) else ("unclear", "PCR unavailable")
    pcr = {
        "ready": bool(isinstance(oi_pcr, (int, float))),
        "oiPcr": float(oi_pcr) if isinstance(oi_pcr, (int, float)) else None,
        "volumePcr": None,
        "putOi": chain.get("putOi"),
        "callOi": chain.get("callOi"),
        "lean": pcr_lean,
        "label": pcr_label,
        "expiry": chain.get("expiry"),
        "source": chain.get("source"),
    }
    hist = record_pcr_snapshot(
        {
            "oiPcr": pcr.get("oiPcr"),
            "volumePcr": pcr.get("volumePcr"),
            "putOi": pcr.get("putOi"),
            "callOi": pcr.get("callOi"),
            "spot": chain.get("spot") or index.get("ltp"),
            "lean": pcr.get("lean"),
            "ceOiWing": wing.get("ceOiWing"),
            "peOiWing": wing.get("peOiWing"),
            "ceOiChgWing": wing.get("ceOiChgWing"),
            "peOiChgWing": wing.get("peOiChgWing"),
            "insight": "Sensex OI snapshot",
        },
        index_key="sensex",
        limit=10,
    )

    segments = list(breadth.get("segments") or [])
    ups = sorted(
        [s for s in segments if s.get("side") == "up"],
        key=lambda x: -float(x.get("weight") or 0),
    )
    downs = sorted(
        [s for s in segments if s.get("side") == "down"],
        key=lambda x: -float(x.get("weight") or 0),
    )

    in_hours = is_live_data_window()
    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "indexKey": "SENSEX",
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
            "quoteSource": breadth.get("quoteSource"),
            "segments": segments,
            "weightTrend": breadth.get("weightTrend"),
        },
        "breadthHistory": br_hist
        or breadth_history(
            breadth_rows,
            index_key="sensex",
            interval_minutes=breadth_interval,
        ),
        "drivers": {"topUp": ups[:10], "topDown": downs[:10]},
        "leadLag": lead,
        "pcr": pcr,
        "pcrHistory": hist or pcr_history(10, index_key="sensex"),
        "optionOi": {
            "ready": bool(wing.get("ready")),
            "source": chain.get("source"),
            "expiry": chain.get("expiry"),
            "spot": chain.get("spot") or index.get("ltp"),
            "atmStrike": wing.get("atmStrike"),
            "ceOiWing": wing.get("ceOiWing"),
            "peOiWing": wing.get("peOiWing"),
            "ceOiChgWing": wing.get("ceOiChgWing"),
            "peOiChgWing": wing.get("peOiChgWing"),
            "plot": plot,
            "rows": wing.get("rows") or [],
        },
        "oiInsight": {
            "headline": "Sensex OI/PCR ready" if pcr.get("ready") else "Waiting for Sensex option chain…",
            "sentiment": pcr.get("lean"),
            "bullets": [],
            "source": chain.get("source") or "system",
            "metrics": {},
        },
        "insights": [
            "Sensex OI/PCR from option-chain snapshot" if pcr.get("ready") else "Sensex OI/PCR building…"
        ],
        "cached": False,
        "stale": False,
        "building": False,
    }


def _refresh_live_slice(
    payload: dict[str, Any],
    *,
    breadth_interval: int = 5,
    breadth_rows: int = 5,
) -> dict[str, Any]:
    weight_pack = get_sensex_weights()
    symbols = [s for s in (weight_pack.get("weights") or {}) if s in UNIVERSE]
    if get_access_token() and symbols:
        ensure_fyers_poller(symbols)

    breadth = build_sensex_breadth(force=True) or {}
    index = _index_quote()
    lead = _lead_lag(breadth, index)
    if is_cash_session_open():
        br_hist = record_breadth_snapshot(
            {
                "weightUp": breadth.get("weightUp"),
                "weightDown": breadth.get("weightDown"),
                "weightFlat": breadth.get("weightFlat"),
                "contributionPct": breadth.get("contributionPct"),
                "advances": breadth.get("advances"),
                "declines": breadth.get("declines"),
                "lean": breadth.get("lean"),
            },
            index_key="sensex",
            limit=breadth_rows,
            interval_minutes=breadth_interval,
        )
    else:
        br_hist = breadth_history(
            breadth_rows,
            index_key="sensex",
            interval_minutes=breadth_interval,
        )

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
        "quoteSource": breadth.get("quoteSource"),
        "segments": segments,
        "weightTrend": breadth.get("weightTrend"),
    }
    out["breadthHistory"] = br_hist or breadth_history(
        breadth_rows,
        index_key="sensex",
        interval_minutes=breadth_interval,
    )
    out["drivers"] = {"topUp": ups[:10], "topDown": downs[:10]}
    out["leadLag"] = lead
    out["liveRefresh"] = True
    out["cached"] = True
    out["stale"] = False
    _BOARD_CACHE["payload"] = out
    _BOARD_CACHE["ts"] = time.time()
    return out


def _rebuild_bg(*, force: bool = False, breadth_interval: int = 5, breadth_rows: int = 5) -> None:
    with _BOARD_LOCK:
        if _BOARD_CACHE["building"]:
            return
        _BOARD_CACHE["building"] = True
        _BOARD_DONE.clear()
    try:
        payload = _build_full(
            force=force,
            breadth_interval=breadth_interval,
            breadth_rows=breadth_rows,
        )
        _BOARD_CACHE["ts"] = time.time()
        _BOARD_CACHE["payload"] = payload
        _BOARD_CACHE["lastError"] = None
        _BOARD_DONE.set()
    except Exception as exc:  # noqa: BLE001
        _BOARD_CACHE["lastError"] = f"{type(exc).__name__}: {exc}"
        _BOARD_DONE.set()
    finally:
        with _BOARD_LOCK:
            _BOARD_CACHE["building"] = False


def _kick_rebuild(*, force: bool = False, breadth_interval: int = 5, breadth_rows: int = 5) -> None:
    threading.Thread(
        target=_rebuild_bg,
        kwargs={
            "force": force,
            "breadth_interval": breadth_interval,
            "breadth_rows": breadth_rows,
        },
        daemon=True,
        name="saint-sensex-rebuild",
    ).start()


def get_sensex_board(
    *,
    force: bool = False,
    breadth_interval: int = 5,
    breadth_rows: int = 5,
) -> dict[str, Any]:
    interval = _sanitize_interval(breadth_interval)
    rows = _sanitize_rows(breadth_rows)
    now = time.time()
    in_hours = is_live_data_window()
    ttl = _BOARD_TTL_S if in_hours else _BOARD_TTL_CLOSED_S
    cached = _BOARD_CACHE.get("payload")
    age = now - float(_BOARD_CACHE["ts"]) if cached is not None else None

    if cached is not None:
        fyers_live = bool(get_access_token()) and in_hours
        if fyers_live and age is not None and age >= _BOARD_LIVE_TTL_S and not force:
            out = _refresh_live_slice(dict(cached), breadth_interval=interval, breadth_rows=rows)
            out["cacheAgeS"] = 0.0
            out["building"] = bool(_BOARD_CACHE.get("building"))
            out["marketHours"] = in_hours
            out["marketHoursLabel"] = live_data_window_label()
            out["liveDataPaused"] = False
            out["breadthIntervalMin"] = interval
            out["breadthRows"] = rows
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
        out["breadthIntervalMin"] = interval
        out["breadthRows"] = rows
        out["breadthHistory"] = breadth_history(
            rows,
            index_key="sensex",
            interval_minutes=interval,
        )
        if force or stale:
            _kick_rebuild(force=force, breadth_interval=interval, breadth_rows=rows)
        return out

    if not in_hours and cached is not None and not force:
        out = dict(cached)
        out["cached"] = True
        out["liveDataPaused"] = True
        return out

    _kick_rebuild(force=force, breadth_interval=interval, breadth_rows=rows)
    empty = _empty_sensex_board(error=_BOARD_CACHE.get("lastError"))
    empty["building"] = bool(_BOARD_CACHE.get("building"))
    empty["breadthIntervalMin"] = interval
    empty["breadthRows"] = rows
    empty["breadthHistory"] = breadth_history(rows, index_key="sensex", interval_minutes=interval)
    return empty
