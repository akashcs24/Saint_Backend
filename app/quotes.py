from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

import yfinance as yf

from .config import settings
from .prices import last_close_from_cache
from .session import effective_quote_ttl_s
from .tickers import INDEX_NAMES, INDEX_YAHOO, to_yahoo_ticker


@dataclass
class Quote:
    symbol: str
    ltp: float
    change: float
    change_pct: float
    volume: float | None = None
    previous_close: float | None = None
    source: str = "yahoo"


_cache: dict[str, tuple[float, Quote]] = {}
_lock = Lock()


def _fi_get(fi: Any, *keys: str) -> Any:
    for key in keys:
        if fi is None:
            return None
        if isinstance(fi, dict):
            if key in fi and fi[key] is not None:
                return fi[key]
        else:
            val = getattr(fi, key, None)
            if val is not None:
                return val
    return None


def _from_yahoo(yahoo: str) -> Quote | None:
    try:
        t = yf.Ticker(yahoo)
        fi = getattr(t, "fast_info", None)
        ltp = _fi_get(fi, "last_price", "lastPrice")
        prev = _fi_get(fi, "previous_close", "previousClose")
        vol = _fi_get(fi, "last_volume", "lastVolume")

        if ltp is None:
            hist = t.history(period="5d", auto_adjust=True)
            if hist is None or hist.empty:
                return None
            ltp = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else ltp
            if "Volume" in hist.columns:
                vol = float(hist["Volume"].iloc[-1])

        ltp_f = float(ltp)
        prev_f = float(prev) if prev is not None else ltp_f
        change = ltp_f - prev_f
        change_pct = (change / prev_f * 100.0) if prev_f else 0.0
        return Quote(
            symbol=yahoo,
            ltp=round(ltp_f, 2),
            change=round(change, 2),
            change_pct=round(change_pct, 2),
            volume=float(vol) if vol is not None else None,
            previous_close=round(prev_f, 2),
            source="yahoo",
        )
    except Exception:
        return None


def _from_parquet(symbol: str) -> Quote | None:
    last, prev = last_close_from_cache(symbol)
    if last is None:
        return None
    prev = prev if prev is not None else last
    change = last - prev
    change_pct = (change / prev * 100.0) if prev else 0.0
    return Quote(
        symbol=to_yahoo_ticker(symbol),
        ltp=round(last, 2),
        change=round(change, 2),
        change_pct=round(change_pct, 2),
        previous_close=round(prev, 2),
        source="parquet",
    )


def get_quote(symbol: str) -> Quote | None:
    yahoo = to_yahoo_ticker(symbol)
    now = time.time()
    ttl = effective_quote_ttl_s()
    with _lock:
        hit = _cache.get(yahoo)
        if hit and now - hit[0] < ttl:
            return hit[1]

    q = _from_yahoo(yahoo) or _from_parquet(symbol)
    if q is None:
        return None
    with _lock:
        _cache[yahoo] = (now, q)
    return q


def get_index_quotes(keys: list[str] | None = None) -> list[dict]:
    keys = keys or ["NIFTY", "BANKNIFTY", "SENSEX", "VIX"]
    out: list[dict] = []
    for key in keys:
        if key not in INDEX_YAHOO and key not in INDEX_NAMES:
            continue
        q = get_quote(key)
        if not q:
            continue
        out.append(
            {
                "key": key,
                "name": INDEX_NAMES.get(key, key),
                "ltp": q.ltp,
                "change": q.change,
                "changePct": q.change_pct,
                "asOf": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": q.source,
            }
        )
    return out
