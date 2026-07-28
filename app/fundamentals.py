"""Yahoo fundamentals for NSE names (delayed / best-effort)."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

import yfinance as yf

from .tickers import to_yahoo_ticker

_cache: dict[str, tuple[float, dict]] = {}
_lock = Lock()
TTL_S = 3600


def _fmt_mcap(val: Any) -> str | None:
    try:
        n = float(val)
    except (TypeError, ValueError):
        return None
    # Yahoo returns absolute INR for .NS sometimes; treat as rupees
    cr = n / 1e7  # crores
    if cr >= 1e5:
        return f"₹{cr / 1e5:.2f} L Cr"
    if cr >= 1:
        return f"₹{cr:,.0f} Cr"
    return f"₹{n:,.0f}"


def get_fundamentals(symbol: str) -> dict:
    yahoo = to_yahoo_ticker(symbol)
    now = time.time()
    with _lock:
        hit = _cache.get(yahoo)
        if hit and now - hit[0] < TTL_S:
            return dict(hit[1])

    out = {
        "name": symbol.upper(),
        "sector": None,
        "marketCap": None,
        "peRatio": None,
        "about": None,
    }
    try:
        t = yf.Ticker(yahoo)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        out["name"] = info.get("longName") or info.get("shortName") or symbol.upper()
        out["sector"] = info.get("sector") or info.get("industry")
        out["marketCap"] = _fmt_mcap(info.get("marketCap"))
        pe = info.get("trailingPE") or info.get("forwardPE")
        out["peRatio"] = round(float(pe), 2) if pe is not None else None
        about = info.get("longBusinessSummary")
        out["about"] = about[:500] if isinstance(about, str) else None
    except Exception:
        pass

    with _lock:
        _cache[yahoo] = (now, out)
    return dict(out)


def _num(val: Any, nd: int = 2) -> float | None:
    try:
        if val is None:
            return None
        n = float(val)
        if n != n:  # NaN
            return None
        return round(n, nd)
    except (TypeError, ValueError):
        return None


def get_fundamentals_deep(symbol: str) -> dict:
    """Richer Yahoo fundamentals packet for the AI helper layer."""
    yahoo = to_yahoo_ticker(symbol)
    cache_key = f"{yahoo}::deep"
    now = time.time()
    with _lock:
        hit = _cache.get(cache_key)
        if hit and now - hit[0] < TTL_S:
            return dict(hit[1])

    base = get_fundamentals(symbol)
    out: dict[str, Any] = {
        "ready": False,
        "source": "yahoo_finance",
        "symbol": symbol.upper(),
        "name": base.get("name"),
        "sector": base.get("sector"),
        "industry": None,
        "marketCap": base.get("marketCap"),
        "marketCapRaw": None,
        "trailingPE": base.get("peRatio"),
        "forwardPE": None,
        "pegRatio": None,
        "priceToBook": None,
        "roe": None,
        "profitMargins": None,
        "operatingMargins": None,
        "revenueGrowth": None,
        "earningsGrowth": None,
        "debtToEquity": None,
        "currentRatio": None,
        "freeCashflow": None,
        "dividendYield": None,
        "payoutRatio": None,
        "52WeekHigh": None,
        "52WeekLow": None,
        "beta": None,
        "recommendationMean": None,
        "targetMeanPrice": None,
        "numberOfAnalystOpinions": None,
        "about": base.get("about"),
    }
    try:
        t = yf.Ticker(yahoo)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        out["ready"] = bool(info)
        out["industry"] = info.get("industry")
        out["marketCapRaw"] = info.get("marketCap")
        out["trailingPE"] = _num(info.get("trailingPE")) or out["trailingPE"]
        out["forwardPE"] = _num(info.get("forwardPE"))
        out["pegRatio"] = _num(info.get("pegRatio"))
        out["priceToBook"] = _num(info.get("priceToBook"))
        out["roe"] = _num((info.get("returnOnEquity") or 0) * 100) if info.get("returnOnEquity") is not None else None
        pm = info.get("profitMargins")
        om = info.get("operatingMargins")
        out["profitMargins"] = _num(pm * 100) if pm is not None else None
        out["operatingMargins"] = _num(om * 100) if om is not None else None
        rg = info.get("revenueGrowth")
        eg = info.get("earningsGrowth")
        out["revenueGrowth"] = _num(rg * 100) if rg is not None else None
        out["earningsGrowth"] = _num(eg * 100) if eg is not None else None
        out["debtToEquity"] = _num(info.get("debtToEquity"), 1)
        out["currentRatio"] = _num(info.get("currentRatio"))
        out["freeCashflow"] = info.get("freeCashflow")
        dy = info.get("dividendYield")
        out["dividendYield"] = _num(dy * 100 if dy is not None and dy < 1 else dy)
        pr = info.get("payoutRatio")
        out["payoutRatio"] = _num(pr * 100) if pr is not None else None
        out["52WeekHigh"] = _num(info.get("fiftyTwoWeekHigh"))
        out["52WeekLow"] = _num(info.get("fiftyTwoWeekLow"))
        out["beta"] = _num(info.get("beta"))
        out["recommendationMean"] = _num(info.get("recommendationMean"))
        out["targetMeanPrice"] = _num(info.get("targetMeanPrice"))
        out["numberOfAnalystOpinions"] = info.get("numberOfAnalystOpinions")
        if not out.get("marketCap"):
            out["marketCap"] = _fmt_mcap(info.get("marketCap"))
    except Exception:
        pass

    with _lock:
        _cache[cache_key] = (now, out)
    return dict(out)
