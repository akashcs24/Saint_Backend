"""Technical snapshot for AI helper — fresh Yahoo daily + session VWAP.

Computes RSI, EMA, MACD, Bollinger, volume structure, and today's VWAP from
15m bars when available. Designed for on-demand AI (force-refresh friendly).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .prices import _daily_cache_path, _normalize_ohlcv, fetch_intraday_15m, load_ohlcv
from .tickers import to_yahoo_ticker

IST = ZoneInfo("Asia/Kolkata")
_cache: dict[str, tuple[float, dict]] = {}
_TTL_S = 5 * 60  # AI path prefers fresher snapshots


def _pct(a: float, b: float) -> float | None:
    if not b:
        return None
    return round((a - b) / b * 100.0, 2)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _last(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    val = float(series.iloc[-1])
    return round(val, 2) if np.isfinite(val) else None


def _ensure_daily(symbol: str, *, force: bool = False, min_rows: int = 80) -> pd.DataFrame:
    """Load daily bars; force Yahoo refresh when stale or AI requests force."""
    path = _daily_cache_path(symbol)
    df = load_ohlcv(symbol)
    need = force or df.empty or len(df) < min_rows
    if not need and path.exists():
        # Stale if last bar older than ~2 calendar days (weekend-safe).
        try:
            last_ts = pd.Timestamp(df.index[-1]).to_pydatetime()
            if getattr(last_ts, "tzinfo", None) is None:
                last_ts = last_ts.replace(tzinfo=IST)
            age = datetime.now(IST) - last_ts.astimezone(IST)
            if age > timedelta(hours=60):
                need = True
        except Exception:  # noqa: BLE001
            need = True
    if not need and path.exists():
        mtime_age = time.time() - path.stat().st_mtime
        if mtime_age > 6 * 3600:
            need = True

    if not need:
        return df

    try:
        import yfinance as yf

        yahoo = to_yahoo_ticker(symbol)
        raw = yf.download(yahoo, period="1y", interval="1d", progress=False, auto_adjust=True)
        norm = _normalize_ohlcv(raw, ticker=yahoo)
        if norm.empty:
            return df
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            norm.to_parquet(path)
        except Exception:  # noqa: BLE001
            pass
        return norm
    except Exception:  # noqa: BLE001
        return df


def _session_vwap_intraday(symbol: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sessionVwap": None,
        "distVwapPct": None,
        "intradayBars": 0,
        "intradayAsOf": None,
        "aboveVwap": None,
    }
    try:
        bars = fetch_intraday_15m(symbol, force=True)
        if bars is None or bars.empty:
            return out
        today = datetime.now(IST).date()
        idx = pd.to_datetime(bars.index)
        day = bars.loc[[d.date() == today for d in idx]]
        if day.empty:
            # fall back to latest session day in the file
            last_day = pd.Timestamp(bars.index[-1]).date()
            day = bars.loc[[pd.Timestamp(d).date() == last_day for d in bars.index]]
        if day.empty or "Close" not in day.columns:
            return out
        typical = (
            day["High"].astype(float) + day["Low"].astype(float) + day["Close"].astype(float)
        ) / 3.0
        vol = day["Volume"].astype(float) if "Volume" in day.columns else pd.Series(1.0, index=day.index)
        denom = float(vol.sum())
        if denom <= 0:
            return out
        vwap = float((typical * vol).sum() / denom)
        last = float(day["Close"].astype(float).iloc[-1])
        out["sessionVwap"] = round(vwap, 2)
        out["distVwapPct"] = _pct(last, vwap)
        out["aboveVwap"] = last >= vwap
        out["intradayBars"] = int(len(day))
        out["intradayAsOf"] = str(pd.Timestamp(day.index[-1]))
    except Exception:  # noqa: BLE001
        return out
    return out


def get_technicals(
    symbol: str,
    *,
    ltp: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Full technical packet for AI timing decisions."""
    sym = symbol.upper()
    now = time.time()
    if not force:
        hit = _cache.get(sym)
        if hit and now - hit[0] < _TTL_S:
            out = dict(hit[1])
            if ltp and out.get("lastClose"):
                out["ltp"] = float(ltp)
                out["distFromLastClosePct"] = _pct(float(ltp), float(out["lastClose"]))
            return out

    df = _ensure_daily(sym, force=force)
    empty = {
        "ready": False,
        "source": "yahoo_daily+intraday",
        "bars": 0,
        "freshness": "stale_or_missing",
        "note": "Insufficient daily history",
        "deliveryPct": None,
        "deliveryNote": "NSE delivery % not available on free Yahoo feed",
    }
    if df.empty or "Close" not in df.columns or len(df) < 30:
        _cache[sym] = (now, empty)
        return dict(empty)

    close = df["Close"].astype(float)
    high = df["High"].astype(float) if "High" in df.columns else close
    low = df["Low"].astype(float) if "Low" in df.columns else close
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    px = float(ltp) if isinstance(ltp, (int, float)) and ltp > 0 else float(close.iloc[-1])
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close

    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    ema9_v, ema21_v, ema50_v = _last(ema9), _last(ema21), _last(ema50)

    # MACD(12,26,9)
    macd_line = _ema(close, 12) - _ema(close, 26)
    signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd_line - signal
    macd_v, signal_v, hist_v = _last(macd_line), _last(signal), _last(hist)
    macd_cross = None
    if len(hist) >= 2 and np.isfinite(hist.iloc[-1]) and np.isfinite(hist.iloc[-2]):
        if hist.iloc[-2] <= 0 < hist.iloc[-1]:
            macd_cross = "bullish_cross"
        elif hist.iloc[-2] >= 0 > hist.iloc[-1]:
            macd_cross = "bearish_cross"
        elif hist.iloc[-1] > 0:
            macd_cross = "bullish_hist"
        else:
            macd_cross = "bearish_hist"

    # Bollinger(20, 2)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    bb_mid, bb_up, bb_lo = _last(mid), _last(upper), _last(lower)
    bb_pos = None
    if bb_up is not None and bb_lo is not None and bb_up > bb_lo:
        bb_pos = round((px - bb_lo) / (bb_up - bb_lo), 3)

    rsi_s = _rsi_series(close, 14)
    rsi = _last(rsi_s)
    rsi_zone = "neutral"
    if rsi is not None:
        if rsi >= 70:
            rsi_zone = "overbought"
        elif rsi <= 30:
            rsi_zone = "oversold"
        elif rsi >= 55:
            rsi_zone = "bullish_bias"
        elif rsi <= 45:
            rsi_zone = "bearish_bias"

    sma20 = float(close.tail(20).mean()) if len(close) >= 20 else None
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

    hi20 = float(high.tail(20).max())
    lo20 = float(low.tail(20).min())
    vol20 = float(vol.tail(20).mean()) if len(vol.dropna()) >= 20 else None
    vol_last = float(vol.iloc[-1]) if len(vol) and np.isfinite(vol.iloc[-1]) else None
    vol_ratio = round(vol_last / vol20, 2) if vol20 and vol_last else None

    # EMA stack trend
    trend = "unclear"
    if ema9_v and ema21_v and ema50_v:
        if ema9_v > ema21_v > ema50_v:
            trend = "uptrend"
        elif ema9_v < ema21_v < ema50_v:
            trend = "downtrend"
        elif ema9_v > ema21_v:
            trend = "short_up"
        elif ema9_v < ema21_v:
            trend = "short_down"

    vwap_pack = _session_vwap_intraday(sym)

    as_of = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])
    try:
        bar_day = pd.Timestamp(df.index[-1]).date()
        today = datetime.now(IST).date()
        freshness = "today_or_prior_session" if (today - bar_day).days <= 3 else "stale"
    except Exception:  # noqa: BLE001
        freshness = "unknown"

    out: dict[str, Any] = {
        "ready": True,
        "source": "yahoo_daily+intraday_15m",
        "asOf": as_of,
        "freshness": freshness,
        "refreshedAt": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        "bars": int(len(df)),
        "ltp": round(px, 2),
        "lastClose": round(last_close, 2),
        "prevClose": round(prev_close, 2),
        "dayChangePct": _pct(px, prev_close),
        "distFromLastClosePct": _pct(px, last_close),
        "rsi14": rsi,
        "rsiZone": rsi_zone,
        "ema9": ema9_v,
        "ema21": ema21_v,
        "ema50": ema50_v,
        "distEma9Pct": _pct(px, ema9_v) if ema9_v else None,
        "distEma21Pct": _pct(px, ema21_v) if ema21_v else None,
        "distEma50Pct": _pct(px, ema50_v) if ema50_v else None,
        "sma20": round(sma20, 2) if sma20 else None,
        "sma50": round(sma50, 2) if sma50 else None,
        "sma200": round(sma200, 2) if sma200 else None,
        "macd": macd_v,
        "macdSignal": signal_v,
        "macdHist": hist_v,
        "macdState": macd_cross,
        "bbMid": bb_mid,
        "bbUpper": bb_up,
        "bbLower": bb_lo,
        "bbPosition": bb_pos,  # 0=lower band, 1=upper band
        "trend": trend,
        "high20": round(hi20, 2),
        "low20": round(lo20, 2),
        "distHigh20Pct": _pct(px, hi20),
        "distLow20Pct": _pct(px, lo20),
        "volumeLast": int(vol_last) if vol_last is not None else None,
        "volumeAvg20": int(vol20) if vol20 is not None else None,
        "volumeRatio20": vol_ratio,
        "sessionVwap": vwap_pack.get("sessionVwap"),
        "distVwapPct": vwap_pack.get("distVwapPct"),
        "aboveVwap": vwap_pack.get("aboveVwap"),
        "intradayAsOf": vwap_pack.get("intradayAsOf"),
        "intradayBars": vwap_pack.get("intradayBars"),
        "deliveryPct": None,
        "deliveryNote": "NSE delivery % not on free Yahoo feed — treat as unknown",
    }

    _cache[sym] = (now, out)
    return dict(out)
