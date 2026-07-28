from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock

import pandas as pd
import yfinance as yf

from .config import settings
from .tickers import parquet_name, to_yahoo_ticker

_fetch_lock = Lock()

# Ranges that use 15m bars (Yahoo keeps ~60 calendar days)
INTRADAY_RANGES = {"1D", "5D", "1M", "60D"}


def _daily_cache_path(symbol: str) -> Path:
    return settings.price_cache / parquet_name(to_yahoo_ticker(symbol))


def _intraday_dir() -> Path:
    d = settings.price_cache.parent / "price_cache_15m"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _intraday_cache_path(symbol: str) -> Path:
    return _intraday_dir() / parquet_name(to_yahoo_ticker(symbol))


def _normalize_ohlcv(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if ticker and ticker in df.columns.get_level_values(0):
            df = df[ticker].copy()
        else:
            df.columns = df.columns.get_level_values(0)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    cols = {str(c).lower(): c for c in df.columns}
    rename = {}
    for want in ("Open", "High", "Low", "Close", "Volume"):
        key = want.lower()
        if key in cols:
            rename[cols[key]] = want
    df = df.rename(columns=rename)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep].dropna(how="all")
    df.index = pd.to_datetime(df.index)
    # Normalize to naive IST wall-clock for chart labels
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    else:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_ohlcv(symbol: str) -> pd.DataFrame:
    """Load daily OHLCV from the local parquet cache. Empty if missing."""
    path = _daily_cache_path(symbol)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
    return _normalize_ohlcv(df)


def _cache_is_fresh(path: Path, ttl_s: int) -> bool:
    if not path.exists():
        return False
    age = datetime.now().timestamp() - path.stat().st_mtime
    return age <= ttl_s


def fetch_intraday_15m(symbol: str, *, force: bool = False) -> pd.DataFrame:
    """
    Load 15m OHLCV for ~60d.
    On stock-page open: reuse parquet if fresh, else pull Yahoo and cache.
    Near cash open the TTL drops so the first bars arrive sooner.
    """
    path = _intraday_cache_path(symbol)
    from .session import effective_intraday_ttl_s, is_open_window

    ttl = effective_intraday_ttl_s()
    # In the open window, a stale overnight cache must not block the first bars.
    if is_open_window() and path.exists():
        force = force or not _cache_is_fresh(path, ttl)

    if not force and _cache_is_fresh(path, ttl):
        try:
            cached = pd.read_parquet(path)
            return _normalize_ohlcv(cached)
        except Exception:
            pass

    yahoo = to_yahoo_ticker(symbol)
    with _fetch_lock:
        # Re-check after waiting for another request's download
        if not force and _cache_is_fresh(path, ttl):
            try:
                return _normalize_ohlcv(pd.read_parquet(path))
            except Exception:
                pass
        try:
            t = yf.Ticker(yahoo)
            raw = t.history(period="60d", interval="15m", auto_adjust=True)
            df = _normalize_ohlcv(raw, ticker=yahoo)
            if df.empty:
                # fall back to whatever cache we have, even if stale
                if path.exists():
                    return _normalize_ohlcv(pd.read_parquet(path))
                return pd.DataFrame()
            df.to_parquet(path)
            return df
        except Exception:
            if path.exists():
                try:
                    return _normalize_ohlcv(pd.read_parquet(path))
                except Exception:
                    return pd.DataFrame()
            return pd.DataFrame()


def _format_bar_time(ts: pd.Timestamp, *, intraday: bool) -> str:
    ts = pd.Timestamp(ts)
    if intraday:
        return ts.strftime("%Y-%m-%d %H:%M")
    return ts.strftime("%Y-%m-%d")


def _slice_intraday(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if df.empty:
        return df
    key = range_key.upper()
    now = df.index.max()
    if key == "1D":
        # last trading session only
        last_day = pd.Timestamp(now).normalize()
        return df.loc[df.index >= last_day]
    if key == "5D":
        # ~5 sessions ≈ 5 calendar days of bars, pad weekends
        cutoff = now - pd.Timedelta(days=8)
        return df.loc[df.index >= cutoff]
    if key == "1M":
        cutoff = now - pd.Timedelta(days=32)
        return df.loc[df.index >= cutoff]
    if key == "60D":
        # Yahoo 15m window is ~60 calendar days — use all cached bars
        return df
    return df


def price_series(
    symbol: str,
    *,
    range_key: str = "1M",
) -> tuple[list[dict], str]:
    """
    Return (points, interval).
    1D/5D/1M/60D → 15m (fetch+cache on demand).
    6M/1Y → daily parquet.
    """
    key = range_key.upper()
    if key in INTRADAY_RANGES:
        df = fetch_intraday_15m(symbol)
        if df.empty or "Close" not in df.columns:
            # graceful fallback to daily
            daily = load_ohlcv(symbol)
            if daily.empty:
                return [], "none"
            days = {"1D": 5, "5D": 5, "1M": 22, "60D": 60}.get(key, 22)
            sliced = daily.tail(days)
            points = _rows_to_points(sliced, intraday=False)
            return points, "1d-fallback"
        sliced = _slice_intraday(df, key)
        return _rows_to_points(sliced, intraday=True), "15m"

    df = load_ohlcv(symbol)
    if df.empty or "Close" not in df.columns:
        return [], "none"
    days = {"6M": 126, "1Y": 252}.get(key, 22)
    sliced = df.tail(days)
    return _rows_to_points(sliced, intraday=False), "1d"


def _rows_to_points(sliced: pd.DataFrame, *, intraday: bool) -> list[dict]:
    out: list[dict] = []
    for ts, row in sliced.iterrows():
        out.append(
            {
                "t": _format_bar_time(pd.Timestamp(ts), intraday=intraday),
                "p": round(float(row["Close"]), 2),
                "o": round(float(row["Open"]), 2) if "Open" in sliced.columns else None,
                "h": round(float(row["High"]), 2) if "High" in sliced.columns else None,
                "l": round(float(row["Low"]), 2) if "Low" in sliced.columns else None,
                "v": int(row["Volume"]) if "Volume" in sliced.columns and pd.notna(row["Volume"]) else None,
            }
        )
    return out


def last_close_from_cache(symbol: str) -> tuple[float | None, float | None]:
    """Return (last_close, prev_close) from parquet, if available."""
    df = load_ohlcv(symbol)
    if df.empty or "Close" not in df.columns or len(df) < 1:
        return None, None
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else None
    return last, prev


def close_on_or_before(symbol: str, when: datetime | pd.Timestamp) -> tuple[str | None, float | None]:
    """Nearest session close on/before `when`. Returns (YYYY-MM-DD, close)."""
    df = load_ohlcv(symbol)
    if df.empty or "Close" not in df.columns:
        return None, None
    ts = pd.Timestamp(when)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
    ts = ts.normalize()
    sliced = df.loc[df.index <= ts]
    if sliced.empty:
        return None, None
    row = sliced.iloc[-1]
    return pd.Timestamp(sliced.index[-1]).strftime("%Y-%m-%d"), float(row["Close"])


def nearest_bar_time(symbol: str, when: datetime | pd.Timestamp, *, prefer_intraday: bool = True) -> str | None:
    """Snap a news timestamp to the nearest 15m (or daily) bar label."""
    ts = pd.Timestamp(when)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)

    df = pd.DataFrame()
    if prefer_intraday:
        path = _intraday_cache_path(symbol)
        if path.exists():
            try:
                df = _normalize_ohlcv(pd.read_parquet(path))
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            df = fetch_intraday_15m(symbol)

    if not df.empty:
        # nearest bar at or before event; else first bar after
        before = df.loc[df.index <= ts]
        if not before.empty:
            return _format_bar_time(pd.Timestamp(before.index[-1]), intraday=True)
        after = df.loc[df.index > ts]
        if not after.empty:
            return _format_bar_time(pd.Timestamp(after.index[0]), intraday=True)

    day, _ = close_on_or_before(symbol, ts)
    return day


def price_at_or_before(
    symbol: str,
    when: datetime | pd.Timestamp,
    *,
    prefer_intraday: bool = True,
) -> tuple[str | None, float | None]:
    """Nearest tradeable price at or before ``when``.

    Prefers the last 15m close ≤ when (IST). Falls back to the prior daily close
    so overnight / weekend stories still have a baseline.
    """
    ts = pd.Timestamp(when)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
    else:
        ts = ts.tz_localize(None)

    if prefer_intraday:
        df = pd.DataFrame()
        path = _intraday_cache_path(symbol)
        if path.exists():
            try:
                df = _normalize_ohlcv(pd.read_parquet(path))
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            # Soft fetch — do not block the whole dashboard if Yahoo is slow.
            try:
                df = fetch_intraday_15m(symbol)
            except Exception:
                df = pd.DataFrame()
        if not df.empty and "Close" in df.columns:
            before = df.loc[df.index <= ts]
            if not before.empty:
                row = before.iloc[-1]
                label = _format_bar_time(pd.Timestamp(before.index[-1]), intraday=True)
                return label, float(row["Close"])

    day, close = close_on_or_before(symbol, ts)
    if day and close is not None:
        return f"close@{day}", close
    return None, None


def move_since_news_pct(symbol: str, news_item: dict, current_ltp: float) -> float | None:
    """% move from close on headline day → current LTP (legacy daily baseline)."""
    when = _news_when(news_item)
    if when is None:
        return None
    _day, close = close_on_or_before(symbol, when)
    if close is None or close <= 0:
        return None
    return round((current_ltp - close) / close * 100.0, 2)


def observed_move_from_news(
    symbol: str,
    news_item: dict,
    current_ltp: float,
    *,
    session_phase: str | None = None,
) -> dict:
    """Session-aware observed move for the dashboard buckets.

    During the cash session the baseline is the 15m bar at news time.
    Overnight / closed-day stories use the prior session close so the open gap
    is measurable.
    """
    when = _news_when(news_item)
    if when is None or current_ltp <= 0:
        return {
            "baselinePrice": None,
            "baselineLabel": None,
            "observedMovePct": None,
        }

    prefer_intraday = session_phase == "during_market"
    if session_phase in {"after_close", "closed_day", "before_open"}:
        # Anchor to the last completed session close at or before the story.
        from .session import prior_session_close

        close_at = prior_session_close(when.to_pydatetime() if hasattr(when, "to_pydatetime") else when)
        label, price = price_at_or_before(symbol, close_at, prefer_intraday=False)
    else:
        label, price = price_at_or_before(symbol, when, prefer_intraday=prefer_intraday)

    if price is None or price <= 0:
        return {
            "baselinePrice": None,
            "baselineLabel": None,
            "observedMovePct": None,
        }
    move = round((current_ltp - price) / price * 100.0, 2)
    return {
        "baselinePrice": round(price, 2),
        "baselineLabel": label,
        "observedMovePct": move,
    }


def _news_when(news_item: dict) -> pd.Timestamp | None:
    published = news_item.get("publishedAt")
    if published:
        try:
            return pd.Timestamp(published)
        except Exception:
            pass
    mins = news_item.get("minutesAgo")
    if mins is None or mins >= 9000:
        return None
    return pd.Timestamp.utcnow() - pd.Timedelta(minutes=int(mins))


def price_at_checkpoint(symbol: str, when: datetime | pd.Timestamp) -> tuple[str | None, float | None]:
    """Price used to judge a prediction at a named checkpoint."""
    return price_at_or_before(symbol, when, prefer_intraday=True)
