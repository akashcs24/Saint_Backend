from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from .config import settings
from .fyers_quotes import to_fyers_eq
from .quotes import Quote
from .session import SESSION_CLOSE, SESSION_OPEN, is_cash_session_open

IST = ZoneInfo("Asia/Kolkata")
_lock = Lock()
_written_minute: dict[str, str] = {}


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(":", "_").replace("-", "_")


def _today_token(now_ist: datetime) -> str:
    return now_ist.date().isoformat()


def _minute_ts(now_ist: datetime) -> pd.Timestamp:
    return pd.Timestamp(now_ist.replace(second=0, microsecond=0).replace(tzinfo=None))


def _in_cash_session(now_ist: datetime) -> bool:
    t = now_ist.time()
    return SESSION_OPEN <= t <= SESSION_CLOSE and is_cash_session_open(now_ist)


def _upsert_minute_close(path: Path, ts_minute: pd.Timestamp, ltp: float, volume: float | None = None) -> None:
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if path.exists():
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df = df.set_index("Datetime")
        if len(df.index):
            df.index = pd.to_datetime(df.index)
        else:
            df.index = pd.DatetimeIndex([], name="Datetime")

    df.index = pd.to_datetime(df.index)
    row = pd.DataFrame(
        [{"Open": ltp, "High": ltp, "Low": ltp, "Close": ltp, "Volume": float(volume or 0.0)}],
        index=[ts_minute],
    )
    merged = pd.concat([df, row])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path)


def _write_symbol_minute(*, fyers_symbol: str, ltp: float, volume: float | None, now_ist: datetime) -> None:
    day = _today_token(now_ist)
    ts_minute = _minute_ts(now_ist)
    safe = _safe_symbol(fyers_symbol)

    stock_1m = settings.market_stocks_1m_dir.resolve() / f"{safe}_1_{day}_{day}.parquet"
    stock_5m = settings.market_stocks_5m_dir.resolve() / f"{safe}_5_{day}_{day}.parquet"
    _upsert_minute_close(stock_1m, ts_minute, ltp, volume)

    if ts_minute.minute % 5 == 0:
        _upsert_minute_close(stock_5m, ts_minute, ltp, volume)


def _write_index_minute(*, fyers_symbol: str, ltp: float, volume: float | None, now_ist: datetime) -> None:
    day = _today_token(now_ist)
    ts_minute = _minute_ts(now_ist)
    safe = _safe_symbol(fyers_symbol)

    idx_1m = settings.nifty_option_1m_dir.resolve() / f"{safe}_1_{day}_{day}.parquet"
    idx_5m = settings.nifty_option_5m_dir.resolve() / f"{safe}_5_{day}_{day}.parquet"
    _upsert_minute_close(idx_1m, ts_minute, ltp, volume)
    if ts_minute.minute % 5 == 0:
        _upsert_minute_close(idx_5m, ts_minute, ltp, volume)


def record_constituent_quotes(index_key: str, quotes: dict[str, Quote]) -> None:
    """Persist 1m/5m minute-close snapshots from the same quotes used for breadth."""
    now_ist = datetime.now(IST)
    if not _in_cash_session(now_ist):
        return
    minute_key = now_ist.strftime("%Y-%m-%d %H:%M")
    gate_key = f"{index_key}:stocks:{minute_key}"

    with _lock:
        if _written_minute.get(gate_key):
            return
        _written_minute[gate_key] = "1"

    for saint_symbol, q in quotes.items():
        if q is None:
            continue
        ltp = float(getattr(q, "ltp", 0.0) or 0.0)
        if ltp <= 0:
            continue
        fy_sym = to_fyers_eq(saint_symbol)
        vol = getattr(q, "volume", None)
        try:
            _write_symbol_minute(fyers_symbol=fy_sym, ltp=ltp, volume=vol, now_ist=now_ist)
        except Exception:  # noqa: BLE001
            continue


def record_index_quote(index_key: str, quote: Quote | None) -> None:
    now_ist = datetime.now(IST)
    if not _in_cash_session(now_ist):
        return
    if quote is None:
        return
    ltp = float(getattr(quote, "ltp", 0.0) or 0.0)
    if ltp <= 0:
        return

    idx_sym = "NSE:NIFTY50-INDEX" if index_key.lower() == "nifty" else "BSE:SENSEX-INDEX"
    minute_key = now_ist.strftime("%Y-%m-%d %H:%M")
    gate_key = f"{index_key}:index:{minute_key}"

    with _lock:
        if _written_minute.get(gate_key):
            return
        _written_minute[gate_key] = "1"

    vol = getattr(quote, "volume", None)
    try:
        _write_index_minute(fyers_symbol=idx_sym, ltp=ltp, volume=vol, now_ist=now_ist)
    except Exception:  # noqa: BLE001
        return


def clear_write_gates(prefixes: Iterable[str] | None = None) -> None:
    """Testing/maintenance helper to clear per-minute write gates."""
    with _lock:
        if not prefixes:
            _written_minute.clear()
            return
        keep = {}
        for k, v in _written_minute.items():
            if not any(k.startswith(p) for p in prefixes):
                keep[k] = v
        _written_minute.clear()
        _written_minute.update(keep)
