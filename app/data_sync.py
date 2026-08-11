from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import time
from typing import Any

import pandas as pd

from .config import settings
from .fyers_auth import fyers_client, get_access_token
from .fyers_quotes import to_fyers_eq
from .nifty_weights import get_nifty_weights
from .sensex_weights import get_sensex_weights

IST = "Asia/Kolkata"


@dataclass(frozen=True)
class SyncTarget:
    cache_symbol: str
    fetch_symbols: list[str]
    resolution: str
    target_dir: Path
    lookback_days: int
    family: str


def _safe_name(symbol: str) -> str:
    return symbol.replace(":", "_").replace("-", "_")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df = df.set_index("Datetime")
        df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(IST).tz_localize(None)
    else:
        df.index = df.index.tz_localize(None)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    out = df[keep].copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _history(symbol: str, resolution: str, range_from: date, range_to: date) -> pd.DataFrame:
    client = fyers_client()
    last_resp: Any = None
    backoff = [0.6, 1.2, 2.4]
    for attempt in range(len(backoff) + 1):
        resp = client.history(
            data={
                "symbol": symbol,
                "resolution": resolution,
                "date_format": "1",
                "range_from": range_from.isoformat(),
                "range_to": range_to.isoformat(),
                "cont_flag": "1",
            }
        )
        last_resp = resp
        if isinstance(resp, dict) and resp.get("s") == "ok":
            break

        code = resp.get("code") if isinstance(resp, dict) else None
        transient = code in {429, -16}
        if transient and attempt < len(backoff):
            time.sleep(backoff[attempt])
            continue
        raise RuntimeError(f"history failed for {symbol}: {resp}")

    resp = last_resp
    if not isinstance(resp, dict) or resp.get("s") != "ok":
        raise RuntimeError(f"history failed for {symbol}: {resp}")
    candles = resp.get("candles") or []
    if not candles:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    rows = []
    for candle in candles:
        ts = pd.to_datetime(int(candle[0]), unit="s", utc=True).tz_convert(IST).tz_localize(None)
        rows.append(
            {
                "Datetime": ts,
                "Open": float(candle[1]),
                "High": float(candle[2]),
                "Low": float(candle[3]),
                "Close": float(candle[4]),
                "Volume": float(candle[5]) if len(candle) > 5 else 0.0,
            }
        )
    return _normalize_ohlcv(pd.DataFrame(rows).set_index("Datetime"))


def _chunk_days_for_resolution(resolution: str) -> int:
    # Keep chunk sizes conservative so Fyers does not truncate long intraday requests.
    if resolution == "1":
        return 7
    if resolution == "5":
        return 21
    if resolution == "15":
        return 45
    return 60


def _history_range(symbol: str, resolution: str, range_from: date, range_to: date) -> pd.DataFrame:
    if range_to < range_from:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    step_days = _chunk_days_for_resolution(resolution)
    cursor = range_from
    chunks: list[pd.DataFrame] = []
    while cursor <= range_to:
        chunk_end = min(cursor + timedelta(days=step_days - 1), range_to)
        part = _history(symbol, resolution, cursor, chunk_end)
        if not part.empty:
            chunks.append(part)
        cursor = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    merged = pd.concat(chunks)
    return _normalize_ohlcv(merged)


def _history_any(symbols: list[str], resolution: str, range_from: date, range_to: date) -> tuple[str, pd.DataFrame]:
    last_err: Exception | None = None
    for sym in symbols:
        try:
            return sym, _history_range(sym, resolution, range_from, range_to)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if last_err is not None:
        raise RuntimeError(str(last_err)) from last_err
    raise RuntimeError("No fetch symbols supplied")


def _latest_cache(symbol: str, resolution: str, target_dir: Path) -> tuple[pd.DataFrame, Path | None]:
    safe = _safe_name(symbol)
    matches = sorted(target_dir.rglob(f"{safe}_{resolution}_*.parquet"))
    if not matches:
        return pd.DataFrame(), None
    path = matches[-1]
    try:
        df = _normalize_ohlcv(pd.read_parquet(path))
    except Exception:  # noqa: BLE001
        return pd.DataFrame(), None
    return df, path


def _write_cache(symbol: str, resolution: str, target_dir: Path, df: pd.DataFrame) -> Path:
    if df.empty:
        raise ValueError(f"Cannot write empty cache for {symbol}")
    start = pd.Timestamp(df.index.min()).date().isoformat()
    end = pd.Timestamp(df.index.max()).date().isoformat()
    path = target_dir / f"{_safe_name(symbol)}_{resolution}_{start}_{end}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    path.with_suffix(".csv").write_text(df.to_csv())
    return path


def _index_targets() -> list[SyncTarget]:
    return [
        SyncTarget(
            cache_symbol="NSE:NIFTY50-INDEX",
            fetch_symbols=["NSE:NIFTY50-INDEX", "NSE:NIFTY-INDEX"],
            resolution="1",
            target_dir=settings.nifty_option_1m_dir.resolve(),
            lookback_days=90,
            family="index",
        ),
        SyncTarget(
            cache_symbol="NSE:NIFTY50-INDEX",
            fetch_symbols=["NSE:NIFTY50-INDEX", "NSE:NIFTY-INDEX"],
            resolution="5",
            target_dir=settings.nifty_option_5m_dir.resolve(),
            lookback_days=90,
            family="index",
        ),
        SyncTarget(
            cache_symbol="BSE:SENSEX-INDEX",
            fetch_symbols=["BSE:SENSEX-INDEX", "BSE:SENSEX", "NSE:SENSEX-INDEX"],
            resolution="1",
            target_dir=settings.nifty_option_1m_dir.resolve(),
            lookback_days=90,
            family="index",
        ),
        SyncTarget(
            cache_symbol="BSE:SENSEX-INDEX",
            fetch_symbols=["BSE:SENSEX-INDEX", "BSE:SENSEX", "NSE:SENSEX-INDEX"],
            resolution="5",
            target_dir=settings.nifty_option_5m_dir.resolve(),
            lookback_days=90,
            family="index",
        ),
    ]


def _stock_symbols() -> list[str]:
    nifty = list((get_nifty_weights().get("weights") or {}).keys())
    sensex = list((get_sensex_weights().get("weights") or {}).keys())
    merged = [s for s in nifty + sensex if s]
    # stable de-dupe
    unique = list(dict.fromkeys(merged))
    return [to_fyers_eq(sym) for sym in unique]


def _stock_targets(symbols: list[str]) -> list[SyncTarget]:
    out: list[SyncTarget] = []
    for sym in symbols:
        out.append(
            SyncTarget(
                cache_symbol=sym,
                fetch_symbols=[sym],
                resolution="1",
                target_dir=settings.market_stocks_1m_dir.resolve(),
                lookback_days=5,
                family="stock",
            )
        )
        out.append(
            SyncTarget(
                cache_symbol=sym,
                fetch_symbols=[sym],
                resolution="5",
                target_dir=settings.market_stocks_5m_dir.resolve(),
                lookback_days=5,
                family="stock",
            )
        )
    return out


def _sync_target(target: SyncTarget, today_ist: date) -> dict[str, Any]:
    existing_df, _existing_path = _latest_cache(target.cache_symbol, target.resolution, target.target_dir)
    # Index caches must always represent the full configured window (e.g. 90d),
    # so refresh from the window start instead of only appending the latest day.
    if target.family == "index":
        latest_ts = pd.Timestamp(existing_df.index.max()) if not existing_df.empty else None
        range_from = today_ist - timedelta(days=target.lookback_days)
    elif not existing_df.empty:
        latest_ts = pd.Timestamp(existing_df.index.max())
        range_from = max(latest_ts.date() - timedelta(days=1), today_ist - timedelta(days=target.lookback_days))
    else:
        latest_ts = None
        range_from = today_ist - timedelta(days=target.lookback_days)

    used_symbol, fresh_df = _history_any(target.fetch_symbols, target.resolution, range_from, today_ist)
    merged = pd.concat([existing_df, fresh_df]) if not existing_df.empty else fresh_df
    merged = _normalize_ohlcv(merged)

    if latest_ts is not None and not merged.empty and pd.Timestamp(merged.index.max()) <= latest_ts:
        if target.family == "index" and not existing_df.empty:
            existing_min = pd.Timestamp(existing_df.index.min())
            merged_min = pd.Timestamp(merged.index.min())
            if len(merged) > len(existing_df) or merged_min < existing_min:
                pass
            else:
                return {
                    "family": target.family,
                    "symbol": target.cache_symbol,
                    "sourceSymbol": used_symbol,
                    "resolution": target.resolution,
                    "updated": False,
                    "latestTsIst": latest_ts.isoformat(sep=" "),
                }
        else:
            return {
                "family": target.family,
                "symbol": target.cache_symbol,
                "sourceSymbol": used_symbol,
                "resolution": target.resolution,
                "updated": False,
                "latestTsIst": latest_ts.isoformat(sep=" "),
            }
    if merged.empty:
        return {
            "family": target.family,
            "symbol": target.cache_symbol,
            "sourceSymbol": used_symbol,
            "resolution": target.resolution,
            "updated": False,
        }

    out_path = _write_cache(target.cache_symbol, target.resolution, target.target_dir, merged)
    return {
        "family": target.family,
        "symbol": target.cache_symbol,
        "sourceSymbol": used_symbol,
        "resolution": target.resolution,
        "updated": True,
        "rows": int(len(merged)),
        "latestTsIst": pd.Timestamp(merged.index.max()).isoformat(sep=" "),
        "path": str(out_path),
    }


def sync_market_data_batch(
    *,
    include_index: bool = True,
    include_stocks: bool = True,
    stock_cursor: int = 0,
    stock_batch_size: int = 20,
) -> dict[str, Any]:
    if not get_access_token():
        return {
            "ok": True,
            "skipped": "no_fyers_token",
            "generatedAt": datetime.utcnow().isoformat() + "Z",
            "includeIndex": include_index,
            "includeStocks": include_stocks,
            "stockCursor": int(stock_cursor),
            "stockBatchSize": int(stock_batch_size),
            "stockTotal": 0,
            "nextStockCursor": None,
            "synced": [],
        }

    cursor = max(0, int(stock_cursor))
    batch = max(1, min(50, int(stock_batch_size)))

    all_stock_symbols = _stock_symbols() if include_stocks else []
    selected_stock_symbols = all_stock_symbols[cursor : cursor + batch] if include_stocks else []
    next_cursor = cursor + batch if include_stocks and cursor + batch < len(all_stock_symbols) else None

    targets: list[SyncTarget] = []
    if include_index:
        targets.extend(_index_targets())
    if include_stocks and selected_stock_symbols:
        targets.extend(_stock_targets(selected_stock_symbols))

    today_ist = datetime.now().date()
    synced: list[dict[str, Any]] = []
    for target in targets:
        try:
            synced.append(_sync_target(target, today_ist))
        except Exception as exc:  # noqa: BLE001
            synced.append(
                {
                    "family": target.family,
                    "symbol": target.cache_symbol,
                    "resolution": target.resolution,
                    "updated": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    updated = sum(1 for row in synced if row.get("updated"))
    errored = sum(1 for row in synced if row.get("error"))

    return {
        "ok": True,
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "includeIndex": include_index,
        "includeStocks": include_stocks,
        "stockCursor": cursor,
        "stockBatchSize": batch,
        "stockTotal": len(all_stock_symbols),
        "nextStockCursor": next_cursor,
        "updated": updated,
        "errors": errored,
        "synced": synced,
    }
