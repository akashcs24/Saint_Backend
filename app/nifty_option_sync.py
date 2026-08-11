from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .config import settings
from .fyers_auth import fyers_client, get_access_token
from .nifty_chain import atm_wing_board, fetch_nifty_option_chain
from .nifty_option_import import import_project_nifty_option_history

IST = "Asia/Kolkata"


@dataclass(frozen=True)
class SyncTarget:
    symbol: str
    resolution: str
    target_dir: Path
    lookback_days: int


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


def _history(symbol: str, resolution: str, range_from: date, range_to: date) -> pd.DataFrame:
    client = fyers_client()
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


def _write_cache(symbol: str, resolution: str, target_dir: Path, df: pd.DataFrame) -> Path:
    if df.empty:
        raise ValueError(f"Cannot write empty cache for {symbol}")
    start = pd.Timestamp(df.index.min()).date().isoformat()
    end = pd.Timestamp(df.index.max()).date().isoformat()
    path = target_dir / f"{_safe_name(symbol)}_{resolution}_{start}_{end}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    csv_path = path.with_suffix(".csv")
    df.to_csv(csv_path)
    return path


def _inventory_for_dir(target_dir: Path, resolution: str) -> dict[str, Any]:
    files = sorted(target_dir.rglob("NSE_NIFTY*")) + sorted(target_dir.rglob("NSE_NIFTY50_INDEX*"))
    files = [p for p in files if p.is_file()]
    latest_path = max(files, key=lambda p: p.stat().st_mtime, default=None)
    parquet_files = [p for p in files if p.suffix == ".parquet"]
    latest_parquet = max(parquet_files, key=lambda p: p.stat().st_mtime, default=None)
    latest_ts = None
    if latest_parquet:
        try:
            df = _normalize_ohlcv(pd.read_parquet(latest_parquet))
            if not df.empty:
                latest_ts = pd.Timestamp(df.index.max()).isoformat(sep=" ")
        except Exception:  # noqa: BLE001
            latest_ts = None
    unique_symbols = {
        "_".join(p.stem.split("_")[:-3]) if len(p.stem.split("_")) > 3 else p.stem
        for p in files
    }
    return {
        "resolution": resolution,
        "dir": str(target_dir),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "symbols": len(unique_symbols),
        "latestTsIst": latest_ts,
        "latestFile": str(latest_path) if latest_path else None,
    }


def _current_targets(*, wing: int = 10, allow_after_hours: bool = True) -> list[SyncTarget]:
    chain = fetch_nifty_option_chain(force=True, prefer_fyers_after_hours=allow_after_hours) or {}
    board = atm_wing_board(chain, wing=wing)
    symbols: list[str] = ["NSE:NIFTY50-INDEX"]
    for row in board.get("rows") or []:
        ce = str(row.get("ceSymbol") or "").strip()
        pe = str(row.get("peSymbol") or "").strip()
        if ce:
            symbols.append(ce)
        if pe:
            symbols.append(pe)
    # preserve order, de-dupe
    unique = list(dict.fromkeys(symbols))
    targets: list[SyncTarget] = []
    for symbol in unique:
        targets.append(SyncTarget(symbol, "1", settings.nifty_option_1m_dir.resolve(), 7))
        targets.append(SyncTarget(symbol, "5", settings.nifty_option_5m_dir.resolve(), 14))
        targets.append(SyncTarget(symbol, "15", settings.nifty_option_15m_dir.resolve(), 21))
    return targets


def sync_project_nifty_option_history(*, wing: int = 10, allow_after_hours: bool = True) -> dict[str, Any]:
    imported = import_project_nifty_option_history()
    if not get_access_token():
        return {
            "ok": True,
            "import": imported,
            "skipped": "no_fyers_token",
            "synced": [],
            "generatedAt": datetime.utcnow().isoformat() + "Z",
        }

    now_ist = datetime.now().astimezone().astimezone(tz=None)
    today_ist = datetime.now().date()
    synced: list[dict[str, Any]] = []
    for target in _current_targets(wing=wing, allow_after_hours=allow_after_hours):
        existing_df, _existing_path = _latest_cache(target.symbol, target.resolution, target.target_dir)
        if not existing_df.empty:
            latest_ts = pd.Timestamp(existing_df.index.max())
            range_from = max(latest_ts.date() - timedelta(days=1), today_ist - timedelta(days=target.lookback_days))
        else:
            latest_ts = None
            range_from = today_ist - timedelta(days=target.lookback_days)
        try:
            fresh_df = _history(target.symbol, target.resolution, range_from, today_ist)
        except Exception as exc:  # noqa: BLE001
            synced.append(
                {
                    "symbol": target.symbol,
                    "resolution": target.resolution,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        merged = pd.concat([existing_df, fresh_df]) if not existing_df.empty else fresh_df
        merged = _normalize_ohlcv(merged)
        if latest_ts is not None and not merged.empty and pd.Timestamp(merged.index.max()) <= latest_ts:
            synced.append(
                {
                    "symbol": target.symbol,
                    "resolution": target.resolution,
                    "updated": False,
                    "latestTsIst": latest_ts.isoformat(sep=" "),
                }
            )
            continue
        if merged.empty:
            synced.append({"symbol": target.symbol, "resolution": target.resolution, "updated": False})
            continue
        out_path = _write_cache(target.symbol, target.resolution, target.target_dir, merged)
        synced.append(
            {
                "symbol": target.symbol,
                "resolution": target.resolution,
                "updated": True,
                "rows": int(len(merged)),
                "latestTsIst": pd.Timestamp(merged.index.max()).isoformat(sep=" "),
                "path": str(out_path),
            }
        )

    return {
        "ok": True,
        "import": imported,
        "allowAfterHours": bool(allow_after_hours),
        "nowIst": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "synced": synced,
        "inventory": [
            _inventory_for_dir(settings.nifty_option_1m_dir.resolve(), "1m"),
            _inventory_for_dir(settings.nifty_option_5m_dir.resolve(), "5m"),
            _inventory_for_dir(settings.nifty_option_15m_dir.resolve(), "15m"),
        ],
    }