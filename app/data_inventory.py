from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import settings
from .fyers_quotes import to_fyers_eq
from .mongo import mongo_db, mongo_ping, mongodb_configured
from .nifty_breadth_history import breadth_history
from .nifty_pcr_history import pcr_history
from .nifty_weights import get_nifty_weights
from .sensex_weights import get_sensex_weights

try:
    import pyarrow.parquet as pq
except Exception:  # noqa: BLE001
    pq = None


def _parse_ymd(token: str) -> date | None:
    try:
        return datetime.strptime(token, "%Y-%m-%d").date()
    except Exception:  # noqa: BLE001
        return None


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(":", "_").replace("-", "_")


def _row_count(path: Path) -> int:
    if pq is None:
        return 0
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001
        return 0


def _parquet_minmax_dates(path: Path) -> tuple[date | None, date | None]:
    if pq is None:
        return None, None
    try:
        pf = pq.ParquetFile(path)
    except Exception:  # noqa: BLE001
        return None, None

    # Pandas-written parquet usually stores Datetime index as __index_level_0__.
    # Fall back to common explicit timestamp column names.
    candidates = ["__index_level_0__", "Datetime", "datetime", "timestamp", "ts", "asOf"]
    names = list(pf.schema.names)
    col_name = next((c for c in candidates if c in names), None)
    if col_name is None:
        return None, None

    try:
        col_idx = names.index(col_name)
    except ValueError:
        return None, None

    min_dt = None
    max_dt = None
    for i in range(pf.num_row_groups):
        try:
            col = pf.metadata.row_group(i).column(col_idx)
            st = col.statistics
            if st is None:
                continue
            lo = getattr(st, "min", None)
            hi = getattr(st, "max", None)
            if lo is None or hi is None:
                continue
            lo_dt = datetime.fromisoformat(str(lo))
            hi_dt = datetime.fromisoformat(str(hi))
            min_dt = lo_dt if min_dt is None or lo_dt < min_dt else min_dt
            max_dt = hi_dt if max_dt is None or hi_dt > max_dt else max_dt
        except Exception:  # noqa: BLE001
            continue

    if min_dt is None or max_dt is None:
        return None, None
    return min_dt.date(), max_dt.date()


def _period_days(start_iso: str | None, end_iso: str | None) -> int | None:
    if not start_iso or not end_iso:
        return None
    s = _parse_ymd(start_iso)
    e = _parse_ymd(end_iso)
    if s is None or e is None:
        return None
    if e < s:
        return None
    # Inclusive day span shown in the UI target column.
    return (e - s).days + 1


def _scan_parquet(dir_path: Path, pattern: str, timeframe_token: str) -> dict[str, Any]:
    files = sorted([p for p in dir_path.rglob(pattern) if p.suffix == ".parquet"]) if dir_path.exists() else []
    if not files:
        return {
            "records": 0,
            "symbols": 0,
            "symbolNames": [],
            "periodStart": None,
            "periodEnd": None,
            "latestFile": None,
            "files": 0,
            "bytes": 0,
        }

    symbols: set[str] = set()
    start_dates: list[date] = []
    end_dates: list[date] = []
    records = 0
    for p in files:
        stem_parts = p.stem.split("_")
        if len(stem_parts) >= 4 and stem_parts[-3] == timeframe_token:
            symbol = "_".join(stem_parts[:-3])
            symbols.add(symbol)
            ds_meta, de_meta = _parquet_minmax_dates(p)
            ds = ds_meta or _parse_ymd(stem_parts[-2])
            de = de_meta or _parse_ymd(stem_parts[-1])
            if ds:
                start_dates.append(ds)
            if de:
                end_dates.append(de)
        records += _row_count(p)
    total_bytes = sum(p.stat().st_size for p in files)

    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    return {
        "records": int(records),
        "symbols": len(symbols),
        "symbolNames": sorted(symbols),
        "periodStart": min(start_dates).isoformat() if start_dates else None,
        "periodEnd": max(end_dates).isoformat() if end_dates else None,
        "latestFile": str(latest_file),
        "files": len(files),
        "bytes": int(total_bytes),
    }


def _status_for(end_iso: str | None) -> tuple[str, int | None]:
    if not end_iso:
        return "empty", None
    end_d = _parse_ymd(end_iso)
    if end_d is None:
        return "lagging", None
    lag_days = (date.today() - end_d).days
    if lag_days <= 1:
        return "up_to_date", lag_days
    return "lagging", lag_days


def _mongo_collection_stats(collection_name: str) -> dict[str, Any]:
    db = mongo_db()
    if db is None:
        return {"collection": collection_name, "records": 0, "latestTs": None}
    try:
        col = db[collection_name]
        records = int(col.estimated_document_count())
        latest = col.find_one(sort=[("t", -1)]) or col.find_one(sort=[("ts", -1)])
        latest_ts = latest.get("t") if isinstance(latest, dict) else None
        if latest_ts is None and isinstance(latest, dict):
            latest_ts = latest.get("ts")
        size_bytes = 0
        storage_bytes = 0
        index_bytes = 0
        try:
            stats = db.command("collStats", collection_name)
            size_bytes = int(stats.get("size") or 0)
            storage_bytes = int(stats.get("storageSize") or 0)
            index_bytes = int(stats.get("totalIndexSize") or 0)
        except Exception:  # noqa: BLE001
            pass
        return {
            "collection": collection_name,
            "records": records,
            "latestTs": latest_ts,
            "sizeBytes": size_bytes,
            "storageBytes": storage_bytes,
            "indexBytes": index_bytes,
        }
    except Exception:  # noqa: BLE001
        return {
            "collection": collection_name,
            "records": 0,
            "latestTs": None,
            "sizeBytes": 0,
            "storageBytes": 0,
            "indexBytes": 0,
        }


def _mongo_db_storage_stats() -> dict[str, Any]:
    db = mongo_db()
    if db is None:
        return {
            "ok": False,
            "dataBytes": 0,
            "storageBytes": 0,
            "indexBytes": 0,
            "totalBytes": 0,
        }
    try:
        stats = db.command("dbStats", scale=1)
        data_bytes = int(stats.get("dataSize") or 0)
        storage_bytes = int(stats.get("storageSize") or 0)
        index_bytes = int(stats.get("indexSize") or 0)
        total_bytes = int(stats.get("totalSize") or (storage_bytes + index_bytes))
        return {
            "ok": True,
            "dataBytes": data_bytes,
            "storageBytes": storage_bytes,
            "indexBytes": index_bytes,
            "totalBytes": total_bytes,
        }
    except Exception:  # noqa: BLE001
        return {
            "ok": False,
            "dataBytes": 0,
            "storageBytes": 0,
            "indexBytes": 0,
            "totalBytes": 0,
        }


def get_data_inventory() -> dict[str, Any]:
    nifty_weight_map = get_nifty_weights().get("weights") or {}
    sensex_weight_map = get_sensex_weights().get("weights") or {}
    nifty_expected = len(nifty_weight_map)
    sensex_expected = len(sensex_weight_map)

    idx_nifty_1m = _scan_parquet(settings.nifty_option_1m_dir.resolve(), "*NIFTY50_INDEX_1_*.parquet", "1")
    idx_nifty_5m = _scan_parquet(settings.nifty_option_5m_dir.resolve(), "*NIFTY50_INDEX_5_*.parquet", "5")
    idx_sensex_1m = _scan_parquet(settings.nifty_option_1m_dir.resolve(), "*SENSEX*INDEX_1_*.parquet", "1")
    idx_sensex_5m = _scan_parquet(settings.nifty_option_5m_dir.resolve(), "*SENSEX*INDEX_5_*.parquet", "5")

    stocks_1m = _scan_parquet(settings.market_stocks_1m_dir.resolve(), "*.parquet", "1")
    stocks_5m = _scan_parquet(settings.market_stocks_5m_dir.resolve(), "*.parquet", "5")

    def _expected_aliases(symbol: str) -> set[str]:
        sym = symbol.strip().upper()
        return {
            sym,
            _safe_symbol(to_fyers_eq(sym)),
        }

    nifty_expected_safe = {_safe_symbol(to_fyers_eq(sym)) for sym in nifty_weight_map.keys()}
    sensex_expected_safe = {_safe_symbol(to_fyers_eq(sym)) for sym in sensex_weight_map.keys()}
    stock_1m_safe = set(stocks_1m.get("symbolNames") or [])
    stock_5m_safe = set(stocks_5m.get("symbolNames") or [])

    def _count_available(expected_map: dict[str, float], stock_symbols: set[str]) -> int:
        hit = 0
        for sym in expected_map.keys():
            if _expected_aliases(sym).intersection(stock_symbols):
                hit += 1
        return hit

    nifty_available_1m = _count_available(nifty_weight_map, stock_1m_safe)
    nifty_available_5m = _count_available(nifty_weight_map, stock_5m_safe)
    sensex_available_1m = _count_available(sensex_weight_map, stock_1m_safe)
    sensex_available_5m = _count_available(sensex_weight_map, stock_5m_safe)

    index_rows = []
    for name, tf, target_days, payload in [
        ("NIFTY", "1m", None, idx_nifty_1m),
        ("NIFTY", "5m", None, idx_nifty_5m),
        ("SENSEX", "1m", None, idx_sensex_1m),
        ("SENSEX", "5m", None, idx_sensex_5m),
    ]:
        status, lag_days = _status_for(payload["periodEnd"])
        computed_days = _period_days(payload.get("periodStart"), payload.get("periodEnd"))
        index_rows.append(
            {
                "name": name,
                "timeframe": tf,
                "records": payload["records"],
                "files": payload["files"],
                "bytes": payload["bytes"],
                "periodStart": payload["periodStart"],
                "periodEnd": payload["periodEnd"],
                "targetDays": computed_days if computed_days is not None else target_days,
                "status": status,
                "lagDays": lag_days,
            }
        )

    stock_rows = []
    for market, expected, available, tf, target_days, payload in [
        ("NIFTY50 constituents", nifty_expected, nifty_available_1m, "1m", None, stocks_1m),
        ("NIFTY50 constituents", nifty_expected, nifty_available_5m, "5m", None, stocks_5m),
        ("SENSEX constituents", sensex_expected, sensex_available_1m, "1m", None, stocks_1m),
        ("SENSEX constituents", sensex_expected, sensex_available_5m, "5m", None, stocks_5m),
    ]:
        status, lag_days = _status_for(payload["periodEnd"])
        coverage_pct = round((available / expected) * 100, 1) if expected > 0 else 0.0
        computed_days = _period_days(payload.get("periodStart"), payload.get("periodEnd"))
        stock_rows.append(
            {
                "market": market,
                "timeframe": tf,
                "records": payload["records"],
                "symbolsAvailable": available,
                "symbolsExpected": expected,
                "coveragePct": coverage_pct,
                "files": payload["files"],
                "bytes": payload["bytes"],
                "periodStart": payload["periodStart"],
                "periodEnd": payload["periodEnd"],
                "targetDays": computed_days if computed_days is not None else target_days,
                "status": status,
                "lagDays": lag_days,
            }
        )

    mongo_health = mongo_ping()
    mongo_db_stats = _mongo_db_storage_stats()
    mongo_rows = []
    for col_name in [
        "market_index_bars_1m",
        "market_index_bars_5m",
        "market_stock_bars_1m",
        "market_stock_bars_5m",
    ]:
        mongo_rows.append(_mongo_collection_stats(col_name))

    local_index_total_bytes = int(sum(int(r.get("bytes") or 0) for r in index_rows))
    local_stock_total_bytes = int(sum(int(r.get("bytes") or 0) for r in stock_rows))
    local_total_bytes = local_index_total_bytes + local_stock_total_bytes
    mongo_total_bytes = int(mongo_db_stats.get("totalBytes") or 0)
    mongo_cap_bytes = 500 * 1024 * 1024
    mongo_usage_pct = round((mongo_total_bytes / mongo_cap_bytes) * 100, 2) if mongo_cap_bytes > 0 else 0.0

    breadth_1m_nifty = breadth_history(limit=3000, index_key="nifty", interval_minutes=1)
    breadth_5m_nifty = breadth_history(limit=3000, index_key="nifty", interval_minutes=5)
    breadth_1m_sensex = breadth_history(limit=3000, index_key="sensex", interval_minutes=1)
    breadth_5m_sensex = breadth_history(limit=3000, index_key="sensex", interval_minutes=5)
    pcr_nifty = pcr_history(limit=3000, index_key="nifty")
    pcr_sensex = pcr_history(limit=3000, index_key="sensex")

    return {
        "asOf": datetime.utcnow().isoformat() + "Z",
        "mongo": {
            "configured": mongodb_configured(),
            "ok": bool(mongo_health.get("ok")),
            "db": getattr(settings, "mongodb_db", "saint"),
            "detail": mongo_health,
            "collections": mongo_rows,
            "storage": {
                "dataBytes": int(mongo_db_stats.get("dataBytes") or 0),
                "storageBytes": int(mongo_db_stats.get("storageBytes") or 0),
                "indexBytes": int(mongo_db_stats.get("indexBytes") or 0),
                "totalBytes": mongo_total_bytes,
                "capBytes": mongo_cap_bytes,
                "usagePct": mongo_usage_pct,
            },
        },
        "indexData": index_rows,
        "stockData": stock_rows,
        "totals": {
            "localIndexBytes": local_index_total_bytes,
            "localStockBytes": local_stock_total_bytes,
            "localTotalBytes": local_total_bytes,
            "mongoTotalBytes": mongo_total_bytes,
            "combinedTotalBytes": local_total_bytes + mongo_total_bytes,
        },
        "historyData": {
            "breadth": {
                "nifty1mRows": len(breadth_1m_nifty),
                "nifty5mRows": len(breadth_5m_nifty),
                "sensex1mRows": len(breadth_1m_sensex),
                "sensex5mRows": len(breadth_5m_sensex),
            },
            "pcr": {
                "niftyRows": len(pcr_nifty),
                "sensexRows": len(pcr_sensex),
            },
        },
    }
