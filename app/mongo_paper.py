"""Mongo persistence for Nifty paper trades (dual-write with SQLite)."""

from __future__ import annotations

import time
from typing import Any

from .mongo import mongodb_configured, paper_trades_collection


def mongo_paper_enabled() -> bool:
    return mongodb_configured() and paper_trades_collection() is not None


def _doc_from_row_dict(trade: dict[str, Any], *, sqlite_id: int | None = None) -> dict[str, Any]:
    tid = sqlite_id if sqlite_id is not None else trade.get("id")
    return {
        "sqliteId": tid,
        "strategyId": trade.get("strategyId") or "decline",
        "status": trade.get("status"),
        "side": trade.get("side"),
        "symbol": trade.get("symbol"),
        "strike": trade.get("strike"),
        "expiry": trade.get("expiry"),
        "lot": trade.get("lot"),
        "entryTs": trade.get("entryTs"),
        "entryPx": trade.get("entryPx"),
        "entrySpot": trade.get("entrySpot"),
        "entryWeightUp": trade.get("entryWeightUp"),
        "entryReason": trade.get("entryReason"),
        "exitTs": trade.get("exitTs"),
        "exitPx": trade.get("exitPx"),
        "exitSpot": trade.get("exitSpot"),
        "exitWeightUp": trade.get("exitWeightUp"),
        "exitReason": trade.get("exitReason"),
        "peakPx": trade.get("peakPx"),
        "pnlRs": trade.get("pnlRs"),
        "pnlPct": trade.get("pnlPct"),
        "marginRs": trade.get("marginRs"),
        "updatedAt": time.time(),
    }


def upsert_paper_trade(trade: dict[str, Any], *, sqlite_id: int | None = None) -> bool:
    """Upsert one trade keyed by (strategyId, entryTs, symbol) or sqliteId."""
    col = paper_trades_collection()
    if col is None:
        return False
    doc = _doc_from_row_dict(trade, sqlite_id=sqlite_id)
    filt: dict[str, Any]
    if sqlite_id is not None:
        filt = {"sqliteId": int(sqlite_id)}
    elif trade.get("entryTs") and trade.get("strategyId"):
        filt = {
            "strategyId": doc["strategyId"],
            "entryTs": doc["entryTs"],
            "symbol": doc["symbol"],
        }
    else:
        return False
    col.update_one(filt, {"$set": doc}, upsert=True)
    return True


def list_mongo_paper_trades(
    limit: int = 50, *, strategy_id: str | None = None
) -> list[dict[str, Any]]:
    col = paper_trades_collection()
    if col is None:
        return []
    q: dict[str, Any] = {}
    if strategy_id:
        q["strategyId"] = strategy_id
    cur = col.find(q, {"_id": 0}).sort("entryTs", -1).limit(limit)
    out = []
    for d in cur:
        out.append(
            {
                "id": d.get("sqliteId") or 0,
                "strategyId": d.get("strategyId") or "decline",
                "status": d.get("status"),
                "side": d.get("side"),
                "symbol": d.get("symbol"),
                "strike": d.get("strike"),
                "expiry": d.get("expiry"),
                "lot": d.get("lot") or 65,
                "entryTs": d.get("entryTs"),
                "entryPx": d.get("entryPx"),
                "entrySpot": d.get("entrySpot"),
                "entryWeightUp": d.get("entryWeightUp"),
                "entryReason": d.get("entryReason"),
                "exitTs": d.get("exitTs"),
                "exitPx": d.get("exitPx"),
                "exitSpot": d.get("exitSpot"),
                "exitWeightUp": d.get("exitWeightUp"),
                "exitReason": d.get("exitReason"),
                "peakPx": d.get("peakPx"),
                "pnlRs": d.get("pnlRs"),
                "pnlPct": d.get("pnlPct"),
                "marginRs": d.get("marginRs"),
            }
        )
    return out


def sync_sqlite_trades_to_mongo(trades: list[dict[str, Any]]) -> int:
    """One-shot / best-effort push of local rows to Atlas."""
    n = 0
    for t in trades:
        try:
            if upsert_paper_trade(t, sqlite_id=t.get("id")):
                n += 1
        except Exception:  # noqa: BLE001
            continue
    return n
