"""Nifty ATM paper trades — multi-strategy buckets.

Strategies (extensible via STRATEGIES dict):
  decline — entry weightUp rising ×3; exit weightUp falling ×4
  tsl     — entry weightUp rising ×3; exit SL −20% or TSL 10% after +15%
  cross   — entry marketSync cross.diffPp > 0; exit when < 0

Prices: Fyers optionchain ATM CE (prefer after-hours too when token live).
Storage: SQLite locally + MongoDB Atlas dual-write when SAINT_MONGODB_URI is set
(Render should use Mongo so trades survive free-tier disk wipes).
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from zoneinfo import ZoneInfo

from .config import settings
from .session import is_live_data_window

IST = ZoneInfo("Asia/Kolkata")
LOT_SIZE = 65
ENTRY_RISE_BARS = 3
EXIT_DECLINE_BARS = 4
SL_PCT = -20.0
TSL_PCT = 10.0
TSL_ARM_PCT = 15.0
# Each strategy book starts with ₹1L so monthly return is comparable side-by-side.
PAPER_STARTING_CAPITAL_RS = 100_000.0

ExitMode = Literal["decline", "tsl", "cross"]
EntryMode = Literal["weight_up_rise", "cross_gt_0"]

STRATEGIES: dict[str, dict[str, Any]] = {
    "decline": {
        "id": "decline",
        "label": "Decline ×4",
        "entry": f"weightUp rising ×{ENTRY_RISE_BARS}",
        "exit": f"weightUp falling ×{EXIT_DECLINE_BARS}",
        "entryMode": "weight_up_rise",
        "exitMode": "decline",
        "lot": LOT_SIZE,
        "side": "ATM CE long",
    },
    "tsl": {
        "id": "tsl",
        "label": "SL / TSL",
        "entry": f"weightUp rising ×{ENTRY_RISE_BARS}",
        "exit": f"SL {SL_PCT}% / TSL {TSL_PCT}% after +{TSL_ARM_PCT}%",
        "entryMode": "weight_up_rise",
        "exitMode": "tsl",
        "lot": LOT_SIZE,
        "side": "ATM CE long",
    },
    "cross": {
        "id": "cross",
        "label": "Sync cross",
        "entry": "cross.diffPp > 0",
        "exit": "cross.diffPp < 0",
        "entryMode": "cross_gt_0",
        "exitMode": "cross",
        "lot": LOT_SIZE,
        "side": "ATM CE long",
    },
}

_lock = Lock()
_LIVE_LTP_CACHE: dict[str, Any] = {"ts": 0.0, "payload": None}
_LIVE_LTP_TTL_S = 2.0
_LAST_EVAL_TS = 0.0
_EVAL_MIN_S = 60.0


def list_strategies() -> list[dict[str, Any]]:
    return [dict(s) for s in STRATEGIES.values()]


def _mongo_ready_flag() -> bool:
    try:
        from .mongo import mongo_is_reachable

        return mongo_is_reachable()
    except Exception:  # noqa: BLE001
        return False


def _storage_label() -> str:
    if getattr(settings, "mongodb_uri", ""):
        return f"mongodb:{getattr(settings, 'mongodb_db', 'saint')}/nifty_paper_trades (+ sqlite mirror)"
    return str(_db_path())


def _db_path() -> Path:
    path = Path(settings.alerts_db).resolve().parent / "nifty_paper_trades.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(_db_path()), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL DEFAULT 'decline',
            status TEXT NOT NULL,
            side TEXT NOT NULL,
            symbol TEXT NOT NULL,
            strike REAL,
            expiry TEXT,
            lot INTEGER NOT NULL,
            entry_ts TEXT NOT NULL,
            entry_px REAL NOT NULL,
            entry_spot REAL,
            entry_weight_up REAL,
            entry_reason TEXT,
            exit_ts TEXT,
            exit_px REAL,
            exit_spot REAL,
            exit_weight_up REAL,
            exit_reason TEXT,
            peak_px REAL,
            pnl_rs REAL,
            pnl_pct REAL,
            meta TEXT
        )
        """
    )
    cols = {r[1] for r in con.execute("PRAGMA table_info(paper_trades)").fetchall()}
    if "strategy_id" not in cols:
        con.execute(
            "ALTER TABLE paper_trades ADD COLUMN strategy_id TEXT NOT NULL DEFAULT 'decline'"
        )
        con.commit()
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_strategy_status ON paper_trades(strategy_id, status)"
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    return con


def _sync_trade_to_mongo(con: sqlite3.Connection, trade_id: int) -> None:
    try:
        from .mongo_paper import upsert_paper_trade

        row = con.execute(
            "SELECT * FROM paper_trades WHERE id=?", (trade_id,)
        ).fetchone()
        if not row:
            return
        upsert_paper_trade(_row_dict(row), sqlite_id=trade_id)
    except Exception:  # noqa: BLE001
        pass


def _atm_ce_quote() -> dict[str, Any] | None:
    """ATM CE LTP — prefer Fyers (works after hours when token is live)."""
    try:
        from .nifty_chain import fetch_nifty_option_chain, atm_wing_board

        chain = (
            fetch_nifty_option_chain(force=True, prefer_fyers_after_hours=True) or {}
        )
        wing = atm_wing_board(chain, wing=2)
        if not wing.get("ready"):
            return None
        atm = wing.get("atmStrike")
        rows = wing.get("rows") or wing.get("strikes") or []
        hit = None
        for r in rows:
            if atm is not None and float(r.get("strike") or 0) == float(atm):
                hit = r
                break
        if hit is None and rows:
            hit = min(
                rows,
                key=lambda r: abs(float(r.get("strike") or 0) - float(atm or 0)),
            )
        if not hit:
            return None
        ltp = hit.get("ceLtp")
        if ltp is None:
            return None
        return {
            "symbol": hit.get("ceSymbol")
            or hit.get("symbol")
            or f"NSE:NIFTY-ATM-{int(float(atm))}CE",
            "strike": float(atm) if atm is not None else float(hit.get("strike") or 0),
            "expiry": chain.get("expiry") or wing.get("expiry"),
            "ltp": float(ltp),
            "spot": chain.get("spot") or wing.get("spot"),
            "source": chain.get("source") or "fyers",
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _chrono_weight_up(limit: int = 12) -> list[dict[str, Any]]:
    from .nifty_breadth_history import breadth_history

    rows = list(reversed(breadth_history(limit=limit)))
    out = []
    for r in rows:
        if r.get("weightUp") is None:
            continue
        out.append(r)
    return out


def _rising_streak(vals: list[float], n: int) -> bool:
    if len(vals) < n + 1:
        return False
    window = vals[-(n + 1) :]
    return all(window[i] > window[i - 1] for i in range(1, len(window)))


def _falling_streak(vals: list[float], n: int) -> bool:
    if len(vals) < n + 1:
        return False
    window = vals[-(n + 1) :]
    return all(window[i] < window[i - 1] for i in range(1, len(window)))


def _open_trade(con: sqlite3.Connection, strategy_id: str) -> sqlite3.Row | None:
    cur = con.execute(
        "SELECT * FROM paper_trades WHERE status='open' AND strategy_id=? ORDER BY id DESC LIMIT 1",
        (strategy_id,),
    )
    row = cur.fetchone()
    if row is not None:
        return row
    # Render / fresh disk: recover open from Atlas
    try:
        from .mongo_paper import list_mongo_paper_trades

        for t in list_mongo_paper_trades(5, strategy_id=strategy_id):
            if t.get("status") == "open" and t.get("entryPx") is not None:
                con.execute(
                    """
                    INSERT INTO paper_trades (
                      strategy_id, status, side, symbol, strike, expiry, lot,
                      entry_ts, entry_px, entry_spot, entry_weight_up, entry_reason,
                      peak_px, meta
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        strategy_id,
                        "open",
                        t.get("side") or "CE",
                        t.get("symbol") or "ATM-CE",
                        t.get("strike"),
                        t.get("expiry"),
                        int(t.get("lot") or LOT_SIZE),
                        t.get("entryTs"),
                        float(t["entryPx"]),
                        t.get("entrySpot"),
                        t.get("entryWeightUp"),
                        t.get("entryReason"),
                        float(t.get("peakPx") or t["entryPx"]),
                        json.dumps({"restoredFrom": "mongo"}),
                    ),
                )
                con.commit()
                return con.execute(
                    "SELECT * FROM paper_trades WHERE status='open' AND strategy_id=? ORDER BY id DESC LIMIT 1",
                    (strategy_id,),
                ).fetchone()
    except Exception:  # noqa: BLE001
        pass
    return None


def _row_dict(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    return {
        "id": d["id"],
        "strategyId": d.get("strategy_id") or "decline",
        "status": d["status"],
        "side": d["side"],
        "symbol": d["symbol"],
        "strike": d["strike"],
        "expiry": d["expiry"],
        "lot": d["lot"],
        "entryTs": d["entry_ts"],
        "entryPx": d["entry_px"],
        "entrySpot": d["entry_spot"],
        "entryWeightUp": d["entry_weight_up"],
        "entryReason": d["entry_reason"],
        "exitTs": d["exit_ts"],
        "exitPx": d["exit_px"],
        "exitSpot": d["exit_spot"],
        "exitWeightUp": d["exit_weight_up"],
        "exitReason": d["exit_reason"],
        "peakPx": d["peak_px"],
        "pnlRs": d["pnl_rs"],
        "pnlPct": d["pnl_pct"],
        "marginRs": round(float(d["entry_px"]) * int(d["lot"]), 2)
        if d["entry_px"] is not None
        else None,
    }


def _wallet_for(
    trades: list[dict[str, Any]],
    *,
    mark_ltp: float | None = None,
) -> dict[str, Any]:
    """₹1L starting book: equity = start + realized + unrealized MTM."""
    start = float(PAPER_STARTING_CAPITAL_RS)
    closed = [t for t in trades if t.get("status") == "closed"]
    open_t = next((t for t in trades if t.get("status") == "open"), None)
    realized = round(
        sum(float(t["pnlRs"]) for t in closed if t.get("pnlRs") is not None), 2
    )
    unrealized = 0.0
    open_margin = None
    mark_source = None
    if open_t and open_t.get("entryPx") is not None:
        lot = int(open_t.get("lot") or LOT_SIZE)
        entry = float(open_t["entryPx"])
        open_margin = round(entry * lot, 2)
        if mark_ltp is not None:
            mark = float(mark_ltp)
            mark_source = "ltp"
        elif open_t.get("peakPx") is not None:
            mark = float(open_t["peakPx"])
            mark_source = "peak"
        else:
            mark = entry
            mark_source = "entry"
        unrealized = round((mark - entry) * lot, 2)
    equity = round(start + realized + unrealized, 2)
    cash = round(start + realized - (open_margin or 0.0), 2)
    return {
        "startingCapitalRs": start,
        "realizedPnlRs": realized,
        "unrealizedPnlRs": unrealized,
        "openMarginRs": open_margin,
        "cashRs": cash,
        "equityRs": equity,
        "returnPct": round((equity / start - 1.0) * 100.0, 2) if start else 0.0,
        "markSource": mark_source,
    }


def _portfolio_wallet(buckets: list[dict[str, Any]]) -> dict[str, Any]:
    wallets = [(b.get("summary") or {}).get("wallet") or {} for b in buckets]
    n = max(len(wallets), 1)
    start = round(sum(float(w.get("startingCapitalRs") or 0) for w in wallets), 2)
    if start <= 0:
        start = round(PAPER_STARTING_CAPITAL_RS * n, 2)
    realized = round(sum(float(w.get("realizedPnlRs") or 0) for w in wallets), 2)
    unrealized = round(sum(float(w.get("unrealizedPnlRs") or 0) for w in wallets), 2)
    equity = round(sum(float(w.get("equityRs") or PAPER_STARTING_CAPITAL_RS) for w in wallets), 2)
    cash = round(sum(float(w.get("cashRs") or 0) for w in wallets), 2)
    return {
        "books": len(wallets),
        "startingCapitalRsPerBook": PAPER_STARTING_CAPITAL_RS,
        "startingCapitalRs": start,
        "realizedPnlRs": realized,
        "unrealizedPnlRs": unrealized,
        "cashRs": cash,
        "equityRs": equity,
        "returnPct": round((equity / start - 1.0) * 100.0, 2) if start else 0.0,
        "mongoReady": _mongo_ready_flag(),
        "storage": _storage_label(),
    }


def _summary_for(
    strategy_id: str,
    trades: list[dict[str, Any]],
    *,
    mark_ltp: float | None = None,
) -> dict[str, Any]:
    meta = STRATEGIES.get(strategy_id) or {
        "id": strategy_id,
        "label": strategy_id,
        "entry": "—",
        "exit": "—",
        "lot": LOT_SIZE,
        "side": "ATM CE long",
    }
    scoped = [t for t in trades if t.get("strategyId") == strategy_id]
    closed = [t for t in scoped if t.get("status") == "closed"]
    open_t = next((t for t in scoped if t.get("status") == "open"), None)
    pnls = [float(t["pnlRs"]) for t in closed if t.get("pnlRs") is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wallet = _wallet_for(scoped, mark_ltp=mark_ltp)
    return {
        "strategy": {
            "id": meta["id"],
            "label": meta.get("label") or meta["id"],
            "tf": "5m",
            "entry": meta.get("entry"),
            "exit": meta.get("exit"),
            "entryMode": meta.get("entryMode"),
            "exitMode": meta.get("exitMode"),
            "lot": meta.get("lot") or LOT_SIZE,
            "side": meta.get("side") or "ATM CE long",
        },
        "open": open_t,
        "closedCount": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "netPnlRs": round(sum(pnls), 2) if pnls else 0.0,
        "avgMarginRs": round(
            sum(float(t["entryPx"]) * LOT_SIZE for t in closed) / len(closed), 2
        )
        if closed
        else None,
        "wallet": wallet,
        "storage": _storage_label(),
        "mongoReady": _mongo_ready_flag(),
    }


def _trades_for_strategy(
    con: sqlite3.Connection, strategy_id: str, limit: int
) -> list[dict[str, Any]]:
    local = [
        _row_dict(r)
        for r in con.execute(
            "SELECT * FROM paper_trades WHERE strategy_id=? ORDER BY id DESC LIMIT ?",
            (strategy_id, limit),
        ).fetchall()
    ]
    try:
        from .mongo_paper import list_mongo_paper_trades, sync_sqlite_trades_to_mongo

        if local:
            sync_sqlite_trades_to_mongo(local)
        remote = list_mongo_paper_trades(limit, strategy_id=strategy_id)
        if remote:
            by_key: dict[str, dict[str, Any]] = {}
            for t in remote + local:
                key = f"{t.get('strategyId')}|{t.get('entryTs')}|{t.get('symbol')}"
                prev = by_key.get(key)
                if prev is None:
                    by_key[key] = t
                elif t.get("status") == "closed" and prev.get("status") != "closed":
                    by_key[key] = t
            merged = sorted(
                by_key.values(),
                key=lambda x: str(x.get("entryTs") or ""),
                reverse=True,
            )
            return merged[:limit]
    except Exception:  # noqa: BLE001
        pass
    return local


def list_paper_trades(
    limit: int = 50, *, strategy_id: str | None = None
) -> list[dict[str, Any]]:
    with _lock:
        con = _conn()
        try:
            if strategy_id:
                return _trades_for_strategy(con, strategy_id, limit)
            out: list[dict[str, Any]] = []
            for sid in STRATEGIES:
                out.extend(_trades_for_strategy(con, sid, limit))
            out.sort(key=lambda x: str(x.get("entryTs") or ""), reverse=True)
            return out[:limit]
        finally:
            con.close()


def paper_trade_summary(strategy_id: str = "decline") -> dict[str, Any]:
    trades = list_paper_trades(200, strategy_id=strategy_id)
    return _summary_for(strategy_id, trades)


def _paper_live_ltp(con: sqlite3.Connection) -> dict[str, Any]:
    """Cached 2s LTP for ATM + each open position strike (Fyers)."""
    from .fyers_auth import get_access_token
    from .fyers_quotes import fetch_fyers_symbol_ltp

    if not is_live_data_window() or not get_access_token():
        return {}
    now = time.time()
    cached = _LIVE_LTP_CACHE.get("payload")
    if cached and now - float(_LIVE_LTP_CACHE.get("ts") or 0) < _LIVE_LTP_TTL_S:
        return dict(cached)

    atm = _atm_ce_quote()
    positions: dict[str, Any] = {}
    for sid in STRATEGIES:
        row = con.execute(
            """
            SELECT symbol, strike, entry_px FROM paper_trades
            WHERE status='open' AND strategy_id=? ORDER BY id DESC LIMIT 1
            """,
            (sid,),
        ).fetchone()
        if not row:
            continue
        sym = str(row["symbol"] or "")
        ltp = fetch_fyers_symbol_ltp(sym) if sym.startswith("NSE:") else None
        if ltp is None and atm and row["strike"] is not None and atm.get("strike") is not None:
            if float(row["strike"]) == float(atm["strike"]):
                ltp = atm.get("ltp")
        entry = float(row["entry_px"])
        positions[sid] = {
            "symbol": sym,
            "strike": row["strike"],
            "ltp": ltp,
            "entryPx": entry,
            "pnlRs": round((float(ltp) - entry) * LOT_SIZE, 2) if ltp is not None else None,
            "pnlPct": round((float(ltp) / entry - 1.0) * 100.0, 2) if ltp is not None and entry else None,
        }

    payload = {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "atm": atm,
        "positions": positions,
        "source": (atm or {}).get("source") or ("fyers" if positions else None),
    }
    _LIVE_LTP_CACHE["ts"] = now
    _LIVE_LTP_CACHE["payload"] = payload
    return payload


def maybe_evaluate_paper_trades() -> None:
    """Throttled strategy evaluation (entry/exit) — not on every LTP poll."""
    global _LAST_EVAL_TS
    from .fyers_auth import get_access_token

    if not is_live_data_window() or not get_access_token():
        return
    now = time.time()
    if now - _LAST_EVAL_TS < _EVAL_MIN_S:
        return
    _LAST_EVAL_TS = now
    tick_paper_trades(force=False)


def paper_entry_signal() -> dict[str, Any]:
    """Explain why decline/tsl did or did not enter (5m weightUp streaks)."""
    rows = _chrono_weight_up(8)
    vals = [float(r["weightUp"]) for r in rows]
    rising = _rising_streak(vals, ENTRY_RISE_BARS)
    falling = _falling_streak(vals, EXIT_DECLINE_BARS)
    if rising:
        hint = "Entry gate open — three consecutive 5m weightUp increases."
    elif len(vals) < ENTRY_RISE_BARS + 1:
        need = ENTRY_RISE_BARS + 1 - len(vals)
        hint = f"Collecting history — need {need} more 5m snapshot(s) for a rising ×3 streak."
    else:
        hint = (
            "Waiting for 3 straight 5m weightUp increases. "
            "Bullish adv/dec alone does not enter — weightUp must keep rising each 5m bar."
        )
    return {
        "weightUpSeries": vals,
        "bucketLabels": [r.get("t") for r in rows],
        "rising3": rising,
        "falling4": falling,
        "entryHint": hint,
        "barTf": "5m",
    }


def paper_trades_board(
    limit_per: int = 40, *, mark_ltp: float | None = None
) -> dict[str, Any]:
    with _lock:
        con = _conn()
        try:
            board = _board_unlocked(con, limit_per, mark_ltp=mark_ltp)
            board["signal"] = paper_entry_signal()
            live = _paper_live_ltp(con)
            if live:
                board["liveLtp"] = live
                # Attach mark to open rows for the trade log UI.
                pos_map = live.get("positions") or {}
                for bucket in board.get("buckets") or []:
                    sid = bucket.get("strategyId")
                    mark = pos_map.get(sid) if sid else None
                    if mark and bucket.get("summary"):
                        bucket["summary"]["markLtp"] = mark.get("ltp")
                        bucket["summary"]["markPnlRs"] = mark.get("pnlRs")
                    for tr in bucket.get("trades") or []:
                        if tr.get("status") == "open" and sid in pos_map:
                            tr["markLtp"] = pos_map[sid].get("ltp")
                            tr["markPnlRs"] = pos_map[sid].get("pnlRs")
                            tr["markPnlPct"] = pos_map[sid].get("pnlPct")
            return board
        finally:
            con.close()


def _exit_reason(
    mode: ExitMode,
    vals: list[float],
    *,
    entry_px: float,
    live: float,
    peak: float,
    cross_diff_pp: float | None = None,
) -> str | None:
    if mode == "cross":
        if cross_diff_pp is not None and cross_diff_pp < 0:
            return f"cross.diffPp < 0 ({cross_diff_pp:.3f})"
        return None
    ret_pct = (live / entry_px - 1.0) * 100.0
    if mode == "decline":
        if _falling_streak(vals, EXIT_DECLINE_BARS):
            return f"weightUp falling ×{EXIT_DECLINE_BARS}"
        return None
    if ret_pct <= SL_PCT:
        return f"SL {SL_PCT}%"
    armed = ret_pct >= TSL_ARM_PCT or peak >= entry_px * (1 + TSL_ARM_PCT / 100.0)
    if armed and live <= peak * (1 - TSL_PCT / 100.0):
        return f"TSL {TSL_PCT}% after +{TSL_ARM_PCT}%"
    return None


def _latest_cross_diff_pp() -> float | None:
    """Read live cross.diffPp from the latest Nifty board cache (if any)."""
    try:
        from .nifty_board import _BOARD_CACHE

        payload = _BOARD_CACHE.get("payload") or {}
        cross = (payload.get("marketSync") or {}).get("cross") or {}
        v = cross.get("diffPp")
        return float(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def _should_enter(
    entry_mode: EntryMode,
    *,
    rising: bool,
    cross_diff_pp: float | None,
) -> tuple[bool, str | None]:
    if entry_mode == "cross_gt_0":
        if cross_diff_pp is not None and cross_diff_pp > 0:
            return True, f"cross.diffPp > 0 ({cross_diff_pp:.3f})"
        return False, None
    if rising:
        return True, f"weightUp rising ×{ENTRY_RISE_BARS}"
    return False, None


def tick_paper_trades(
    *,
    force: bool = False,
    cross_diff_pp: float | None = None,
) -> dict[str, Any]:
    """Evaluate entry/exit for every registered strategy."""
    with _lock:
        con = _conn()
        try:
            if not force and not is_live_data_window():
                board = _board_unlocked(con)
                return {"ok": True, "skipped": "outside_hours", **board}

            if cross_diff_pp is None:
                cross_diff_pp = _latest_cross_diff_pp()

            chrono = _chrono_weight_up(12)
            vals = [float(r["weightUp"]) for r in chrono]
            quote = _atm_ce_quote()
            events: list[str] = []

            if not quote or quote.get("ltp") is None:
                board = _board_unlocked(con)
                return {
                    "ok": False,
                    "error": (quote or {}).get("error")
                    or "ATM CE quote unavailable (need Fyers)",
                    "crossDiffPp": cross_diff_pp,
                    **board,
                }

            now_iso = datetime.now(timezone.utc).isoformat()
            w_now = vals[-1] if vals else None
            rising = _rising_streak(vals, ENTRY_RISE_BARS)
            touched: list[int] = []

            for sid, meta in STRATEGIES.items():
                entry_mode: EntryMode = meta.get("entryMode") or "weight_up_rise"
                mode: ExitMode = meta.get("exitMode") or "decline"
                open_row = _open_trade(con, sid)

                if open_row is None:
                    enter, entry_reason = _should_enter(
                        entry_mode, rising=rising, cross_diff_pp=cross_diff_pp
                    )
                    if enter and entry_reason:
                        cur = con.execute(
                            """
                            INSERT INTO paper_trades (
                              strategy_id, status, side, symbol, strike, expiry, lot,
                              entry_ts, entry_px, entry_spot, entry_weight_up, entry_reason,
                              peak_px, meta
                            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                sid,
                                "open",
                                "CE",
                                str(quote.get("symbol") or "ATM-CE"),
                                quote.get("strike"),
                                quote.get("expiry"),
                                LOT_SIZE,
                                now_iso,
                                float(quote["ltp"]),
                                quote.get("spot"),
                                w_now if entry_mode == "weight_up_rise" else cross_diff_pp,
                                entry_reason,
                                float(quote["ltp"]),
                                json.dumps(
                                    {
                                        "source": quote.get("source"),
                                        "vals": vals[-6:],
                                        "strategy": sid,
                                        "crossDiffPp": cross_diff_pp,
                                    }
                                ),
                            ),
                        )
                        touched.append(int(cur.lastrowid))
                        events.append(f"{sid}:entered")
                else:
                    entry_px = float(open_row["entry_px"])
                    peak = float(open_row["peak_px"] or entry_px)
                    live = float(quote["ltp"])
                    peak = max(peak, live)
                    con.execute(
                        "UPDATE paper_trades SET peak_px=? WHERE id=?",
                        (peak, open_row["id"]),
                    )
                    touched.append(int(open_row["id"]))
                    reason = _exit_reason(
                        mode,
                        vals,
                        entry_px=entry_px,
                        live=live,
                        peak=peak,
                        cross_diff_pp=cross_diff_pp,
                    )
                    if reason:
                        pnl_pct = (live / entry_px - 1.0) * 100.0
                        pnl_rs = (live - entry_px) * LOT_SIZE
                        con.execute(
                            """
                            UPDATE paper_trades SET
                              status='closed', exit_ts=?, exit_px=?, exit_spot=?,
                              exit_weight_up=?, exit_reason=?, peak_px=?,
                              pnl_rs=?, pnl_pct=?
                            WHERE id=?
                            """,
                            (
                                now_iso,
                                live,
                                quote.get("spot"),
                                w_now if mode != "cross" else cross_diff_pp,
                                reason,
                                peak,
                                round(pnl_rs, 2),
                                round(pnl_pct, 2),
                                open_row["id"],
                            ),
                        )
                        events.append(f"{sid}:exited:{reason}")

            con.commit()
            for tid in touched:
                _sync_trade_to_mongo(con, tid)

            mark = float(quote["ltp"]) if quote.get("ltp") is not None else None
            board = _board_unlocked(con, mark_ltp=mark)
            return {
                "ok": True,
                "events": events,
                "quote": quote,
                "weightUpSeries": vals[-8:],
                "rising3": rising,
                "falling4": _falling_streak(vals, EXIT_DECLINE_BARS),
                "crossDiffPp": cross_diff_pp,
                "liveDataWindow": is_live_data_window(),
                **board,
            }
        finally:
            con.close()


def _board_unlocked(
    con: sqlite3.Connection,
    limit_per: int = 40,
    *,
    mark_ltp: float | None = None,
) -> dict[str, Any]:
    buckets = []
    for sid in STRATEGIES:
        trades = _trades_for_strategy(con, sid, limit_per)
        buckets.append(
            {
                "strategyId": sid,
                "summary": _summary_for(sid, trades, mark_ltp=mark_ltp),
                "trades": trades,
            }
        )
    first = buckets[0] if buckets else None
    return {
        "strategies": list_strategies(),
        "buckets": buckets,
        "wallet": _portfolio_wallet(buckets),
        "summary": first["summary"] if first else None,
        "trades": first["trades"] if first else [],
        "storage": _storage_label(),
        "mongoReady": _mongo_ready_flag(),
    }
