"""Nifty ATM paper trades — multi-strategy buckets.

Strategies (extensible via STRATEGIES dict):
    decline   — entry weightUp rising ×3; exit weightUp falling ×4
    tsl       — entry weightUp rising ×3; exit weightUp falling ×3
    cross     — entry marketSync cross.diffPp > 0; exit when < 0
    vwap_hma15 — entry VWAP rising + close > HMA(46) on Nifty 15m; exit reverse
    adx_obv5m_itm3 — entry ADX trend cross + OBV cross on ITM3 Nifty option 5m; exit reverse
    stoch_rsi1m_itm3 — entry STOCH cross + RSI cross on ITM3 Nifty option 1m; exit reverse

Short mode (`side=short`) uses ATM PE with reversed signals:
    decline   — entry weightUp falling ×3; exit weightUp rising ×4
    tsl       — entry weightUp falling ×3; exit weightUp rising ×3
    cross     — entry marketSync cross.diffPp < 0; exit when > 0
    vwap_hma15 — entry VWAP falling + close < HMA(46) on Nifty 15m; exit reverse
    adx_obv5m_itm3 — entry ADX trend cross + OBV cross on ITM3 Nifty option 5m; exit reverse
    stoch_rsi1m_itm3 — entry STOCH cross + RSI cross on ITM3 Nifty option 1m; exit reverse

Prices: Fyers optionchain ATM CE (prefer after-hours too when token live).
Storage: SQLite locally + MongoDB Atlas dual-write when SAINT_MONGODB_URI is set
(Render should use Mongo so trades survive free-tier disk wipes).
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd

from .config import settings
from .session import is_live_data_window

IST = ZoneInfo("Asia/Kolkata")
LOT_SIZE = 65
AUTO_EXIT_HOUR = 15
AUTO_EXIT_MINUTE = 15
ENTRY_RISE_BARS = 3
EXIT_DECLINE_BARS = 4
EXIT_TREND_BARS = 3
SL_PCT = -20.0
TSL_PCT = 10.0
TSL_ARM_PCT = 15.0
# Each strategy book starts with ₹1L so monthly return is comparable side-by-side.
PAPER_STARTING_CAPITAL_RS = 100_000.0

ExitMode = Literal[
    "decline",
    "trend",
    "cross",
    "vwap_hma_15m",
    "adx_obv_5m_itm3",
    "stoch_rsi_1m_itm3",
]
EntryMode = Literal[
    "weight_up_rise",
    "cross_gt_0",
    "vwap_hma_15m",
    "adx_obv_5m_itm3",
    "stoch_rsi_1m_itm3",
]
PaperSide = Literal["long", "short"]
OptionLeg = Literal["CE", "PE"]
VwapHmaSignal = dict[str, Any]

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
        "label": "Breadth trend x3",
        "entry": f"weightUp rising ×{ENTRY_RISE_BARS}",
        "exit": f"weightUp falling ×{EXIT_TREND_BARS}",
        "entryMode": "weight_up_rise",
        "exitMode": "trend",
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
        "tf": "5m",
    },
    "vwap_hma15": {
        "id": "vwap_hma15",
        "label": "VWAP + HMA 46 (15m)",
        "entry": "9:15 Nifty ATM option VWAP rising and close > HMA46 (15m)",
        "exit": "same option VWAP falling and close < HMA46 (15m)",
        "entryMode": "vwap_hma_15m",
        "exitMode": "vwap_hma_15m",
        "lot": LOT_SIZE,
        "side": "ATM CE long",
        "tf": "15m",
    },
    "adx_obv5m_itm3": {
        "id": "adx_obv5m_itm3",
        "label": "ADX + OBV (5m, ITM3)",
        "entry": "ITM3 option ADX>20 with +DI cross above -DI and OBV cross above OBV EMA20",
        "exit": "same option +DI cross below -DI or ADX<18 or OBV cross below OBV EMA20",
        "entryMode": "adx_obv_5m_itm3",
        "exitMode": "adx_obv_5m_itm3",
        "lot": LOT_SIZE,
        "side": "ITM3 CE long",
        "tf": "5m",
    },
    "stoch_rsi1m_itm3": {
        "id": "stoch_rsi1m_itm3",
        "label": "STOCH + RSI (1m, ITM3)",
        "entry": "ITM3 option STOCH K cross above D with K<40 and RSI cross above 55",
        "exit": "same option STOCH K cross below D with K>60 or RSI cross below 45",
        "entryMode": "stoch_rsi_1m_itm3",
        "exitMode": "stoch_rsi_1m_itm3",
        "lot": LOT_SIZE,
        "side": "ITM3 CE long",
        "tf": "1m",
    },
}

_lock = Lock()
_LIVE_LTP_CACHE: dict[str, Any] = {"ts": 0.0, "payload": None}
_LIVE_LTP_TTL_S = 2.0
_LAST_EVAL_TS = 0.0
_EVAL_MIN_S = 60.0


def list_strategies() -> list[dict[str, Any]]:
    return [dict(s) for s in STRATEGIES.values()]


def _normalize_side(side: str | None) -> PaperSide:
    return "short" if str(side or "").strip().lower() == "short" else "long"


def _option_leg_for_side(side: PaperSide) -> OptionLeg:
    return "PE" if side == "short" else "CE"


def _time_exit_due(entry_ts: str | None, now_ist: datetime) -> bool:
    """Universal fallback: auto-close all open trades at 15:15 IST.

    If a prior day close was missed (no tick near 15:15), close at the first
    subsequent tick.
    """
    try:
        ent = pd.Timestamp(entry_ts) if entry_ts else None
    except Exception:  # noqa: BLE001
        ent = None
    if ent is None:
        return False
    try:
        ent_ist = ent.tz_convert(IST) if ent.tzinfo is not None else ent.tz_localize(IST)
    except Exception:  # noqa: BLE001
        ent_ist = ent
    if ent_ist.date() < now_ist.date():
        return True
    cutoff = now_ist.replace(
        hour=AUTO_EXIT_HOUR,
        minute=AUTO_EXIT_MINUTE,
        second=0,
        microsecond=0,
    )
    return now_ist >= cutoff


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


def _atm_option_quote(leg: OptionLeg = "CE") -> dict[str, Any] | None:
    """ATM option LTP (CE/PE) — prefer Fyers (works after hours when token is live)."""
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
        ltp_key = "peLtp" if leg == "PE" else "ceLtp"
        sym_key = "peSymbol" if leg == "PE" else "ceSymbol"
        ltp = hit.get(ltp_key)
        if ltp is None:
            return None
        strike_f = float(atm) if atm is not None else float(hit.get("strike") or 0)
        return {
            "symbol": hit.get(sym_key)
            or hit.get("symbol")
            or f"NSE:NIFTY-ATM-{int(strike_f)}{leg}",
            "strike": strike_f,
            "expiry": chain.get("expiry") or wing.get("expiry"),
            "ltp": float(ltp),
            "spot": chain.get("spot") or wing.get("spot"),
            "source": chain.get("source") or "fyers",
            "optionLeg": leg,
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


def _wma(series: pd.Series, length: int) -> pd.Series:
    if length <= 1:
        return series.astype(float)
    denom = float(length * (length + 1) / 2.0)
    return series.rolling(length).apply(
        lambda vals: float(
            sum((index + 1) * float(vals[index]) for index in range(len(vals))) / denom
        ),
        raw=True,
    )


def _hma(series: pd.Series, length: int) -> pd.Series:
    n = max(2, int(length))
    half = max(1, n // 2)
    root = max(1, int(n**0.5))
    wma_half = _wma(series, half)
    wma_full = _wma(series, n)
    raw = 2.0 * wma_half - wma_full
    return _wma(raw, root)


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.astype(float).ewm(span=max(1, int(length)), adjust=False).mean()


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


def _obv(df: pd.DataFrame) -> pd.Series:
    vol = df["Volume"].fillna(0)
    direction = pd.Series((df["Close"].diff().fillna(0)).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)), index=df.index)
    return (direction * vol).fillna(0).cumsum()


def _adx(df: pd.DataFrame, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    high, low = df["High"].astype(float), df["Low"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_mask = (up_move > down_move) & (up_move > 0)
    minus_mask = (down_move > up_move) & (down_move > 0)
    plus_dm.loc[plus_mask] = up_move.loc[plus_mask]
    minus_dm.loc[minus_mask] = down_move.loc[minus_mask]

    prev_close = df["Close"].shift(1).astype(float)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / max(1, int(length)), adjust=False).mean().replace(0, pd.NA)
    plus_di = 100 * plus_dm.ewm(alpha=1 / max(1, int(length)), adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / max(1, int(length)), adjust=False).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)).fillna(0)
    adx_v = dx.ewm(alpha=1 / max(1, int(length)), adjust=False).mean()
    return adx_v, plus_di, minus_di


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.astype(float).diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    up_ema = up.ewm(alpha=1 / max(1, int(length)), adjust=False).mean()
    down_ema = down.ewm(alpha=1 / max(1, int(length)), adjust=False).mean().replace(0, pd.NA)
    rs = up_ema / down_ema
    return 100 - (100 / (1 + rs))


def _stochastic(df: pd.DataFrame, k_len: int = 14, d_len: int = 3) -> tuple[pd.Series, pd.Series]:
    low_n = df["Low"].rolling(k_len).min()
    high_n = df["High"].rolling(k_len).max()
    k = 100 * (df["Close"] - low_n) / (high_n - low_n).replace(0, pd.NA)
    d = k.rolling(d_len).mean()
    return k, d


def _current_nifty_915_contract(side: PaperSide, *, itm_steps: int = 0) -> dict[str, Any] | None:
    try:
        from .prices import fetch_intraday_15m
        from .nifty_chain import atm_wing_board, fetch_nifty_option_chain

        df = fetch_intraday_15m("NIFTY", force=False)
        if df is None or df.empty:
            return None
        day = pd.Timestamp(df.index.max()).date()
        day_rows = df.loc[df.index.date == day]
        if day_rows.empty:
            return None
        open_row = day_rows.iloc[0]
        anchor_spot = float(open_row.get("Open") or open_row.get("Close") or 0)
        if anchor_spot <= 0:
            return None
        leg = _option_leg_for_side(side)
        strike = int(round(anchor_spot / 50.0) * 50)
        if itm_steps > 0:
            offset = int(itm_steps) * 50
            strike = strike - offset if leg == "CE" else strike + offset
        chain = fetch_nifty_option_chain(force=True, prefer_fyers_after_hours=True) or {}
        board = atm_wing_board(chain, wing=12)
        hit = None
        for row in board.get("rows") or []:
            try:
                if int(round(float(row.get("strike") or 0))) == strike:
                    hit = row
                    break
            except Exception:  # noqa: BLE001
                continue
        if hit is None:
            return None
        symbol = hit.get("peSymbol") if leg == "PE" else hit.get("ceSymbol")
        if not symbol:
            return None
        return {
            "symbol": str(symbol),
            "strike": strike,
            "expiry": chain.get("expiry"),
            "spotOpen915": anchor_spot,
            "spotSession": float(chain.get("spot") or anchor_spot),
            "optionLeg": leg,
        }
    except Exception:  # noqa: BLE001
        return None


def _option_contract_15m(symbol: str) -> pd.DataFrame:
    from .nifty_option_sync import _history, _latest_cache, _normalize_ohlcv

    cached, _ = _latest_cache(symbol, "15", settings.nifty_option_15m_dir.resolve())
    if cached.empty:
        try:
            fresh = _history(symbol, "15", datetime.now(IST).date() - timedelta(days=7), datetime.now(IST).date())
            return _normalize_ohlcv(fresh)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    try:
        fresh = _history(
            symbol,
            "15",
            pd.Timestamp(cached.index.max()).date() - timedelta(days=1),
            datetime.now(IST).date(),
        )
        return _normalize_ohlcv(pd.concat([cached, fresh]))
    except Exception:  # noqa: BLE001
        return cached


def _option_contract_1m(symbol: str) -> pd.DataFrame:
    from .nifty_option_sync import _history, _latest_cache, _normalize_ohlcv

    cached, _ = _latest_cache(symbol, "1", settings.nifty_option_1m_dir.resolve())
    if cached.empty:
        try:
            fresh = _history(symbol, "1", datetime.now(IST).date() - timedelta(days=7), datetime.now(IST).date())
            return _normalize_ohlcv(fresh)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    try:
        fresh = _history(
            symbol,
            "1",
            pd.Timestamp(cached.index.max()).date() - timedelta(days=1),
            datetime.now(IST).date(),
        )
        return _normalize_ohlcv(pd.concat([cached, fresh]))
    except Exception:  # noqa: BLE001
        return cached


def _option_signal_quote(
    sig: VwapHmaSignal | None,
    fallback_quote: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    sig = sig or {}
    symbol = str(sig.get("symbol") or "").strip()
    if not symbol:
        return fallback_quote
    ltp = None
    try:
        from .fyers_quotes import fetch_fyers_symbol_ltp

        ltp = fetch_fyers_symbol_ltp(symbol)
    except Exception:  # noqa: BLE001
        ltp = None
    if ltp is None:
        try:
            ltp = float(sig.get("close")) if sig.get("close") is not None else None
        except Exception:  # noqa: BLE001
            ltp = None
    if ltp is None and fallback_quote is None:
        return None
    return {
        "symbol": symbol,
        "strike": sig.get("strike") if sig.get("strike") is not None else (fallback_quote or {}).get("strike"),
        "expiry": sig.get("expiry") if sig.get("expiry") is not None else (fallback_quote or {}).get("expiry"),
        "ltp": float(ltp) if ltp is not None else (fallback_quote or {}).get("ltp"),
        "spot": sig.get("spotSession") or (fallback_quote or {}).get("spot"),
        "source": "fyers" if ltp is not None else (fallback_quote or {}).get("source"),
        "optionLeg": sig.get("optionLeg") or (fallback_quote or {}).get("optionLeg"),
    }


def _option_vwap_hma_15m_signal(
    side: PaperSide,
    *,
    symbol: str | None = None,
    strike: float | int | None = None,
    expiry: str | None = None,
    hma_length: int = 24,
) -> VwapHmaSignal:
    """Compute VWAP+HMA46 signal from the monitored ATM option's 15m bars."""
    contract = (
        {
            "symbol": symbol,
            "strike": strike,
            "expiry": expiry,
            "spotOpen915": None,
            "spotSession": None,
            "optionLeg": _option_leg_for_side(side),
        }
        if symbol
        else _current_nifty_915_contract(side)
    )
    if not contract or not contract.get("symbol"):
        return {"ready": False}
    df = _option_contract_15m(str(contract["symbol"]))
    if df is None or df.empty:
        return {"ready": False, "symbol": contract.get("symbol"), "strike": contract.get("strike")}
    work = df.copy().dropna(subset=["Close", "High", "Low", "Volume"])
    if work.empty:
        return {"ready": False, "symbol": contract.get("symbol"), "strike": contract.get("strike")}

    today = pd.Timestamp(work.index.max()).date()
    prior = work.loc[work.index.date < today]
    if len(prior) < hma_length:
        return {
            "ready": False,
            "symbol": contract.get("symbol"),
            "strike": contract.get("strike"),
            "reason": f"insufficient_prior_bars<{hma_length}",
            "priorBars": int(len(prior)),
        }

    src = (work["High"] + work["Low"] + work["Close"]) / 3.0
    vol = work["Volume"].astype(float)
    if float(vol.fillna(0.0).sum()) <= 0:
        vol = pd.Series(1.0, index=work.index)
    day_key = pd.Series(work.index.date, index=work.index)
    cum_src_vol = (src * vol).groupby(day_key).cumsum()
    cum_vol = vol.groupby(day_key).cumsum()
    vwap = cum_src_vol / cum_vol.replace(0.0, pd.NA)

    hma_line = _hma(work["Close"].astype(float), hma_length)
    work = work.assign(vwap=vwap, hma=hma_line).dropna(subset=["vwap", "hma", "Close"])
    if len(work) < 2:
        return {"ready": False, "symbol": contract.get("symbol"), "strike": contract.get("strike")}

    work = work.loc[work.index.date == today]
    if len(work) < 2:
        return {"ready": False, "symbol": contract.get("symbol"), "strike": contract.get("strike"), "reason": "no_today_window"}

    curr = work.iloc[-1]
    prev = work.iloc[-2]
    current_vwap = float(curr["vwap"])
    prev_vwap = float(prev["vwap"])
    close_now = float(curr["Close"])
    hma_now = float(curr["hma"])
    vwap_rising = current_vwap > prev_vwap
    vwap_falling = current_vwap < prev_vwap
    price_above_hma = close_now > hma_now
    price_below_hma = close_now < hma_now
    bullish = vwap_rising and price_above_hma
    bearish = vwap_falling and price_below_hma
    return {
        "ready": True,
        "symbol": str(contract.get("symbol") or ""),
        "strike": float(contract.get("strike") or 0),
        "expiry": contract.get("expiry"),
        "optionLeg": contract.get("optionLeg"),
        "spotOpen915": contract.get("spotOpen915"),
        "spotSession": contract.get("spotSession"),
        "currentVwap": current_vwap,
        "prevVwap": prev_vwap,
        "close": close_now,
        "hma": hma_now,
        "vwapRising": vwap_rising,
        "vwapFalling": vwap_falling,
        "priceAboveHma": price_above_hma,
        "priceBelowHma": price_below_hma,
        "bullish": bullish,
        "bearish": bearish,
    }


def _option_adx_obv_5m_itm3_signal(
    side: PaperSide,
    *,
    symbol: str | None = None,
    strike: float | int | None = None,
    expiry: str | None = None,
) -> dict[str, Any]:
    """Compute ADX+OBV combo signal from ITM3 option 5m bars."""
    contract = (
        {
            "symbol": symbol,
            "strike": strike,
            "expiry": expiry,
            "spotOpen915": None,
            "spotSession": None,
            "optionLeg": _option_leg_for_side(side),
        }
        if symbol
        else _current_nifty_915_contract(side, itm_steps=3)
    )
    if not contract or not contract.get("symbol"):
        return {"ready": False}

    from .nifty_option_sync import _history, _latest_cache, _normalize_ohlcv

    symbol_key = str(contract["symbol"])
    cached, _ = _latest_cache(symbol_key, "5", settings.nifty_option_5m_dir.resolve())
    if cached.empty:
        try:
            fresh = _history(
                symbol_key,
                "5",
                datetime.now(IST).date() - timedelta(days=7),
                datetime.now(IST).date(),
            )
            bars = _normalize_ohlcv(fresh)
        except Exception:  # noqa: BLE001
            bars = pd.DataFrame()
    else:
        try:
            fresh = _history(
                symbol_key,
                "5",
                pd.Timestamp(cached.index.max()).date() - timedelta(days=1),
                datetime.now(IST).date(),
            )
            bars = _normalize_ohlcv(pd.concat([cached, fresh]))
        except Exception:  # noqa: BLE001
            bars = cached

    if bars.empty:
        return {"ready": False, "symbol": symbol_key, "strike": contract.get("strike")}

    work = bars.copy().dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if work.empty:
        return {"ready": False, "symbol": symbol_key, "strike": contract.get("strike")}

    today = pd.Timestamp(work.index.max()).date()
    prior = work.loc[work.index.date < today]
    if len(prior) < 30:
        return {
            "ready": False,
            "symbol": symbol_key,
            "strike": contract.get("strike"),
            "reason": "insufficient_prior_bars<30",
            "priorBars": int(len(prior)),
        }

    adx_v, plus_di, minus_di = _adx(work, 14)
    obv = _obv(work)
    obv_ema20 = _ema(obv, 20)

    adx_entry = ((adx_v > 20) & _cross_up(plus_di, minus_di)).fillna(False)
    adx_exit = (_cross_down(plus_di, minus_di) | (adx_v < 18)).fillna(False)
    obv_entry = _cross_up(obv, obv_ema20).fillna(False)
    obv_exit = _cross_down(obv, obv_ema20).fillna(False)

    day = work.loc[work.index.date == today].copy()
    if len(day) < 2:
        return {
            "ready": False,
            "symbol": symbol_key,
            "strike": contract.get("strike"),
            "reason": "no_today_window",
        }

    idx = day.index[-1]
    bullish = bool(adx_entry.loc[idx] and obv_entry.loc[idx])
    bearish = bool(adx_exit.loc[idx] or obv_exit.loc[idx])
    return {
        "ready": True,
        "symbol": symbol_key,
        "strike": float(contract.get("strike") or 0),
        "expiry": contract.get("expiry"),
        "optionLeg": contract.get("optionLeg"),
        "spotOpen915": contract.get("spotOpen915"),
        "spotSession": contract.get("spotSession"),
        "close": float(day.iloc[-1]["Close"]),
        "adx": float(adx_v.loc[idx]) if pd.notna(adx_v.loc[idx]) else None,
        "plusDi": float(plus_di.loc[idx]) if pd.notna(plus_di.loc[idx]) else None,
        "minusDi": float(minus_di.loc[idx]) if pd.notna(minus_di.loc[idx]) else None,
        "obv": float(obv.loc[idx]) if pd.notna(obv.loc[idx]) else None,
        "obvEma20": float(obv_ema20.loc[idx]) if pd.notna(obv_ema20.loc[idx]) else None,
        "adxEntry": bool(adx_entry.loc[idx]),
        "adxExit": bool(adx_exit.loc[idx]),
        "obvEntry": bool(obv_entry.loc[idx]),
        "obvExit": bool(obv_exit.loc[idx]),
        "bullish": bullish,
        "bearish": bearish,
    }


def _option_stoch_rsi_1m_itm3_signal(
    side: PaperSide,
    *,
    symbol: str | None = None,
    strike: float | int | None = None,
    expiry: str | None = None,
) -> dict[str, Any]:
    """Compute STOCH+RSI combo signal from ITM3 option 1m bars."""
    contract = (
        {
            "symbol": symbol,
            "strike": strike,
            "expiry": expiry,
            "spotOpen915": None,
            "spotSession": None,
            "optionLeg": _option_leg_for_side(side),
        }
        if symbol
        else _current_nifty_915_contract(side, itm_steps=3)
    )
    if not contract or not contract.get("symbol"):
        return {"ready": False}

    symbol_key = str(contract["symbol"])
    bars = _option_contract_1m(symbol_key)
    if bars.empty:
        return {"ready": False, "symbol": symbol_key, "strike": contract.get("strike")}

    work = bars.copy().dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if work.empty:
        return {"ready": False, "symbol": symbol_key, "strike": contract.get("strike")}

    today = pd.Timestamp(work.index.max()).date()
    prior = work.loc[work.index.date < today]
    if len(prior) < 90:
        return {
            "ready": False,
            "symbol": symbol_key,
            "strike": contract.get("strike"),
            "reason": "insufficient_prior_bars<90",
            "priorBars": int(len(prior)),
        }

    k, d = _stochastic(work, 14, 3)
    rsi = _rsi(work["Close"], 14)

    stoch_entry = (_cross_up(k, d) & (k < 40)).fillna(False)
    stoch_exit = (_cross_down(k, d) & (k > 60)).fillna(False)
    rsi_entry = _cross_up(rsi, pd.Series(55.0, index=rsi.index)).fillna(False)
    rsi_exit = _cross_down(rsi, pd.Series(45.0, index=rsi.index)).fillna(False)

    day = work.loc[work.index.date == today].copy()
    if len(day) < 2:
        return {
            "ready": False,
            "symbol": symbol_key,
            "strike": contract.get("strike"),
            "reason": "no_today_window",
        }

    idx = day.index[-1]
    bullish = bool(stoch_entry.loc[idx] and rsi_entry.loc[idx])
    bearish = bool(stoch_exit.loc[idx] or rsi_exit.loc[idx])
    return {
        "ready": True,
        "symbol": symbol_key,
        "strike": float(contract.get("strike") or 0),
        "expiry": contract.get("expiry"),
        "optionLeg": contract.get("optionLeg"),
        "spotOpen915": contract.get("spotOpen915"),
        "spotSession": contract.get("spotSession"),
        "close": float(day.iloc[-1]["Close"]),
        "stochK": float(k.loc[idx]) if pd.notna(k.loc[idx]) else None,
        "stochD": float(d.loc[idx]) if pd.notna(d.loc[idx]) else None,
        "rsi": float(rsi.loc[idx]) if pd.notna(rsi.loc[idx]) else None,
        "stochEntry": bool(stoch_entry.loc[idx]),
        "stochExit": bool(stoch_exit.loc[idx]),
        "rsiEntry": bool(rsi_entry.loc[idx]),
        "rsiExit": bool(rsi_exit.loc[idx]),
        "bullish": bullish,
        "bearish": bearish,
    }


def _open_trade(
    con: sqlite3.Connection,
    strategy_id: str,
    side: PaperSide,
) -> sqlite3.Row | None:
    leg = _option_leg_for_side(side)
    cur = con.execute(
        "SELECT * FROM paper_trades WHERE status='open' AND strategy_id=? AND side=? ORDER BY id DESC LIMIT 1",
        (strategy_id, leg),
    )
    row = cur.fetchone()
    if row is not None:
        return row
    # Render / fresh disk: recover open from Atlas
    try:
        from .mongo_paper import list_mongo_paper_trades

        for t in list_mongo_paper_trades(5, strategy_id=strategy_id):
            if t.get("status") == "open" and t.get("entryPx") is not None:
                t_side = str(t.get("side") or "CE").upper()
                if t_side != leg:
                    continue
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
                        t_side,
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
                    "SELECT * FROM paper_trades WHERE status='open' AND strategy_id=? AND side=? ORDER BY id DESC LIMIT 1",
                    (strategy_id, leg),
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
    side: PaperSide = "long",
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
    leg = _option_leg_for_side(side)
    scoped = [
        t
        for t in trades
        if t.get("strategyId") == strategy_id and str(t.get("side") or "").upper() == leg
    ]
    closed = [t for t in scoped if t.get("status") == "closed"]
    open_t = next((t for t in scoped if t.get("status") == "open"), None)
    pnls = [float(t["pnlRs"]) for t in closed if t.get("pnlRs") is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wallet = _wallet_for(scoped, mark_ltp=mark_ltp)
    side_label = f"ATM {leg} {'short' if side == 'short' else 'long'}"
    return {
        "strategy": {
            "id": meta["id"],
            "label": meta.get("label") or meta["id"],
            "tf": meta.get("tf") or "5m",
            "entry": meta.get("entry"),
            "exit": meta.get("exit"),
            "entryMode": meta.get("entryMode"),
            "exitMode": meta.get("exitMode"),
            "lot": meta.get("lot") or LOT_SIZE,
            "side": side_label,
            "positionSide": side,
            "optionLeg": leg,
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
    con: sqlite3.Connection,
    strategy_id: str,
    limit: int,
    *,
    side: PaperSide = "long",
) -> list[dict[str, Any]]:
    leg = _option_leg_for_side(side)
    local = [
        _row_dict(r)
        for r in con.execute(
            "SELECT * FROM paper_trades WHERE strategy_id=? AND side=? ORDER BY id DESC LIMIT ?",
            (strategy_id, leg, limit),
        ).fetchall()
    ]
    try:
        from .mongo_paper import list_mongo_paper_trades, sync_sqlite_trades_to_mongo

        if local:
            sync_sqlite_trades_to_mongo(local)
        remote = list_mongo_paper_trades(limit, strategy_id=strategy_id)
        remote = [
            t
            for t in remote
            if str(t.get("side") or "").upper() == leg
        ]
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
    limit: int = 50,
    *,
    strategy_id: str | None = None,
    side: PaperSide = "long",
) -> list[dict[str, Any]]:
    side = _normalize_side(side)
    with _lock:
        con = _conn()
        try:
            if strategy_id:
                return _trades_for_strategy(con, strategy_id, limit, side=side)
            out: list[dict[str, Any]] = []
            for sid in STRATEGIES:
                out.extend(_trades_for_strategy(con, sid, limit, side=side))
            out.sort(key=lambda x: str(x.get("entryTs") or ""), reverse=True)
            return out[:limit]
        finally:
            con.close()


def paper_trade_summary(
    strategy_id: str = "decline",
    *,
    side: PaperSide = "long",
) -> dict[str, Any]:
    side = _normalize_side(side)
    trades = list_paper_trades(200, strategy_id=strategy_id, side=side)
    return _summary_for(strategy_id, trades, side=side)


def _position_mark_ltp(
    *,
    symbol: str | None,
    strike: float | int | None,
    atm: dict[str, Any] | None,
    leg: OptionLeg = "CE",
) -> float | None:
    """LTP for an open position's contract (not the current ATM quote)."""
    from .fyers_quotes import fetch_fyers_symbol_ltp

    sym = str(symbol or "")
    if sym.startswith("NSE:"):
        ltp = fetch_fyers_symbol_ltp(sym)
        if ltp is not None:
            return float(ltp)
    if atm and str(atm.get("optionLeg") or "CE").upper() == leg and strike is not None and atm.get("strike") is not None:
        if float(strike) == float(atm["strike"]) and atm.get("ltp") is not None:
            return float(atm["ltp"])
    return None


def _marks_for_open_positions(
    con: sqlite3.Connection,
    *,
    side: PaperSide = "long",
    atm: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Per-strategy mark prices for wallet MTM (each open strike, not shared ATM)."""
    side = _normalize_side(side)
    leg = _option_leg_for_side(side)
    if atm is None:
        atm = _atm_option_quote(leg)
    marks: dict[str, float] = {}
    for sid in STRATEGIES:
        row = con.execute(
            """
            SELECT symbol, strike FROM paper_trades
            WHERE status='open' AND strategy_id=? AND side=? ORDER BY id DESC LIMIT 1
            """,
            (sid, leg),
        ).fetchone()
        if not row:
            continue
        ltp = _position_mark_ltp(symbol=row["symbol"], strike=row["strike"], atm=atm, leg=leg)
        if ltp is not None:
            marks[sid] = ltp
    return marks


def _paper_live_ltp(con: sqlite3.Connection, *, side: PaperSide = "long") -> dict[str, Any]:
    """Cached 2s LTP for ATM + each open position strike (Fyers)."""
    from .fyers_auth import get_access_token

    if not is_live_data_window() or not get_access_token():
        return {}
    now = time.time()
    side = _normalize_side(side)
    leg = _option_leg_for_side(side)
    cache_key = f"payload:{side}"
    ts_key = f"ts:{side}"
    cached = _LIVE_LTP_CACHE.get(cache_key)
    if cached and now - float(_LIVE_LTP_CACHE.get(ts_key) or 0) < _LIVE_LTP_TTL_S:
        return dict(cached)

    atm = _atm_option_quote(leg)
    positions: dict[str, Any] = {}
    for sid in STRATEGIES:
        row = con.execute(
            """
            SELECT symbol, strike, entry_px FROM paper_trades
            WHERE status='open' AND strategy_id=? AND side=? ORDER BY id DESC LIMIT 1
            """,
            (sid, leg),
        ).fetchone()
        if not row:
            continue
        ltp = _position_mark_ltp(symbol=row["symbol"], strike=row["strike"], atm=atm, leg=leg)
        entry = float(row["entry_px"])
        sym = str(row["symbol"] or "")
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
        "positionSide": side,
        "optionLeg": leg,
        "atm": atm,
        "positions": positions,
        "source": (atm or {}).get("source") or ("fyers" if positions else None),
    }
    _LIVE_LTP_CACHE[ts_key] = now
    _LIVE_LTP_CACHE[cache_key] = payload
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
    tick_paper_trades(force=False, side="long")
    tick_paper_trades(force=False, side="short")


def paper_entry_signal(side: PaperSide = "long") -> dict[str, Any]:
    """Explain why decline/tsl did or did not enter (5m weightUp streaks)."""
    rows = _chrono_weight_up(8)
    vals = [float(r["weightUp"]) for r in rows]
    side = _normalize_side(side)
    rising = _rising_streak(vals, ENTRY_RISE_BARS)
    falling = _falling_streak(vals, EXIT_DECLINE_BARS)
    enter_ready = falling if side == "short" else rising
    if enter_ready:
        hint = (
            "Entry gate open — three consecutive 5m weightUp decreases."
            if side == "short"
            else "Entry gate open — three consecutive 5m weightUp increases."
        )
    elif len(vals) < ENTRY_RISE_BARS + 1:
        need = ENTRY_RISE_BARS + 1 - len(vals)
        hint = (
            f"Collecting history — need {need} more 5m snapshot(s) for a falling ×3 streak."
            if side == "short"
            else f"Collecting history — need {need} more 5m snapshot(s) for a rising ×3 streak."
        )
    else:
        hint = (
            "Waiting for 3 straight 5m weightUp decreases. "
            "Entry does not depend on positive/negative level — only trend versus prior bars."
            if side == "short"
            else "Waiting for 3 straight 5m weightUp increases. "
            "Entry does not depend on positive/negative level — only trend versus prior bars."
        )
    return {
        "positionSide": side,
        "optionLeg": _option_leg_for_side(side),
        "weightUpSeries": vals,
        "bucketLabels": [r.get("t") for r in rows],
        "rising3": rising,
        "falling4": falling,
        "entryReady": enter_ready,
        "entryHint": hint,
        "barTf": "5m",
    }


def paper_trades_board(
    limit_per: int = 40,
    *,
    side: PaperSide = "long",
    mark_ltp: float | None = None,
) -> dict[str, Any]:
    side = _normalize_side(side)
    with _lock:
        con = _conn()
        try:
            live = _paper_live_ltp(con, side=side)
            marks = {
                sid: float(pos["ltp"])
                for sid, pos in (live.get("positions") or {}).items()
                if pos.get("ltp") is not None
            }
            board = _board_unlocked(
                con,
                limit_per,
                side=side,
                mark_ltp=mark_ltp,
                marks_by_strategy=marks or None,
            )
            board["signal"] = paper_entry_signal(side)
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
    side: PaperSide,
    vwap_hma_signal: VwapHmaSignal | None,
    entry_px: float,
    live: float,
    peak: float,
    cross_diff_pp: float | None = None,
) -> str | None:
    side = _normalize_side(side)
    if mode == "cross":
        if side == "short":
            if cross_diff_pp is not None and cross_diff_pp > 0:
                return f"cross.diffPp > 0 ({cross_diff_pp:.3f})"
            return None
        if cross_diff_pp is not None and cross_diff_pp < 0:
            return f"cross.diffPp < 0 ({cross_diff_pp:.3f})"
        return None
    ret_pct = (live / entry_px - 1.0) * 100.0
    if mode == "decline":
        falling = _falling_streak(vals, EXIT_DECLINE_BARS)
        rising = _rising_streak(vals, EXIT_DECLINE_BARS)
        if (rising if side == "short" else falling):
            return (
                f"weightUp rising ×{EXIT_DECLINE_BARS}"
                if side == "short"
                else f"weightUp falling ×{EXIT_DECLINE_BARS}"
            )
        return None
    if mode == "trend":
        falling = _falling_streak(vals, EXIT_TREND_BARS)
        rising = _rising_streak(vals, EXIT_TREND_BARS)
        if (rising if side == "short" else falling):
            return (
                f"weightUp rising ×{EXIT_TREND_BARS}"
                if side == "short"
                else f"weightUp falling ×{EXIT_TREND_BARS}"
            )
        return None
    if mode == "vwap_hma_15m":
        sig = vwap_hma_signal or {}
        if not bool(sig.get("ready")):
            return None
        if bool(sig.get("bearish")):
            return "Option VWAP falling + close < HMA46"
        return None
    if mode == "adx_obv_5m_itm3":
        sig = vwap_hma_signal or {}
        if not bool(sig.get("ready")):
            return None
        if bool(sig.get("bearish")):
            return "ITM3 ADX/OBV bearish"
        return None
    if mode == "stoch_rsi_1m_itm3":
        sig = vwap_hma_signal or {}
        if not bool(sig.get("ready")):
            return None
        if bool(sig.get("bearish")):
            return "ITM3 STOCH/RSI bearish"
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
    side: PaperSide,
    rising: bool,
    falling: bool,
    cross_diff_pp: float | None,
    vwap_hma_signal: VwapHmaSignal | None,
) -> tuple[bool, str | None]:
    side = _normalize_side(side)
    if entry_mode == "vwap_hma_15m":
        sig = vwap_hma_signal or {}
        if not bool(sig.get("ready")):
            return False, None
        if bool(sig.get("bullish")):
            return True, "Option VWAP rising + close > HMA46"
        return False, None
    if entry_mode == "adx_obv_5m_itm3":
        sig = vwap_hma_signal or {}
        if not bool(sig.get("ready")):
            return False, None
        if bool(sig.get("bullish")):
            return True, "ITM3 ADX+OBV bullish"
        return False, None
    if entry_mode == "stoch_rsi_1m_itm3":
        sig = vwap_hma_signal or {}
        if not bool(sig.get("ready")):
            return False, None
        if bool(sig.get("bullish")):
            return True, "ITM3 STOCH+RSI bullish"
        return False, None
    if entry_mode == "cross_gt_0":
        if side == "short":
            if cross_diff_pp is not None and cross_diff_pp < 0:
                return True, f"cross.diffPp < 0 ({cross_diff_pp:.3f})"
            return False, None
        if cross_diff_pp is not None and cross_diff_pp > 0:
            return True, f"cross.diffPp > 0 ({cross_diff_pp:.3f})"
        return False, None
    if side == "short" and falling:
        return True, f"weightUp falling ×{ENTRY_RISE_BARS}"
    if side == "long" and rising:
        return True, f"weightUp rising ×{ENTRY_RISE_BARS}"
    return False, None


def tick_paper_trades(
    *,
    force: bool = False,
    cross_diff_pp: float | None = None,
    side: PaperSide = "long",
) -> dict[str, Any]:
    """Evaluate entry/exit for every registered strategy."""
    side = _normalize_side(side)
    leg = _option_leg_for_side(side)
    with _lock:
        con = _conn()
        try:
            if not force and not is_live_data_window():
                board = _board_unlocked(con, side=side)
                return {"ok": True, "skipped": "outside_hours", "positionSide": side, **board}

            if cross_diff_pp is None:
                cross_diff_pp = _latest_cross_diff_pp()

            chrono = _chrono_weight_up(12)
            vals = [float(r["weightUp"]) for r in chrono]
            vwap_hma_signal = _option_vwap_hma_15m_signal(side, hma_length=46)
            adx_obv_signal = _option_adx_obv_5m_itm3_signal(side)
            stoch_rsi_signal = _option_stoch_rsi_1m_itm3_signal(side)
            quote = _atm_option_quote(leg)
            events: list[str] = []

            if not quote or quote.get("ltp") is None:
                board = _board_unlocked(con, side=side)
                return {
                    "ok": False,
                    "error": (quote or {}).get("error")
                    or f"ATM {leg} quote unavailable (need Fyers)",
                    "crossDiffPp": cross_diff_pp,
                    "positionSide": side,
                    **board,
                }

            now_iso = datetime.now(timezone.utc).isoformat()
            now_ist = datetime.now(IST)
            w_now = vals[-1] if vals else None
            rising = _rising_streak(vals, ENTRY_RISE_BARS)
            falling = _falling_streak(vals, ENTRY_RISE_BARS)
            touched: list[int] = []

            for sid, meta in STRATEGIES.items():
                entry_mode: EntryMode = meta.get("entryMode") or "weight_up_rise"
                mode: ExitMode = meta.get("exitMode") or "decline"
                open_row = _open_trade(con, sid, side)

                if open_row is None:
                    strategy_signal = (
                        vwap_hma_signal
                        if entry_mode == "vwap_hma_15m"
                        else (
                            adx_obv_signal
                            if entry_mode == "adx_obv_5m_itm3"
                            else (stoch_rsi_signal if entry_mode == "stoch_rsi_1m_itm3" else None)
                        )
                    )
                    strategy_quote = (
                        _option_signal_quote(strategy_signal, quote)
                        if entry_mode in {"vwap_hma_15m", "adx_obv_5m_itm3", "stoch_rsi_1m_itm3"}
                        else quote
                    )
                    enter, entry_reason = _should_enter(
                        entry_mode,
                        side=side,
                        rising=rising,
                        falling=falling,
                        cross_diff_pp=cross_diff_pp,
                        vwap_hma_signal=strategy_signal,
                    )
                    if enter and entry_reason and strategy_quote and strategy_quote.get("ltp") is not None:
                        entry_metric = (
                            w_now
                            if entry_mode == "weight_up_rise"
                            else (
                                cross_diff_pp
                                if entry_mode == "cross_gt_0"
                                else (
                                    strategy_signal.get("currentVwap")
                                    if entry_mode == "vwap_hma_15m"
                                    else (
                                        strategy_signal.get("adx")
                                        if entry_mode == "adx_obv_5m_itm3"
                                        else strategy_signal.get("rsi")
                                    )
                                )
                            )
                        )
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
                                leg,
                                str(strategy_quote.get("symbol") or f"ATM-{leg}"),
                                strategy_quote.get("strike"),
                                strategy_quote.get("expiry"),
                                LOT_SIZE,
                                now_iso,
                                float(strategy_quote["ltp"]),
                                strategy_quote.get("spot"),
                                entry_metric,
                                entry_reason,
                                float(strategy_quote["ltp"]),
                                json.dumps(
                                    {
                                        "source": strategy_quote.get("source"),
                                        "positionSide": side,
                                        "vals": vals[-6:],
                                        "option15m": strategy_signal,
                                        "strategy": sid,
                                        "crossDiffPp": cross_diff_pp,
                                    }
                                ),
                            ),
                        )
                        touched.append(int(cur.lastrowid))
                        events.append(f"{sid}:entered")
                else:
                    strategy_signal = (
                        _option_vwap_hma_15m_signal(
                            side,
                            symbol=str(open_row["symbol"] or "").strip() or None,
                            strike=open_row["strike"],
                            expiry=open_row["expiry"],
                            hma_length=46,
                        )
                        if mode == "vwap_hma_15m"
                        else (
                            _option_adx_obv_5m_itm3_signal(
                                side,
                                symbol=str(open_row["symbol"] or "").strip() or None,
                                strike=open_row["strike"],
                                expiry=open_row["expiry"],
                            )
                            if mode == "adx_obv_5m_itm3"
                            else (
                                _option_stoch_rsi_1m_itm3_signal(
                                    side,
                                    symbol=str(open_row["symbol"] or "").strip() or None,
                                    strike=open_row["strike"],
                                    expiry=open_row["expiry"],
                                )
                                if mode == "stoch_rsi_1m_itm3"
                                else vwap_hma_signal
                            )
                        )
                    )
                    strategy_quote = _option_signal_quote(strategy_signal, quote) if mode in {"vwap_hma_15m", "adx_obv_5m_itm3", "stoch_rsi_1m_itm3"} else quote
                    entry_px = float(open_row["entry_px"])
                    peak = float(open_row["peak_px"] or entry_px)
                    pos_ltp = _position_mark_ltp(
                        symbol=open_row["symbol"],
                        strike=open_row["strike"],
                        atm=strategy_quote,
                        leg=leg,
                    )
                    live = float(pos_ltp if pos_ltp is not None else strategy_quote["ltp"])
                    peak = max(peak, live)
                    con.execute(
                        "UPDATE paper_trades SET peak_px=? WHERE id=?",
                        (peak, open_row["id"]),
                    )
                    touched.append(int(open_row["id"]))
                    reason = _exit_reason(
                        mode,
                        vals,
                        side=side,
                        vwap_hma_signal=strategy_signal,
                        entry_px=entry_px,
                        live=live,
                        peak=peak,
                        cross_diff_pp=cross_diff_pp,
                    )
                    if reason is None and _time_exit_due(str(open_row["entry_ts"]), now_ist):
                        reason = f"Time exit {AUTO_EXIT_HOUR:02d}:{AUTO_EXIT_MINUTE:02d} IST"
                    if reason:
                        exit_metric = (
                            cross_diff_pp
                            if mode == "cross"
                            else (
                                strategy_signal.get("currentVwap")
                                if mode == "vwap_hma_15m"
                                else (
                                    strategy_signal.get("adx")
                                    if mode == "adx_obv_5m_itm3"
                                    else (
                                        strategy_signal.get("rsi")
                                        if mode == "stoch_rsi_1m_itm3"
                                        else w_now
                                    )
                                )
                            )
                        )
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
                                strategy_quote.get("spot"),
                                exit_metric,
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

            marks = _marks_for_open_positions(con, side=side, atm=quote)
            board = _board_unlocked(con, side=side, marks_by_strategy=marks or None)
            return {
                "ok": True,
                "positionSide": side,
                "events": events,
                "quote": quote,
                "weightUpSeries": vals[-8:],
                "rising3": rising,
                "falling3": falling,
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
    side: PaperSide = "long",
    mark_ltp: float | None = None,
    marks_by_strategy: dict[str, float] | None = None,
) -> dict[str, Any]:
    side = _normalize_side(side)
    buckets = []
    for sid in STRATEGIES:
        trades = _trades_for_strategy(con, sid, limit_per, side=side)
        sid_mark = (marks_by_strategy or {}).get(sid, mark_ltp)
        buckets.append(
            {
                "strategyId": sid,
                "summary": _summary_for(sid, trades, side=side, mark_ltp=sid_mark),
                "trades": trades,
            }
        )
    first = buckets[0] if buckets else None
    return {
        "positionSide": side,
        "optionLeg": _option_leg_for_side(side),
        "strategies": list_strategies(),
        "buckets": buckets,
        "wallet": _portfolio_wallet(buckets),
        "summary": first["summary"] if first else None,
        "trades": first["trades"] if first else [],
        "storage": _storage_label(),
        "mongoReady": _mongo_ready_flag(),
    }


def import_historical_trades(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Import verified historical paper trades (exact entry/exit from user records).

    This is intended for one-time repair/backfill when runtime signal history is
    unavailable or incomplete.
    """
    inserted = 0
    skipped = 0
    touched: list[int] = []
    with _lock:
        con = _conn()
        try:
            for row in rows or []:
                sid = str(row.get("strategyId") or "").strip().lower()
                if sid not in STRATEGIES:
                    skipped += 1
                    continue
                status = str(row.get("status") or "").strip().lower()
                if status not in {"open", "closed"}:
                    skipped += 1
                    continue
                entry_ts = str(row.get("entryTs") or "").strip()
                entry_px = row.get("entryPx")
                symbol = str(row.get("symbol") or "ATM-CE").strip() or "ATM-CE"
                if not entry_ts or entry_px is None:
                    skipped += 1
                    continue

                # Avoid duplicates on (strategy, symbol, entry_ts).
                exists = con.execute(
                    """
                    SELECT id FROM paper_trades
                    WHERE strategy_id=? AND symbol=? AND entry_ts=?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (sid, symbol, entry_ts),
                ).fetchone()
                if exists:
                    skipped += 1
                    continue

                exit_ts = row.get("exitTs")
                exit_px = row.get("exitPx")
                lot = int(row.get("lot") or LOT_SIZE)
                peak = row.get("peakPx")
                entry_px_f = float(entry_px)
                peak_f = float(peak) if peak is not None else entry_px_f

                pnl_rs = row.get("pnlRs")
                pnl_pct = row.get("pnlPct")
                if status == "closed" and exit_px is not None:
                    exit_px_f = float(exit_px)
                    if pnl_rs is None:
                        pnl_rs = round((exit_px_f - entry_px_f) * lot, 2)
                    if pnl_pct is None and entry_px_f:
                        pnl_pct = round((exit_px_f / entry_px_f - 1.0) * 100.0, 2)

                cur = con.execute(
                    """
                    INSERT INTO paper_trades (
                        strategy_id, status, side, symbol, strike, expiry, lot,
                        entry_ts, entry_px, entry_spot, entry_weight_up, entry_reason,
                        exit_ts, exit_px, exit_spot, exit_weight_up, exit_reason,
                        peak_px, pnl_rs, pnl_pct, meta
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        sid,
                        status,
                        str(row.get("side") or "CE"),
                        symbol,
                        row.get("strike"),
                        row.get("expiry"),
                        lot,
                        entry_ts,
                        entry_px_f,
                        row.get("entrySpot"),
                        row.get("entryWeightUp"),
                        row.get("entryReason") or "historical import",
                        exit_ts,
                        float(exit_px) if exit_px is not None else None,
                        row.get("exitSpot"),
                        row.get("exitWeightUp"),
                        row.get("exitReason"),
                        peak_f,
                        pnl_rs,
                        pnl_pct,
                        json.dumps({"imported": True, "source": "manual_backfill"}),
                    ),
                )
                tid = int(cur.lastrowid)
                touched.append(tid)
                inserted += 1

            con.commit()
            for tid in touched:
                _sync_trade_to_mongo(con, tid)

            board = _board_unlocked(con, limit_per=200)
            return {
                "ok": True,
                "inserted": inserted,
                "skipped": skipped,
                "rows": len(rows or []),
                **board,
            }
        finally:
            con.close()
