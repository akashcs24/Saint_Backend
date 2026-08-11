from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .config import settings
from .fyers_quotes import to_fyers_eq
from .nifty_weights import get_nifty_weights
from .sensex_weights import get_sensex_weights
from .session import SESSION_CLOSE, SESSION_OPEN, is_trading_day, now_ist, prev_trading_day

IST = ZoneInfo("Asia/Kolkata")
_FLAT_PCT = 0.05


def _safe_symbol(symbol: str) -> str:
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


def _target_session_date() -> date:
    now = now_ist()
    if is_trading_day(now.date()) and now.time() >= SESSION_OPEN:
        return now.date()
    return prev_trading_day(now.date())


def _latest_parquet(symbol: str, target_dir: Path, resolution: str) -> Path | None:
    safe = _safe_symbol(symbol)
    matches = sorted(target_dir.rglob(f"{safe}_{resolution}_*.parquet"))
    return matches[-1] if matches else None


def _session_minutes(session_day: date) -> pd.DatetimeIndex:
    start = datetime.combine(session_day, SESSION_OPEN)
    # NSE minute bars typically end at 15:29 for the 15:30 close.
    end = datetime.combine(session_day, (datetime.combine(session_day, SESSION_CLOSE) - timedelta(minutes=1)).time())
    return pd.date_range(start=start, end=end, freq="1min")


def _status(
    *,
    weight_up: float,
    weight_down: float,
    contrib: float,
    prev_weight_up: float | None,
    prev_contrib: float | None,
    prev_status: str | None,
) -> tuple[str, float | None, float | None]:
    if prev_weight_up is None or prev_contrib is None:
        return "Normal", None, None
    dw_up = weight_up - prev_weight_up
    d_contrib = contrib - prev_contrib
    bull_dominant = weight_up >= (weight_down + 5.0)
    bear_dominant = weight_down >= (weight_up + 5.0)

    if bear_dominant and (d_contrib <= -0.07 or (dw_up <= -6.0 and d_contrib <= -0.04)):
        return "Bearish shock", dw_up, d_contrib
    if bull_dominant and (d_contrib >= 0.07 or (dw_up >= 6.0 and d_contrib >= 0.04)):
        return "Bullish shock", dw_up, d_contrib

    if bull_dominant and (dw_up >= 2.0 and d_contrib >= 0.05):
        return "Bullish buildup", dw_up, d_contrib
    if bear_dominant and (dw_up <= -2.0 and d_contrib <= -0.05):
        return "Bearish buildup", dw_up, d_contrib
    if prev_status == "Bullish buildup" and (dw_up <= -1.0 or d_contrib <= -0.05):
        return "Post bullish", dw_up, d_contrib
    if prev_status == "Bearish buildup" and (dw_up >= 1.0 or d_contrib >= 0.05):
        return "Post bearish", dw_up, d_contrib
    return "Normal", dw_up, d_contrib


def rebuild_breadth_1m_from_cached_data(*, index_key: str, session_day: date | None = None) -> dict[str, Any]:
    from .nifty_breadth_history import replace_session_history

    k = (index_key or "nifty").lower()
    if k not in {"nifty", "sensex"}:
        raise ValueError(f"Unsupported index_key: {index_key}")

    if session_day is None:
        session_day = _target_session_date()

    weights = (get_nifty_weights() if k == "nifty" else get_sensex_weights()).get("weights") or {}
    if not weights:
        return {"ok": False, "indexKey": k, "sessionDay": session_day.isoformat(), "rows": 0, "reason": "no_weights"}

    minute_index = _session_minutes(session_day)
    symbol_series: dict[str, pd.Series] = {}
    symbol_weights: dict[str, float] = {}

    for saint_symbol, wt in weights.items():
        fy_sym = to_fyers_eq(saint_symbol)
        path = _latest_parquet(fy_sym, settings.market_stocks_1m_dir.resolve(), "1")
        if path is None:
            continue
        try:
            df = _normalize_ohlcv(pd.read_parquet(path))
        except Exception:  # noqa: BLE001
            continue
        if df.empty or "Close" not in df.columns:
            continue

        prev = df[df.index.date < session_day]
        if prev.empty:
            continue
        prev_close = float(prev["Close"].dropna().iloc[-1])
        if prev_close <= 0:
            continue

        day_df = df[df.index.date == session_day]
        if day_df.empty:
            continue

        close_series = day_df["Close"].astype(float).reindex(minute_index).ffill()
        chg_series = (close_series / prev_close - 1.0) * 100.0
        symbol_series[fy_sym] = chg_series
        symbol_weights[fy_sym] = float(wt)

    if not symbol_series:
        return {
            "ok": False,
            "indexKey": k,
            "sessionDay": session_day.isoformat(),
            "rows": 0,
            "reason": "no_symbol_series",
        }

    rows: list[dict[str, Any]] = []
    prev_weight_up: float | None = None
    prev_contrib: float | None = None
    prev_status: str | None = None

    for ts in minute_index:
        live: list[tuple[str, float, float]] = []
        for sym, series in symbol_series.items():
            val = series.get(ts)
            if val is None or pd.isna(val):
                continue
            live.append((sym, float(symbol_weights.get(sym) or 0.0), float(val)))
        if not live:
            continue

        total_w = sum(w for _sym, w, _chg in live)
        if total_w <= 0:
            continue

        advances = declines = unchanged = 0
        weight_up = weight_down = weight_flat = 0.0
        contrib = 0.0

        for _sym, raw_w, chg in live:
            w = (raw_w / total_w) * 100.0
            contrib += (w / 100.0) * chg
            if chg > _FLAT_PCT:
                advances += 1
                weight_up += w
            elif chg < -_FLAT_PCT:
                declines += 1
                weight_down += w
            else:
                unchanged += 1
                weight_flat += w

        weight_up_r = round(weight_up, 1)
        weight_down_r = round(weight_down, 1)
        contrib_r = round(contrib, 3)
        status, dw_up_1, d_contrib_1 = _status(
            weight_up=weight_up_r,
            weight_down=weight_down_r,
            contrib=contrib_r,
            prev_weight_up=prev_weight_up,
            prev_contrib=prev_contrib,
            prev_status=prev_status,
        )

        ts_aware = ts.replace(tzinfo=IST)
        bucket_ts = int(ts_aware.astimezone(ZoneInfo("UTC")).timestamp())
        row = {
            "bucketTs": bucket_ts,
            "t": ts.strftime("%H:%M"),
            "asOf": ts_aware.isoformat(),
            "weightUp": weight_up_r,
            "weightDown": weight_down_r,
            "weightFlat": round(weight_flat, 1),
            "contributionPct": contrib_r,
            "advances": advances,
            "declines": declines,
            "lean": "bullish" if contrib_r > 0 else "bearish" if contrib_r < 0 else "mixed",
            "dwUp1": dw_up_1,
            "dContrib1": d_contrib_1,
            "breadthStatus": status,
        }
        rows.append(row)
        prev_weight_up = weight_up_r
        prev_contrib = contrib_r
        prev_status = status

    replace_session_history(index_key=k, interval_minutes=1, session_day=session_day, rows=rows)
    return {
        "ok": True,
        "indexKey": k,
        "sessionDay": session_day.isoformat(),
        "rows": len(rows),
        "symbolsUsed": len(symbol_series),
    }
