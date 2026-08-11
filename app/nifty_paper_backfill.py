from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import settings
from .nifty_paper_trades import LOT_SIZE, import_historical_trades
from .prices import fetch_intraday_15m


def _research_root() -> Path:
    return settings.price_cache.parent.parent


def _output_dir() -> Path:
    return _research_root() / "backtest" / "output"


def _data_dir() -> Path:
    return settings.price_cache.parent


def _project_option_roots() -> list[Path]:
    return [
        settings.nifty_option_15m_dir.resolve(),
        settings.nifty_option_1m_dir.resolve(),
        settings.nifty_option_5m_dir.resolve(),
    ]


def _csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _hist_symbol(atm: Any, leg: str) -> str:
    try:
        strike = int(round(float(atm)))
    except Exception:  # noqa: BLE001
        strike = 0
    return f"NSE:NIFTY-ATM-{strike}{leg}"


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
    return _wma(2.0 * _wma(series, half) - _wma(series, n), root)


def _load_option_series(atm: int, leg: str) -> pd.Series:
    roots = _project_option_roots() + [_data_dir() / "fyers_fo", _data_dir() / "fyers_1m"]
    patterns = [
        f"NSE_NIFTY*{atm}{leg}_15_*.parquet",
        f"NSE_NIFTY*{atm}{leg}_1_*.parquet",
        f"NSE_NIFTY*{atm}{leg}_5_*.parquet",
    ]
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(sorted(root.rglob(pattern)))
    if not matches:
        return pd.Series(dtype=float)
    for path in sorted(matches, reverse=True):
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Datetime" in df.columns:
                df = df.set_index("Datetime")
            df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        if "Close" not in df.columns:
            continue
        ser = df["Close"].sort_index()
        if not ser.index.is_unique:
            ser = ser[~ser.index.duplicated(keep="last")]
        return ser
    return pd.Series(dtype=float)


def _load_option_frame(atm: int, leg: str, *, day: datetime.date | None = None) -> pd.DataFrame:
    roots = _project_option_roots() + [_data_dir() / "fyers_fo", _data_dir() / "fyers_1m"]
    patterns = [
        f"NSE_NIFTY*{atm}{leg}_15_*.parquet",
        f"NSE_NIFTY*{atm}{leg}_5_*.parquet",
        f"NSE_NIFTY*{atm}{leg}_1_*.parquet",
    ]
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(sorted(root.rglob(pattern)))
    fallback = pd.DataFrame()
    for path in sorted(matches, reverse=True):
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Datetime" in df.columns:
                df = df.set_index("Datetime")
            df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        if "Close" not in keep:
            continue
        out = df[keep].sort_index()
        out = out[~out.index.duplicated(keep="last")]
        if day is not None and any(pd.Timestamp(ts).date() == day for ts in out.index):
            return out
        if fallback.empty:
            fallback = out
    return fallback


def _rows_from_weightup_long() -> list[dict[str, Any]]:
    path = _output_dir() / "nifty_weightUp_entry3_exit_variants_trades.csv"
    df = _csv(path)
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    for strategy_id, exit_mode, exit_label in (
        ("decline", "decline×4", "weightUp falling ×4 (backtest)"),
        ("tsl", "decline×3", "weightUp falling ×3 (backtest)"),
    ):
        sub = df[(df["tf_min"] == 5) & (df["exit_mode"] == exit_mode)].copy()
        for _, row in sub.iterrows():
            out.append(
                {
                    "strategyId": strategy_id,
                    "status": "closed",
                    "side": "CE",
                    "symbol": _hist_symbol(row.get("atm"), "CE"),
                    "strike": row.get("atm"),
                    "lot": LOT_SIZE,
                    "entryTs": str(row.get("entry_ts")),
                    "entryPx": float(row.get("entry_px")),
                    "entrySpot": row.get("spot"),
                    "entryWeightUp": row.get("wUp_entry"),
                    "entryReason": "weightUp rising ×3 (backtest)",
                    "exitTs": str(row.get("exit_ts")),
                    "exitPx": float(row.get("exit_px")),
                    "exitWeightUp": row.get("wUp_exit"),
                    "exitReason": exit_label,
                    "peakPx": max(float(row.get("peak_px") or row.get("entry_px")), float(row.get("exit_px"))),
                    "pnlRs": float(row.get("pnl_rs")),
                    "pnlPct": float(row.get("retPct")),
                }
            )
    return out


def _rows_from_weightup_short() -> list[dict[str, Any]]:
    rise4_path = _output_dir() / "nifty_pe_reverse_trades_latest.csv"
    rise3_path = _output_dir() / "nifty_pe_fall4_rise3_tf_trades_latest.csv"
    out: list[dict[str, Any]] = []

    df = _csv(rise4_path)
    if not df.empty:
        sub = df[(df["family"] == "weightup_pe") & (df["exitRule"] == "rise_4")].copy()
        for _, row in sub.iterrows():
            out.append(
                {
                    "strategyId": "decline",
                    "status": "closed",
                    "side": "PE",
                    "symbol": _hist_symbol(row.get("atm"), "PE"),
                    "strike": row.get("atm"),
                    "lot": LOT_SIZE,
                    "entryTs": str(row.get("entry")),
                    "entryPx": float(row.get("entryPx")),
                    "entryWeightUp": row.get("weightUp"),
                    "entryReason": "weightUp falling ×3 (backtest)",
                    "exitTs": str(row.get("exit")),
                    "exitPx": float(row.get("exitPx")),
                    "exitReason": "weightUp rising ×4 (backtest)",
                    "peakPx": max(float(row.get("entryPx")), float(row.get("exitPx"))),
                    "pnlRs": float(row.get("pnlRs")),
                    "pnlPct": float(row.get("retPct")),
                }
            )

    df = _csv(rise3_path)
    if not df.empty:
        sub = df[df["tfMin"] == 5].copy()
        for _, row in sub.iterrows():
            out.append(
                {
                    "strategyId": "tsl",
                    "status": "closed",
                    "side": "PE",
                    "symbol": _hist_symbol(row.get("atm"), "PE"),
                    "strike": row.get("atm"),
                    "lot": LOT_SIZE,
                    "entryTs": str(row.get("entry")),
                    "entryPx": float(row.get("entryPx")),
                    "entryReason": "weightUp falling ×3 (backtest)",
                    "exitTs": str(row.get("exit")),
                    "exitPx": float(row.get("exitPx")),
                    "exitReason": "weightUp rising ×3 (backtest)",
                    "peakPx": max(float(row.get("entryPx")), float(row.get("exitPx"))),
                    "pnlRs": float(row.get("pnlRs")),
                    "pnlPct": float(row.get("retPct")),
                }
            )
    return out


def _rows_from_cross() -> list[dict[str, Any]]:
    long_path = _output_dir() / "nifty_sync_cross_trades_latest.csv"
    short_path = _output_dir() / "nifty_pe_reverse_trades_latest.csv"
    out: list[dict[str, Any]] = []

    df = _csv(long_path)
    if not df.empty:
        sub = df[df["exitRule"] == "flip_le_0"].copy()
        for _, row in sub.iterrows():
            out.append(
                {
                    "strategyId": "cross",
                    "status": "closed",
                    "side": "CE",
                    "symbol": _hist_symbol(row.get("atm"), "CE"),
                    "strike": row.get("atm"),
                    "lot": LOT_SIZE,
                    "entryTs": str(row.get("entry")),
                    "entryPx": float(row.get("entryPx")),
                    "entryWeightUp": row.get("crossDiffPp"),
                    "entryReason": f"cross.diffPp > 0 ({float(row.get('crossDiffPp')):.3f}) (backtest)",
                    "exitTs": str(row.get("exit")),
                    "exitPx": float(row.get("exitPx")),
                    "exitWeightUp": row.get("crossDiffPp"),
                    "exitReason": str(row.get("exitReason") or "cross<=0") + " (backtest)",
                    "peakPx": max(float(row.get("entryPx")), float(row.get("exitPx"))),
                    "pnlRs": float(row.get("pnlRs")),
                    "pnlPct": float(row.get("retPct")),
                }
            )

    df = _csv(short_path)
    if not df.empty:
        sub = df[(df["family"] == "sync_cross_pe") & (df["exitRule"] == "flip_ge_0")].copy()
        for _, row in sub.iterrows():
            out.append(
                {
                    "strategyId": "cross",
                    "status": "closed",
                    "side": "PE",
                    "symbol": _hist_symbol(row.get("atm"), "PE"),
                    "strike": row.get("atm"),
                    "lot": LOT_SIZE,
                    "entryTs": str(row.get("entry")),
                    "entryPx": float(row.get("entryPx")),
                    "entryWeightUp": row.get("crossDiffPp"),
                    "entryReason": f"cross.diffPp < 0 ({float(row.get('crossDiffPp')):.3f}) (backtest)",
                    "exitTs": str(row.get("exit")),
                    "exitPx": float(row.get("exitPx")),
                    "exitWeightUp": row.get("crossDiffPp"),
                    "exitReason": str(row.get("exitReason") or "cross>=0") + " (backtest)",
                    "peakPx": max(float(row.get("entryPx")), float(row.get("exitPx"))),
                    "pnlRs": float(row.get("pnlRs")),
                    "pnlPct": float(row.get("retPct")),
                }
            )
    return out


def _rows_from_vwap_hma() -> list[dict[str, Any]]:
    df = fetch_intraday_15m("NIFTY", force=False).copy()
    if df.empty:
        return []
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        return []
    df = df.loc[(df.index >= pd.Timestamp("2026-07-22 09:15:00")) & (df.index <= pd.Timestamp("2026-07-31 15:29:00"))]
    rows: list[dict[str, Any]] = []
    for side, leg in (("long", "CE"), ("short", "PE")):
        for day in sorted({pd.Timestamp(ts).date() for ts in df.index}):
            day_nifty = df.loc[df.index.date == day]
            if day_nifty.empty:
                continue
            anchor_spot = float(day_nifty.iloc[0].get("Open") or day_nifty.iloc[0].get("Close") or 0)
            if anchor_spot <= 0:
                continue
            atm = int(round(anchor_spot / 50.0) * 50)
            opt = _load_option_frame(atm, leg, day=day)
            if opt.empty:
                continue
            opt = opt.copy().dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(opt) < 2:
                continue

            prior = opt.loc[opt.index.date < day]
            if len(prior) < 46:
                # Need at least 46 prior bars to warm up HMA46 for this day.
                continue

            src = (opt["High"] + opt["Low"] + opt["Close"]) / 3.0
            vol = opt["Volume"].astype(float)
            if float(vol.fillna(0.0).sum()) <= 0:
                vol = pd.Series(1.0, index=opt.index)
            day_key = pd.Series(opt.index.date, index=opt.index)
            vwap = (src * vol).groupby(day_key).cumsum() / vol.groupby(day_key).cumsum()

            # Warm up HMA46 on full history so early-day bars can have valid values.
            hma = _hma(opt["Close"].astype(float), 46)
            work_all = opt.assign(vwap=vwap, hma=hma).dropna(subset=["vwap", "hma", "Close"])
            if work_all.empty:
                continue
            work = work_all.loc[work_all.index.date == day]
            if len(work) < 2:
                continue

            open_trade: dict[str, Any] | None = None
            for i in range(1, len(work)):
                prev = work.iloc[i - 1]
                curr = work.iloc[i]
                ts = pd.Timestamp(work.index[i]).to_pydatetime()
                spot_now = float(day_nifty["Close"].asof(ts)) if not day_nifty.empty else anchor_spot
                px = float(curr["Close"])
                bullish = float(curr["vwap"]) > float(prev["vwap"]) and float(curr["Close"]) > float(curr["hma"])
                bearish = float(curr["vwap"]) < float(prev["vwap"]) and float(curr["Close"]) < float(curr["hma"])
                entry_ready = bullish
                exit_ready = bearish

                if open_trade is None:
                    if entry_ready and px > 0:
                        open_trade = {
                            "strategyId": "vwap_hma15",
                            "status": "open",
                            "side": leg,
                            "symbol": _hist_symbol(atm, leg),
                            "strike": atm,
                            "lot": LOT_SIZE,
                            "entryTs": ts.isoformat(sep=" "),
                            "entryPx": px,
                            "entrySpot": spot_now,
                            "entryWeightUp": float(curr["vwap"]),
                            "entryReason": "Option VWAP rising + close > HMA46 (backtest)",
                            "peakPx": float(curr.get("High") or px),
                        }
                    continue

                open_trade["peakPx"] = max(float(open_trade["peakPx"]), float(curr.get("High") or px))
                if exit_ready and px > 0:
                    entry_px = float(open_trade["entryPx"])
                    rows.append(
                        {
                            **open_trade,
                            "status": "closed",
                            "exitTs": ts.isoformat(sep=" "),
                            "exitPx": px,
                            "exitSpot": spot_now,
                            "exitWeightUp": float(curr["vwap"]),
                            "exitReason": "Option VWAP falling + close < HMA46 (backtest)",
                            "pnlRs": round((float(px) - entry_px) * LOT_SIZE, 2),
                            "pnlPct": round((float(px) / entry_px - 1.0) * 100.0, 2),
                        }
                    )
                    open_trade = None
            if open_trade is not None:
                last = work.iloc[-1]
                entry_px = float(open_trade["entryPx"])
                exit_px = float(last["Close"])
                ts = pd.Timestamp(work.index[-1]).to_pydatetime()
                spot_now = float(day_nifty["Close"].asof(ts)) if not day_nifty.empty else anchor_spot
                rows.append(
                    {
                        **open_trade,
                        "status": "closed",
                        "exitTs": ts.isoformat(sep=" "),
                        "exitPx": exit_px,
                        "exitSpot": spot_now,
                        "exitWeightUp": float(last["vwap"]),
                        "exitReason": "EOD (backtest)",
                        "pnlRs": round((exit_px - entry_px) * LOT_SIZE, 2),
                        "pnlPct": round((exit_px / entry_px - 1.0) * 100.0, 2),
                    }
                )
    return rows


def backfill_paper_trades() -> dict[str, Any]:
    rows = []
    rows.extend(_rows_from_weightup_long())
    rows.extend(_rows_from_weightup_short())
    rows.extend(_rows_from_cross())
    rows.extend(_rows_from_vwap_hma())
    result = import_historical_trades(rows)
    result["generatedRows"] = len(rows)
    result["sources"] = {
        "outputDir": str(_output_dir()),
        "dataDir": str(_data_dir()),
        "generatedAt": datetime.utcnow().isoformat() + "Z",
    }
    return result