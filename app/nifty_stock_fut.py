"""Weighted Nifty stock-futures basket vs Nifty index futures."""

from __future__ import annotations

import time
from typing import Any

from .fyers_auth import (
    fyers_client,
    get_access_token,
    is_fyers_auth_failure,
    note_fyers_live_fail,
    note_fyers_live_ok,
)
from .nifty_futures import near_month_fut_code
from .universe import UNIVERSE

# Root used in Fyers stock-futures symbols (usually same as cash ticker).
_FUT_ROOT: dict[str, str] = {
    "BAJAJ-AUTO": "BAJAJ-AUTO",
    "M&M": "M&M",
    "J&KBANK": "J&KBANK",
}

_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
_TTL_S = 45.0
_SYNC_BAND = 0.08


def to_fyers_stock_fut(symbol: str, month_code: str) -> str:
    root = _FUT_ROOT.get(symbol.strip().upper(), symbol.strip().upper())
    return f"NSE:{root}{month_code}FUT"


def _batch_fyers_fut_quotes(
    saint_to_fyers: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Return saint → {ltp, changePct} for stock futures."""
    if not get_access_token() or not saint_to_fyers:
        return {}
    try:
        fyers = fyers_client()
    except Exception as exc:  # noqa: BLE001
        note_fyers_live_fail(f"{type(exc).__name__}: {exc}", auth=is_fyers_auth_failure(exc))
        return {}

    fyers_to_saint = {v: k for k, v in saint_to_fyers.items()}
    syms = list(saint_to_fyers.values())
    out: dict[str, dict[str, float]] = {}
    saw_auth = False
    last_err: str | None = None

    for i in range(0, len(syms), 40):
        batch = syms[i : i + 40]
        try:
            resp = fyers.quotes(data={"symbols": ",".join(batch)})
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if is_fyers_auth_failure(exc):
                saw_auth = True
                break
            continue
        if not isinstance(resp, dict):
            continue
        if resp.get("s") != "ok" or is_fyers_auth_failure(resp):
            last_err = str(resp)
            if is_fyers_auth_failure(resp):
                saw_auth = True
                break
            continue
        for row in resp.get("d") or []:
            if not isinstance(row, dict):
                continue
            n = str(row.get("n") or "")
            saint = fyers_to_saint.get(n)
            if not saint:
                continue
            v = row.get("v") or {}
            lp = v.get("lp")
            if lp is None:
                continue
            try:
                ltp = float(lp)
            except (TypeError, ValueError):
                continue
            prev = v.get("prev_close_price") or v.get("prev_close")
            chp = v.get("chp")
            try:
                prev_f = float(prev) if prev is not None else ltp
            except (TypeError, ValueError):
                prev_f = ltp
            try:
                chp_f = float(chp) if chp is not None else ((ltp - prev_f) / prev_f * 100.0 if prev_f else 0.0)
            except (TypeError, ValueError):
                chp_f = (ltp - prev_f) / prev_f * 100.0 if prev_f else 0.0
            out[saint] = {"ltp": ltp, "changePct": chp_f}
        time.sleep(0.05)

    if out:
        note_fyers_live_ok()
    elif saw_auth:
        note_fyers_live_fail(last_err or "Fyers token invalid", auth=True)
    return out


def build_stock_fut_basket(
    weights: dict[str, float],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Σ(weight × stock-futures day%) over Nifty names with a near-month future quote."""
    now = time.time()
    if (
        not force
        and _cache["payload"] is not None
        and now - float(_cache["ts"]) < _TTL_S
    ):
        out = dict(_cache["payload"])
        out["cached"] = True
        return out

    month_code, expiry = near_month_fut_code()
    symbols = [s for s in weights if s in UNIVERSE]
    mapping = {s: to_fyers_stock_fut(s, month_code) for s in symbols}

    if not get_access_token():
        payload = {
            "ready": False,
            "monthCode": month_code,
            "expiry": expiry.isoformat(),
            "quoted": 0,
            "universe": len(symbols),
            "label": "Connect Fyers to load stock-futures basket",
            "source": None,
            "cached": False,
        }
        _cache["ts"] = now
        _cache["payload"] = payload
        return payload

    quotes = _batch_fyers_fut_quotes(mapping)
    if not quotes:
        payload = {
            "ready": False,
            "monthCode": month_code,
            "expiry": expiry.isoformat(),
            "quoted": 0,
            "universe": len(symbols),
            "label": "Stock-futures quotes unavailable (check Fyers token / symbols)",
            "source": "fyers",
            "cached": False,
        }
        _cache["ts"] = now
        _cache["payload"] = payload
        return payload

    raw_w = {s: float(weights[s]) for s in quotes if s in weights}
    wsum = sum(raw_w.values()) or 1.0
    norm = {s: (w / wsum) * 100.0 for s, w in raw_w.items()}
    basket = sum(norm[s] / 100.0 * float(quotes[s]["changePct"]) for s in norm)

    payload = {
        "ready": True,
        "monthCode": month_code,
        "expiry": expiry.isoformat(),
        "expiryLabel": expiry.strftime("%d %b"),
        "basketMovePct": round(basket, 3),
        "quoted": len(quotes),
        "universe": len(symbols),
        "coveragePct": round(wsum, 1),  # raw weight coverage before renorm
        "source": "fyers",
        "label": f"Stock-futures basket · {len(quotes)}/{len(symbols)} names · {month_code}",
        "cached": False,
    }
    _cache["ts"] = now
    _cache["payload"] = payload
    return payload


def build_market_sync_card(
    *,
    cash_basket_pct: float | None,
    nifty_spot_pct: float | None,
    stock_fut_basket_pct: float | None,
    nifty_fut_pct: float | None,
    spot: float | None = None,
    nifty_fut_ltp: float | None = None,
    stock_fut_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Three-row sync: cash gap, FO gap, and (cash gap − FO gap)."""
    band = _SYNC_BAND

    def _gap(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return round(float(left) - float(right), 3)

    def _pts(pp: float | None, level: float | None) -> float | None:
        if pp is None or level is None or float(level) <= 0:
            return None
        return round(float(level) * float(pp) / 100.0, 1)

    def _stance(pp: float | None) -> str:
        if pp is None:
            return "unclear"
        if pp >= band:
            return "ahead"
        if pp <= -band:
            return "lagging"
        return "in_sync"

    # 1) Nifty spot vs stock cash basket (positive => Nifty ahead of stocks)
    cash_gap = _gap(nifty_spot_pct, cash_basket_pct)
    # 2) Nifty fut vs stock-fut basket
    fo_gap = _gap(nifty_fut_pct, stock_fut_basket_pct)
    # 3) Does FO lead/lag match cash? cash_gap − fo_gap
    cross = None
    if cash_gap is not None and fo_gap is not None:
        cross = round(cash_gap - fo_gap, 3)

    cash_stance = _stance(cash_gap)
    fo_stance = _stance(fo_gap)
    if cross is None:
        cross_stance = "unclear"
    elif abs(cross) < band:
        cross_stance = "aligned"
    elif cross > 0:
        # Cash says Nifty more ahead (or less lagging) than FO does
        cross_stance = "cash_more_nifty_led"
    else:
        cross_stance = "fo_more_nifty_led"

    insight_bits: list[str] = []
    if cash_stance == "ahead":
        insight_bits.append("Cash: Nifty print ahead of stock basket.")
    elif cash_stance == "lagging":
        insight_bits.append("Cash: Nifty print lagging stock basket.")
    elif cash_stance == "in_sync":
        insight_bits.append("Cash: Nifty print ≈ stock basket.")

    if fo_stance == "ahead":
        insight_bits.append("FO: Nifty futures ahead of stock-futures basket.")
    elif fo_stance == "lagging":
        insight_bits.append("FO: Nifty futures lagging stock-futures basket.")
    elif fo_stance == "in_sync":
        insight_bits.append("FO: Nifty futures ≈ stock-futures basket.")

    if cross_stance == "aligned":
        insight_bits.append("Cash and FO lead/lag agree.")
    elif cross_stance == "cash_more_nifty_led":
        insight_bits.append(
            "Cash shows more Nifty-led than FO — index print strong vs stocks relative to futures tape."
        )
    elif cross_stance == "fo_more_nifty_led":
        insight_bits.append(
            "FO shows more Nifty-led than cash — index futures leading stock futures harder than spot vs cash."
        )

    meta = stock_fut_meta or {}
    return {
        "ready": cash_gap is not None or fo_gap is not None,
        "syncBandPp": band,
        "cash": {
            "label": "1 · Stocks basket vs Nifty",
            "basketMovePct": cash_basket_pct,
            "niftyMovePct": nifty_spot_pct,
            "niftyVsBasketPp": cash_gap,
            "niftyVsBasketPts": _pts(cash_gap, spot),
            "stance": cash_stance,
            "verdict": (
                "Nifty ahead"
                if cash_stance == "ahead"
                else "Nifty lagging"
                if cash_stance == "lagging"
                else "In sync"
                if cash_stance == "in_sync"
                else "—"
            ),
        },
        "fo": {
            "label": "2 · Stock futures vs Nifty futures",
            "basketMovePct": stock_fut_basket_pct,
            "niftyMovePct": nifty_fut_pct,
            "niftyVsBasketPp": fo_gap,
            "niftyVsBasketPts": _pts(fo_gap, nifty_fut_ltp),
            "stance": fo_stance,
            "verdict": (
                "Nifty fut ahead"
                if fo_stance == "ahead"
                else "Nifty fut lagging"
                if fo_stance == "lagging"
                else "In sync"
                if fo_stance == "in_sync"
                else "—"
            ),
            "quoted": meta.get("quoted"),
            "universe": meta.get("universe"),
            "monthCode": meta.get("monthCode"),
            "ready": bool(meta.get("ready")) and fo_gap is not None,
            "note": meta.get("label"),
        },
        "cross": {
            "label": "1 vs 2 · Cash gap minus FO gap",
            "diffPp": cross,
            "stance": cross_stance,
            "verdict": (
                "Cash & FO aligned"
                if cross_stance == "aligned"
                else "Cash more Nifty-led"
                if cross_stance == "cash_more_nifty_led"
                else "FO more Nifty-led"
                if cross_stance == "fo_more_nifty_led"
                else "—"
            ),
        },
        "insight": " ".join(insight_bits) if insight_bits else "Building sync read…",
        "howToRead": (
            "Row 1: spot stocks (weight×%) vs printed Nifty. "
            "Row 2: same weights on near-month stock futures vs Nifty futures. "
            "Row 3: do cash and FO tell the same lead/lag story? "
            f"Band ±{band:.2f}pp. Exploratory — not a trade signal by itself."
        ),
    }
