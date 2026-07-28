"""Nifty Put-Call Ratio from nearest-expiry option chain.

Source: NiftyTrader option-chain JSON (NSE chain is often blocked from servers).
OI PCR = total Put OI / total Call OI.

Trading bands (writer-driven, India index options):
  < 0.8  → call selling (bearish lean)
  0.8–1.0 → decision zone
  > 1.0  → put selling (bullish lean)
  ≥ 1.2  → rare extreme / one-sided trend
"""

from __future__ import annotations

import time
from typing import Any

import requests

from .trend_history import record as record_trend
from .trend_history import trend_pair

_URL = "https://webapi.niftytrader.in/webapi/option/option-chain-data?symbol=nifty&expiryDate="
_TTL_S = 60
_cache: dict[str, Any] = {"ts": 0.0, "payload": None}

_CALL_SELL = 0.80  # strictly below → call writing
_PUT_SELL = 1.00  # strictly above → put writing
_EXTREME = 1.20  # rare one-sided
_PCR_TREND_EPS = 0.008


def _bias(oi_pcr: float) -> tuple[str, str]:
    if oi_pcr >= _EXTREME:
        return "bullish", "Extreme put writing — rare one-sided"
    if oi_pcr > _PUT_SELL:
        return "bullish", "Put selling — bullish lean"
    if oi_pcr < _CALL_SELL:
        return "bearish", "Call selling — bearish lean"
    return "neutral", "Decision zone (0.8–1.0)"


def _attach_trend(payload: dict) -> dict:
    oi = float(payload.get("oiPcr") or 0)
    record_trend("nifty_oi_pcr", oi)
    payload = dict(payload)
    payload["trend"] = trend_pair("nifty_oi_pcr", current=oi, eps=_PCR_TREND_EPS)
    return payload


def fetch_nifty_pcr(*, force: bool = False) -> dict | None:
    """Nearest-expiry Nifty OI + volume PCR, or None if unavailable."""
    now = time.time()
    if not force and _cache["payload"] is not None and now - float(_cache["ts"]) < _TTL_S:
        return _attach_trend(_cache["payload"])

    try:
        r = requests.get(
            _URL,
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Saint/1.0)",
                "Accept": "application/json",
            },
        )
        r.raise_for_status()
        body = r.json()
        tot = ((body.get("resultData") or {}).get("opTotals") or {}).get("total_calls_puts") or {}
        rows = (body.get("resultData") or {}).get("opDatas") or []
        ce_oi = float(tot.get("total_calls_oi") or 0)
        pe_oi = float(tot.get("total_puts_oi") or 0)
        ce_vol = float(tot.get("total_calls_volume") or 0)
        pe_vol = float(tot.get("total_puts_volume") or 0)
        if ce_oi <= 0 and rows:
            ce_oi = sum(float(x.get("calls_oi") or 0) for x in rows)
            pe_oi = sum(float(x.get("puts_oi") or 0) for x in rows)
            ce_vol = sum(float(x.get("calls_volume") or 0) for x in rows)
            pe_vol = sum(float(x.get("puts_volume") or 0) for x in rows)
        if ce_oi <= 0:
            return None
        oi_pcr = pe_oi / ce_oi
        vol_pcr = (pe_vol / ce_vol) if ce_vol > 0 else None
        expiry = None
        as_of = None
        if rows:
            expiry = (rows[0].get("expiry_date") or "")[:10] or None
            as_of = rows[0].get("time") or None
        lean, label = _bias(oi_pcr)
        payload = {
            "ready": True,
            "oiPcr": round(oi_pcr, 3),
            "volumePcr": round(vol_pcr, 3) if vol_pcr is not None else None,
            "putOi": int(pe_oi),
            "callOi": int(ce_oi),
            "lean": lean,
            "label": label,
            "expiry": expiry,
            "asOf": as_of,
            "source": "niftytrader",
        }
        _cache["ts"] = now
        _cache["payload"] = payload
        return _attach_trend(payload)
    except Exception:  # noqa: BLE001
        if _cache["payload"] is not None:
            return _attach_trend(_cache["payload"])
        return None
