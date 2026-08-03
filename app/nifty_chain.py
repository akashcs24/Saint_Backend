"""Nifty option chain helpers (Fyers when live; NiftyTrader fallback)."""

from __future__ import annotations

import time
from typing import Any

import requests

from .fyers_auth import get_access_token
from .fyers_quotes import fetch_fyers_quotes
from .quotes import get_quote

_NT_URL = "https://webapi.niftytrader.in/webapi/option/option-chain-data?symbol=nifty&expiryDate="
_chain_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
_CHAIN_TTL_S = 45.0


def _nifty_spot() -> tuple[float | None, str]:
    """Return (ltp, source). Prefer Fyers index quote, else Yahoo ^NSEI."""
    from .session import is_live_data_window

    if is_live_data_window() and get_access_token():
        # Try common Fyers index symbols via quotes batch
        from .fyers_auth import fyers_client

        try:
            f = fyers_client()
            for sym in ("NSE:NIFTY50-INDEX", "NSE:NIFTY-INDEX"):
                resp = f.quotes(data={"symbols": sym})
                if isinstance(resp, dict) and resp.get("s") == "ok":
                    rows = resp.get("d") or []
                    if rows and isinstance(rows[0], dict):
                        v = rows[0].get("v") or {}
                        lp = v.get("lp")
                        if lp is not None:
                            return float(lp), "fyers"
        except Exception:  # noqa: BLE001
            pass
    q = get_quote("NIFTY")
    if q:
        return float(q.ltp), q.source
    return None, "none"


def _fetch_niftytrader_chain() -> dict[str, Any] | None:
    try:
        r = requests.get(
            _NT_URL,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Saint/1.0)",
                "Accept": "application/json",
            },
        )
        r.raise_for_status()
        body = r.json()
        rd = body.get("resultData") or {}
        rows = rd.get("opDatas") or []
        if not rows:
            return None
        spot = None
        for row in rows:
            if row.get("index_close"):
                spot = float(row["index_close"])
                break
        strikes: list[dict[str, Any]] = []
        for row in rows:
            try:
                strike = float(row.get("strike_price") or 0)
            except (TypeError, ValueError):
                continue
            if strike <= 0:
                continue
            strikes.append(
                {
                    "strike": strike,
                    "ceOi": int(float(row.get("calls_oi") or 0)),
                    "peOi": int(float(row.get("puts_oi") or 0)),
                    "ceOiChg": int(float(row.get("calls_change_oi") or 0)),
                    "peOiChg": int(float(row.get("puts_change_oi") or 0)),
                    "ceVol": int(float(row.get("calls_volume") or 0)),
                    "peVol": int(float(row.get("puts_volume") or 0)),
                    "ceLtp": float(row["calls_ltp"]) if row.get("calls_ltp") is not None else None,
                    "peLtp": float(row["puts_ltp"]) if row.get("puts_ltp") is not None else None,
                    "ceIv": float(row["calls_iv"]) if row.get("calls_iv") is not None else None,
                    "peIv": float(row["puts_iv"]) if row.get("puts_iv") is not None else None,
                }
            )
        expiry = (rows[0].get("expiry_date") or "")[:10] or None
        as_of = rows[0].get("time") or rows[0].get("created_at")
        ce_oi = sum(s["ceOi"] for s in strikes)
        pe_oi = sum(s["peOi"] for s in strikes)
        return {
            "source": "niftytrader",
            "spot": spot,
            "expiry": expiry,
            "asOf": as_of,
            "strikes": strikes,
            "callOi": ce_oi,
            "putOi": pe_oi,
            "oiPcr": round(pe_oi / ce_oi, 3) if ce_oi > 0 else None,
        }
    except Exception:  # noqa: BLE001
        return None


def _fetch_fyers_chain(
    spot: float | None, *, allow_after_hours: bool = False
) -> dict[str, Any] | None:
    from .session import is_live_data_window

    # Board OI prefers live session; paper trades may still need last LTPs after hours.
    if not allow_after_hours and not is_live_data_window():
        return None
    if not get_access_token():
        return None
    try:
        from .fyers_auth import fyers_client

        f = fyers_client()
        resp = None
        for sym in ("NSE:NIFTY50-INDEX", "NSE:NIFTY-INDEX"):
            resp = f.optionchain(data={"symbol": sym, "strikecount": 20, "timestamp": ""})
            if isinstance(resp, dict) and resp.get("s") == "ok":
                break
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            return None
        data = resp.get("data") or {}
        rows = data.get("optionsChain") or data.get("options_chain") or []
        by_strike: dict[float, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                strike = float(row.get("strike_price") or row.get("strike") or 0)
            except (TypeError, ValueError):
                continue
            if strike <= 0:
                continue
            slot = by_strike.setdefault(
                strike,
                {
                    "strike": strike,
                    "ceOi": 0,
                    "peOi": 0,
                    "ceOiChg": 0,
                    "peOiChg": 0,
                    "ceVol": 0,
                    "peVol": 0,
                    "ceLtp": None,
                    "peLtp": None,
                    "ceIv": None,
                    "peIv": None,
                },
            )
            opt = str(row.get("option_type") or row.get("optionType") or "").upper()
            oi = int(float(row.get("oi") or row.get("OI") or 0))
            oich = int(float(row.get("oich") or row.get("oi_change") or row.get("previousOi") or 0))
            # previousOi is not change — prefer explicit change fields
            if row.get("oich") is None and row.get("oi_change") is None:
                oich = 0
            vol = int(float(row.get("volume") or row.get("v") or 0))
            ltp = row.get("ltp") or row.get("last_price")
            iv = row.get("iv")
            if opt in {"CE", "CALL"}:
                slot["ceOi"] = oi
                slot["ceOiChg"] = oich
                slot["ceVol"] = vol
                slot["ceLtp"] = float(ltp) if ltp is not None else None
                slot["ceIv"] = float(iv) if iv is not None else None
                slot["ceSymbol"] = row.get("symbol") or row.get("ex_symbol")
            elif opt in {"PE", "PUT"}:
                slot["peOi"] = oi
                slot["peOiChg"] = oich
                slot["peVol"] = vol
                slot["peLtp"] = float(ltp) if ltp is not None else None
                slot["peIv"] = float(iv) if iv is not None else None
                slot["peSymbol"] = row.get("symbol") or row.get("ex_symbol")
        strikes = sorted(by_strike.values(), key=lambda x: x["strike"])
        if not strikes:
            return None
        ce_oi = sum(s["ceOi"] for s in strikes)
        pe_oi = sum(s["peOi"] for s in strikes)
        expiry_dates = data.get("expiryDates") or data.get("expiry_dates") or []
        return {
            "source": "fyers",
            "spot": spot,
            "expiry": str(expiry_dates[0])[:10] if expiry_dates else None,
            "asOf": None,
            "strikes": strikes,
            "callOi": ce_oi,
            "putOi": pe_oi,
            "oiPcr": round(pe_oi / ce_oi, 3) if ce_oi > 0 else None,
        }
    except Exception:  # noqa: BLE001
        return None


def fetch_nifty_option_chain(
    *, force: bool = False, prefer_fyers_after_hours: bool = False
) -> dict[str, Any] | None:
    now = time.time()
    if (
        not force
        and _chain_cache["payload"] is not None
        and now - float(_chain_cache["ts"]) < _CHAIN_TTL_S
        # Don't reuse niftytrader cache when caller explicitly wants Fyers LTP.
        and (
            not prefer_fyers_after_hours
            or (_chain_cache["payload"] or {}).get("source") == "fyers"
        )
    ):
        return dict(_chain_cache["payload"])

    spot, spot_src = _nifty_spot()
    chain = (
        _fetch_fyers_chain(spot, allow_after_hours=prefer_fyers_after_hours)
        or _fetch_niftytrader_chain()
    )
    if not chain:
        return dict(_chain_cache["payload"]) if _chain_cache["payload"] else None

    if chain.get("spot") is None and spot is not None:
        chain["spot"] = spot
    chain["spotSource"] = spot_src
    _chain_cache["ts"] = now
    _chain_cache["payload"] = chain
    return dict(chain)


def atm_wing_board(chain: dict[str, Any], *, wing: int = 15) -> dict[str, Any]:
    """Slice ATM ± wing strikes; tag ITM/OTM/ATM for CE perspective on index."""
    strikes = list(chain.get("strikes") or [])
    spot = chain.get("spot")
    if not strikes or spot is None:
        return {
            "ready": False,
            "atmStrike": None,
            "rows": [],
            "ceOiWing": 0,
            "peOiWing": 0,
            "ceOiChgWing": 0,
            "peOiChgWing": 0,
        }
    atm = min(strikes, key=lambda s: abs(float(s["strike"]) - float(spot)))
    atm_k = float(atm["strike"])
    # Prefer strikes around ATM
    ordered = sorted(strikes, key=lambda s: abs(float(s["strike"]) - atm_k))
    picked = sorted(ordered[: wing * 2 + 1], key=lambda s: float(s["strike"]))
    rows = []
    for s in picked:
        k = float(s["strike"])
        if k == atm_k:
            moneyness = "ATM"
        elif k < spot:
            moneyness = "ITM"  # CE ITM / PE OTM for index
        else:
            moneyness = "OTM"
        rows.append(
            {
                **s,
                "moneyness": moneyness,
                "distPct": round((k - float(spot)) / float(spot) * 100.0, 3),
            }
        )
    return {
        "ready": True,
        "atmStrike": atm_k,
        "wing": wing,
        "rows": rows,
        "ceOiWing": sum(int(r["ceOi"]) for r in rows),
        "peOiWing": sum(int(r["peOi"]) for r in rows),
        "ceOiChgWing": sum(int(r["ceOiChg"]) for r in rows),
        "peOiChgWing": sum(int(r["peOiChg"]) for r in rows),
        "plot": [
            {
                "strike": r["strike"],
                "ceOi": r["ceOi"],
                "peOi": r["peOi"],
                "ceOiChg": r["ceOiChg"],
                "peOiChg": r["peOiChg"],
                "moneyness": r["moneyness"],
            }
            for r in rows
        ],
    }


def enrich_oi_plot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add stacked unwind/build fields for scratched peak visualization.

    Day change from chain: current OI = ceOi, change = ceOiChg.
    If change < 0, peak was higher by |change| → stack as ceUnwind (hatched).
    If change > 0, show ceBuild as the added portion above prior.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        ce = int(r.get("ceOi") or 0)
        pe = int(r.get("peOi") or 0)
        dce = int(r.get("ceOiChg") or 0)
        dpe = int(r.get("peOiChg") or 0)
        # Prior ≈ current - change
        ce_prior = max(0, ce - dce)
        pe_prior = max(0, pe - dpe)
        out.append(
            {
                **r,
                # Solid = current OI
                "ceOi": ce,
                "peOi": pe,
                # Unwound amount (peak above current) — show as hatched stack on top
                "ceUnwind": max(0, -dce),
                "peUnwind": max(0, -dpe),
                # Built today (for tooltip / optional second view)
                "ceBuild": max(0, dce),
                "peBuild": max(0, dpe),
                "cePrior": ce_prior,
                "pePrior": pe_prior,
                "ceOiChg": dce,
                "peOiChg": dpe,
            }
        )
    return out


def oi_insights(prev: dict[str, Any] | None, cur: dict[str, Any], spot: float | None) -> dict[str, Any]:
    """Rule-based OI read with a clear sentiment headline."""
    wing = cur.get("wingBoard") or {}
    if not wing.get("ready"):
        return {
            "headline": "Waiting for option chain…",
            "sentiment": "unclear",
            "bullets": [],
            "metrics": {},
        }

    ce_chg = int(wing.get("ceOiChgWing") or 0)
    pe_chg = int(wing.get("peOiChgWing") or 0)
    ce_oi = int(wing.get("ceOiWing") or 0)
    pe_oi = int(wing.get("peOiWing") or 0)
    atm = wing.get("atmStrike")
    net_put_bias = pe_chg - ce_chg
    bullets: list[str] = []
    sentiment = "neutral"

    # Session (day) change narrative
    mag = abs(ce_chg) + abs(pe_chg)
    if pe_chg > 0 and ce_chg <= 0 and pe_chg >= max(50_000, abs(ce_chg)):
        sentiment = "bullish"
        bullets.append(
            f"Put OI up {pe_chg:+,} while calls flat/down near ATM — writers building support (often bullish)."
        )
    elif ce_chg > 0 and pe_chg <= 0 and ce_chg >= max(50_000, abs(pe_chg)):
        sentiment = "bearish"
        bullets.append(
            f"Call OI up {ce_chg:+,} while puts flat/down near ATM — resistance / call writing (often bearish)."
        )
    elif ce_chg > 0 and pe_chg > 0:
        if pe_chg > ce_chg * 1.25:
            sentiment = "mild_bullish"
            bullets.append(
                f"Both sides adding OI, but puts lead (+{pe_chg:,} vs CE +{ce_chg:,}) — put-heavy range, mild support bias."
            )
        elif ce_chg > pe_chg * 1.25:
            sentiment = "mild_bearish"
            bullets.append(
                f"Both sides adding OI, but calls lead (+{ce_chg:,} vs PE +{pe_chg:,}) — call-heavy range, mild resistance bias."
            )
        else:
            sentiment = "neutral"
            bullets.append(
                f"Both CE & PE OI rising (CE {ce_chg:+,} · PE {pe_chg:+,}) — range / premium build-up."
            )
    elif ce_chg < 0 and pe_chg < 0:
        if abs(ce_chg) > abs(pe_chg) * 1.25:
            sentiment = "mild_bullish"
            bullets.append(
                f"OI unwinding both sides; calls shedding more ({ce_chg:+,}) — short-cover / resistance fade (mild bullish)."
            )
        elif abs(pe_chg) > abs(ce_chg) * 1.25:
            sentiment = "mild_bearish"
            bullets.append(
                f"OI unwinding both sides; puts shedding more ({pe_chg:+,}) — support fade (mild bearish)."
            )
        else:
            sentiment = "neutral"
            bullets.append(
                f"OI unwinding on both sides near ATM (CE {ce_chg:+,} · PE {pe_chg:+,}) — position exit / covering."
            )
    elif mag > 0:
        if net_put_bias > 0:
            sentiment = "mild_bullish"
            bullets.append(f"Net put-side OI bias today (PE−CE change {net_put_bias:+,}).")
        elif net_put_bias < 0:
            sentiment = "mild_bearish"
            bullets.append(f"Net call-side OI bias today (PE−CE change {net_put_bias:+,}).")

    # Snapshot-to-snapshot (approx 5m when board refreshes)
    d_ce = d_pe = 0
    spot_move = None
    if prev and prev.get("wingBoard"):
        pw = prev["wingBoard"]
        d_ce = int(wing.get("ceOiWing") or 0) - int(pw.get("ceOiWing") or 0)
        d_pe = int(wing.get("peOiWing") or 0) - int(pw.get("peOiWing") or 0)
        if abs(d_ce) + abs(d_pe) > 0:
            bullets.append(f"Since last snapshot · CE wing OI {d_ce:+,} · PE wing OI {d_pe:+,}.")
            # Strengthen sentiment from fresh flow if day signal was quiet
            if abs(d_pe - d_ce) >= 100_000:
                if d_pe > d_ce and sentiment in ("neutral", "unclear"):
                    sentiment = "mild_bullish"
                    bullets.append("Fresh put OI lead since last check — short-term supportive tone.")
                elif d_ce > d_pe and sentiment in ("neutral", "unclear"):
                    sentiment = "mild_bearish"
                    bullets.append("Fresh call OI lead since last check — short-term resistive tone.")
        psp = prev.get("spot")
        if spot is not None and psp is not None:
            spot_move = (float(spot) - float(psp)) / float(psp) * 100.0
            if abs(spot_move) >= 0.05:
                bullets.append(f"Spot moved {spot_move:+.2f}% over the last snapshot.")

    if atm is not None and spot is not None:
        bullets.append(f"ATM ≈ {atm:g} (spot {float(spot):,.1f}).")

    # Headline
    if sentiment == "bullish":
        headline = (
            f"OI lean bullish — puts added {pe_chg:+,} vs calls {ce_chg:+,} near ATM "
            f"(support / put-writing bias)."
        )
    elif sentiment == "bearish":
        headline = (
            f"OI lean bearish — calls added {ce_chg:+,} vs puts {pe_chg:+,} near ATM "
            f"(resistance / call-writing bias)."
        )
    elif sentiment == "mild_bullish":
        headline = f"Mild bullish OI tone — put-side flow leading (PE day Δ {pe_chg:+,} · CE {ce_chg:+,})."
    elif sentiment == "mild_bearish":
        headline = f"Mild bearish OI tone — call-side flow leading (CE day Δ {ce_chg:+,} · PE {pe_chg:+,})."
    elif not bullets:
        headline = "OI near ATM quiet — no strong writer signal yet."
    else:
        headline = f"OI mixed / range — CE day Δ {ce_chg:+,} · PE day Δ {pe_chg:+,} near ATM."

    return {
        "headline": headline,
        "sentiment": sentiment,
        "bullets": bullets[:5],
        "metrics": {
            "ceOiWing": ce_oi,
            "peOiWing": pe_oi,
            "ceOiChgWing": ce_chg,
            "peOiChgWing": pe_chg,
            "snapCeDelta": d_ce,
            "snapPeDelta": d_pe,
            "spotMovePct": round(spot_move, 3) if spot_move is not None else None,
            "atmStrike": atm,
        },
    }
