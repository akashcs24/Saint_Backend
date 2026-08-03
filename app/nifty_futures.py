"""Near-month Nifty futures quote + simple fair-value / basis."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .fyers_auth import fyers_client, get_access_token
from .session import now_ist

IST = ZoneInfo("Asia/Kolkata")
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

# Rough carry inputs — good enough for rich/cheap vs FV, not bank arb.
_RISK_FREE = 0.065
_DIV_YIELD = 0.012

_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
_TTL_S = 45.0


def near_month_fut_code(when: date | None = None) -> tuple[str, date]:
    """Return (YYMMM, expiry date) for the nearest monthly futures still trading."""
    today = when or now_ist().date()
    for code, month_start in _month_codes(today, 3):
        expiry = _last_thursday(month_start.year, month_start.month)
        if expiry >= today:
            return code, expiry
    # Fallback current month code
    code, month_start = _month_codes(today, 1)[0]
    return code, _last_thursday(month_start.year, month_start.month)


def _month_codes(from_d: date, count: int = 3) -> list[tuple[str, date]]:
    """Candidate (YYMMM, approx month-start date) for near futures."""
    out: list[tuple[str, date]] = []
    y, m = from_d.year, from_d.month
    for _ in range(count):
        code = f"{y % 100:02d}{_MONTHS[m - 1]}"
        out.append((code, date(y, m, 1)))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _last_thursday(year: int, month: int) -> date:
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != 3:  # Thu
        d -= timedelta(days=1)
    return d


def _parse_fyers_quote(resp: Any) -> dict[str, float | None] | None:
    if not isinstance(resp, dict) or resp.get("s") != "ok":
        return None
    rows = resp.get("d") or []
    if not rows or not isinstance(rows[0], dict):
        return None
    v = rows[0].get("v") or {}
    try:
        lp = float(v.get("lp"))
    except (TypeError, ValueError):
        return None
    prev = v.get("prev_close_price") or v.get("prev_close")
    try:
        prev_f = float(prev) if prev is not None else None
    except (TypeError, ValueError):
        prev_f = None
    chp = v.get("chp")
    try:
        chp_f = float(chp) if chp is not None else None
    except (TypeError, ValueError):
        chp_f = None
    if chp_f is None and prev_f and prev_f > 0:
        chp_f = (lp - prev_f) / prev_f * 100.0
    return {"ltp": lp, "previousClose": prev_f, "changePct": chp_f}


def _quote_fyers(symbol: str) -> dict[str, float | None] | None:
    try:
        f = fyers_client()
        return _parse_fyers_quote(f.quotes(data={"symbols": symbol}))
    except Exception:  # noqa: BLE001
        return None


def _quote_yahoo_fut(code: str) -> dict[str, float | None] | None:
    """Best-effort Yahoo: NIFTY26AUGFUT.NS style."""
    try:
        import yfinance as yf

        t = yf.Ticker(f"{code}.NS")
        fi = getattr(t, "fast_info", None)
        lp = None
        prev = None
        if fi is not None:
            lp = getattr(fi, "last_price", None) if not isinstance(fi, dict) else fi.get("last_price")
            prev = (
                getattr(fi, "previous_close", None)
                if not isinstance(fi, dict)
                else fi.get("previous_close")
            )
        if lp is None:
            hist = t.history(period="5d", auto_adjust=True)
            if hist is None or hist.empty:
                return None
            lp = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else lp
        lp_f = float(lp)
        prev_f = float(prev) if prev is not None else lp_f
        return {
            "ltp": lp_f,
            "previousClose": prev_f,
            "changePct": (lp_f - prev_f) / prev_f * 100.0 if prev_f else 0.0,
        }
    except Exception:  # noqa: BLE001
        return None


def _quote_nse_fut() -> dict[str, Any] | None:
    """Near Nifty index future from NSE public liveEquity-derivatives feed."""
    try:
        import requests

        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; Saint/1.0)",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/market-data/equity-derivatives-watch",
            }
        )
        s.get("https://www.nseindia.com", timeout=10)
        r = s.get(
            "https://www.nseindia.com/api/liveEquity-derivatives?index=nse50_fut",
            timeout=12,
        )
        r.raise_for_status()
        rows = (r.json() or {}).get("data") or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("underlying") or "").upper() != "NIFTY":
                continue
            if str(row.get("instrumentType") or "") != "FUTIDX":
                continue
            lp = row.get("lastPrice")
            if lp is None:
                continue
            exp_raw = str(row.get("expiryDate") or "")
            # 25-Aug-2026
            expiry = None
            try:
                expiry = datetime.strptime(exp_raw, "%d-%b-%Y").date().isoformat()
            except Exception:  # noqa: BLE001
                expiry = None
            chp = row.get("pChange")
            prev = None
            try:
                lp_f = float(lp)
                ch = float(row.get("change") or 0)
                prev = lp_f - ch
            except (TypeError, ValueError):
                lp_f = float(lp)
            return {
                "symbol": str(row.get("contract") or "NIFTY FUT"),
                "expiry": expiry,
                "expiryLabel": exp_raw.replace("-", " ") if exp_raw else None,
                "ltp": lp_f,
                "changePct": float(chp) if chp is not None else None,
                "previousClose": prev,
                "source": "nse",
            }
    except Exception:  # noqa: BLE001
        return None
    return None


def fetch_nifty_futures(*, spot: float | None = None, force: bool = False) -> dict[str, Any]:
    """Near-month Nifty futures + basis vs spot / rough fair value."""
    now = time.time()
    if (
        not force
        and _cache["payload"] is not None
        and now - float(_cache["ts"]) < _TTL_S
    ):
        out = dict(_cache["payload"])
        out["cached"] = True
        return out

    today = now_ist().date()
    candidates = _month_codes(today, 3)
    picked: dict[str, Any] | None = None

    use_fyers = bool(get_access_token())
    # Prefer Fyers (1 symbol), then NSE public feed, then Yahoo.
    for code, month_start in candidates:
        expiry = _last_thursday(month_start.year, month_start.month)
        if expiry < today:
            continue
        if not use_fyers:
            break
        fyers_sym = f"NSE:NIFTY{code}FUT"
        q = _quote_fyers(fyers_sym)
        if not q or q.get("ltp") is None:
            continue
        picked = {
            "symbol": fyers_sym,
            "expiry": expiry.isoformat(),
            "expiryLabel": expiry.strftime("%d %b"),
            "ltp": round(float(q["ltp"]), 2),
            "changePct": round(float(q["changePct"]), 3) if q.get("changePct") is not None else None,
            "previousClose": q.get("previousClose"),
            "source": "fyers",
        }
        break

    if not picked:
        nse = _quote_nse_fut()
        if nse and nse.get("ltp") is not None:
            exp = nse.get("expiry")
            picked = {
                "symbol": nse["symbol"],
                "expiry": exp,
                "expiryLabel": nse.get("expiryLabel")
                or (date.fromisoformat(exp).strftime("%d %b") if exp else None),
                "ltp": round(float(nse["ltp"]), 2),
                "changePct": (
                    round(float(nse["changePct"]), 3) if nse.get("changePct") is not None else None
                ),
                "previousClose": nse.get("previousClose"),
                "source": "nse",
            }

    if not picked:
        for code, month_start in candidates:
            expiry = _last_thursday(month_start.year, month_start.month)
            if expiry < today:
                continue
            q = _quote_yahoo_fut(f"NIFTY{code}FUT")
            if not q or q.get("ltp") is None:
                continue
            picked = {
                "symbol": f"NIFTY{code}FUT",
                "expiry": expiry.isoformat(),
                "expiryLabel": expiry.strftime("%d %b"),
                "ltp": round(float(q["ltp"]), 2),
                "changePct": round(float(q["changePct"]), 3) if q.get("changePct") is not None else None,
                "previousClose": q.get("previousClose"),
                "source": "yahoo",
            }
            break

    if not picked:
        payload = {
            "ready": False,
            "source": "none",
            "label": "Nifty futures quote unavailable",
            "cached": False,
        }
        _cache["ts"] = now
        _cache["payload"] = payload
        return payload

    fut = float(picked["ltp"])
    sp = float(spot) if spot is not None else None
    basis_pts = None
    basis_pct = None
    fair_value = None
    vs_fair_pts = None
    days_to_expiry = None
    if sp and sp > 0:
        basis_pts = round(fut - sp, 2)
        basis_pct = round((fut - sp) / sp * 100.0, 4)
        try:
            exp = date.fromisoformat(str(picked["expiry"]))
            days_to_expiry = max(0, (exp - today).days)
            t = days_to_expiry / 365.0
            fair_value = round(sp * (1.0 + (_RISK_FREE - _DIV_YIELD) * t), 2)
            vs_fair_pts = round(fut - fair_value, 2)
        except Exception:  # noqa: BLE001
            pass

    # Stance vs spot (basis) and vs fair value
    if basis_pct is None:
        basis_stance = "unclear"
        basis_label = "Need spot to compute futures basis"
    elif abs(basis_pct) < 0.02:
        basis_stance = "flat"
        basis_label = "Futures ≈ spot (tiny basis)"
    elif basis_pct > 0:
        basis_stance = "premium"
        basis_label = f"Futures at premium +{basis_pts:.0f} pts ({basis_pct:+.3f}%)"
    else:
        basis_stance = "discount"
        basis_label = f"Futures at discount {basis_pts:.0f} pts ({basis_pct:+.3f}%)"

    if vs_fair_pts is None:
        fv_stance = "unclear"
        fv_label = "Fair value not computed"
    elif abs(vs_fair_pts) < 8:
        fv_stance = "fair"
        fv_label = "Futures near rough fair value"
    elif vs_fair_pts > 0:
        fv_stance = "rich"
        fv_label = f"Futures rich vs FV by +{vs_fair_pts:.0f} pts"
    else:
        fv_stance = "cheap"
        fv_label = f"Futures cheap vs FV by {vs_fair_pts:.0f} pts"

    payload = {
        "ready": True,
        "cached": False,
        "symbol": picked["symbol"],
        "expiry": picked["expiry"],
        "expiryLabel": picked["expiryLabel"],
        "ltp": picked["ltp"],
        "changePct": picked["changePct"],
        "previousClose": picked.get("previousClose"),
        "source": picked["source"],
        "spot": sp,
        "basisPts": basis_pts,
        "basisPct": basis_pct,
        "basisStance": basis_stance,
        "basisLabel": basis_label,
        "fairValue": fair_value,
        "vsFairPts": vs_fair_pts,
        "fvStance": fv_stance,
        "fvLabel": fv_label,
        "daysToExpiry": days_to_expiry,
        "carryAssumptions": {"r": _RISK_FREE, "d": _DIV_YIELD},
        "label": f"{basis_label}. {fv_label}.",
    }
    _cache["ts"] = now
    _cache["payload"] = payload
    return payload


def combine_sync_insight(
    *,
    index_vs_basket_pp: float | None,
    cash_stance: str,
    futures: dict[str, Any] | None,
) -> str:
    """One short trader line from cash lead/lag + futures basis."""
    bits: list[str] = []
    if cash_stance == "nifty_ahead":
        bits.append(
            f"Printed Nifty is ahead of the cash basket by {abs(index_vs_basket_pp or 0):.3f}pp"
        )
    elif cash_stance == "nifty_lagging":
        bits.append(
            f"Printed Nifty is lagging the cash basket by {abs(index_vs_basket_pp or 0):.3f}pp"
        )
    else:
        bits.append("Printed Nifty and cash basket are in sync")

    fut = futures or {}
    if fut.get("ready"):
        if fut.get("basisStance") == "premium":
            bits.append("futures trading at a premium to spot")
        elif fut.get("basisStance") == "discount":
            bits.append("futures trading at a discount to spot")
        if fut.get("fvStance") == "rich":
            bits.append("and look rich vs carry fair value")
        elif fut.get("fvStance") == "cheap":
            bits.append("and look cheap vs carry fair value")

        # Cross-read
        if cash_stance == "nifty_lagging" and fut.get("basisStance") == "premium":
            bits.append("— index print soft vs stocks while futures stay bid (watch catch-up).")
        elif cash_stance == "nifty_ahead" and fut.get("basisStance") == "discount":
            bits.append("— index print firm vs stocks while futures soft (fade risk).")
        elif cash_stance == "nifty_ahead" and fut.get("basisStance") == "premium":
            bits.append("— both print and futures leading cash (momentum / FO lead).")
        elif cash_stance == "nifty_lagging" and fut.get("basisStance") == "discount":
            bits.append("— both print and futures soft vs cash (risk-off / lag).")

    return " ".join(bits)
