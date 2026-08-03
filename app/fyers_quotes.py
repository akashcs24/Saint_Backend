"""Fyers batch quotes for Nifty breadth (and optional single-symbol use)."""

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
from .quotes import Quote

# Saint symbol → Fyers equity ticker
_FYERS_OVERRIDES: dict[str, str] = {
    "BAJAJ-AUTO": "NSE:BAJAJ-AUTO-EQ",
    "M&M": "NSE:M&M-EQ",
    "J&KBANK": "NSE:J&KBANK-EQ",
}


def to_fyers_eq(symbol: str) -> str:
    sym = symbol.strip().upper()
    if sym in _FYERS_OVERRIDES:
        return _FYERS_OVERRIDES[sym]
    if sym.startswith("NSE:"):
        return sym
    return f"NSE:{sym}-EQ"


def _parse_quote_row(row: dict[str, Any], saint_symbol: str) -> Quote | None:
    # Fyers v3: {"n": "NSE:SBIN-EQ", "v": {...}, "s": "ok"}
    v = row.get("v") if isinstance(row, dict) else None
    if not isinstance(v, dict):
        return None
    lp = v.get("lp")
    if lp is None:
        return None
    try:
        ltp = float(lp)
    except (TypeError, ValueError):
        return None
    prev = v.get("prev_close_price") or v.get("prev_close") or v.get("open_price")
    try:
        prev_f = float(prev) if prev is not None else ltp
    except (TypeError, ValueError):
        prev_f = ltp
    ch = v.get("ch")
    chp = v.get("chp")
    try:
        change = float(ch) if ch is not None else (ltp - prev_f)
    except (TypeError, ValueError):
        change = ltp - prev_f
    try:
        change_pct = float(chp) if chp is not None else ((change / prev_f * 100.0) if prev_f else 0.0)
    except (TypeError, ValueError):
        change_pct = (change / prev_f * 100.0) if prev_f else 0.0
    vol = v.get("volume") or v.get("ttv")
    try:
        volume = float(vol) if vol is not None else None
    except (TypeError, ValueError):
        volume = None
    return Quote(
        symbol=saint_symbol,
        ltp=round(ltp, 2),
        change=round(change, 2),
        change_pct=round(change_pct, 2),
        volume=volume,
        previous_close=round(prev_f, 2),
        source="fyers",
    )


def fetch_fyers_quotes(symbols: list[str]) -> dict[str, Quote]:
    """Batch-fetch LTP/change for Saint symbols. Empty dict if not connected / error."""
    from .session import is_live_data_window

    # Pause Fyers quote traffic outside 09:14–15:30 IST.
    if not is_live_data_window():
        return {}
    if not get_access_token():
        return {}
    if not symbols:
        return {}
    try:
        fyers = fyers_client()
    except Exception as exc:  # noqa: BLE001
        note_fyers_live_fail(f"{type(exc).__name__}: {exc}", auth=is_fyers_auth_failure(exc))
        return {}

    # map fyers name → saint symbol
    fyers_to_saint: dict[str, str] = {}
    fyers_syms: list[str] = []
    for s in symbols:
        f = to_fyers_eq(s)
        fyers_to_saint[f] = s
        fyers_syms.append(f)

    out: dict[str, Quote] = {}
    saw_auth_fail = False
    last_err: str | None = None
    for i in range(0, len(fyers_syms), 40):
        batch = fyers_syms[i : i + 40]
        try:
            resp = fyers.quotes(data={"symbols": ",".join(batch)})
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if is_fyers_auth_failure(exc):
                saw_auth_fail = True
                break
            continue
        if not isinstance(resp, dict):
            last_err = f"Unexpected quotes response: {resp!r}"
            continue
        if resp.get("s") != "ok" or is_fyers_auth_failure(resp):
            last_err = str(resp)
            if is_fyers_auth_failure(resp):
                saw_auth_fail = True
                break
            continue
        rows = resp.get("d") or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            n = str(row.get("n") or "")
            saint = fyers_to_saint.get(n)
            if not saint:
                # try strip match
                for fk, sk in fyers_to_saint.items():
                    if fk == n or n.endswith(fk.split(":")[-1]):
                        saint = sk
                        break
            if not saint:
                continue
            q = _parse_quote_row(row, saint)
            if q:
                out[saint] = q
        time.sleep(0.05)

    if out:
        note_fyers_live_ok()
    elif saw_auth_fail:
        note_fyers_live_fail(last_err or "Fyers token invalid", auth=True)
    elif get_access_token() and last_err:
        # Had a token but no quotes — grey until a probe succeeds again.
        note_fyers_live_fail(last_err, auth=False)
    return out
