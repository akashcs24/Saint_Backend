"""Fyers batch quotes for Nifty breadth (and optional single-symbol use)."""

from __future__ import annotations

import threading
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

FYERS_BATCH_SIZE = 40
FYERS_BATCH_INTERVAL_S = 1.0

_live_cache: dict[str, tuple[float, Quote]] = {}
_live_lock = threading.Lock()
_poller: dict[str, Any] = {
    "thread": None,
    "running": False,
    "symbols": [],
    "batch_idx": 0,
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


def _store_live_quotes(quotes: dict[str, Quote]) -> None:
    if not quotes:
        return
    now = time.time()
    with _live_lock:
        for sym, q in quotes.items():
            _live_cache[sym] = (now, q)


def _fetch_fyers_batch(fyers_syms: list[str], fyers_to_saint: dict[str, str]) -> dict[str, Quote]:
    """One Fyers API batch (≤40 symbols)."""
    if not fyers_syms:
        return {}
    try:
        fyers = fyers_client()
    except Exception as exc:  # noqa: BLE001
        note_fyers_live_fail(f"{type(exc).__name__}: {exc}", auth=is_fyers_auth_failure(exc))
        return {}

    out: dict[str, Quote] = {}
    saw_auth_fail = False
    last_err: str | None = None
    try:
        resp = fyers.quotes(data={"symbols": ",".join(fyers_syms)})
    except Exception as exc:  # noqa: BLE001
        last_err = f"{type(exc).__name__}: {exc}"
        if is_fyers_auth_failure(exc):
            saw_auth_fail = True
        else:
            return {}
    else:
        if not isinstance(resp, dict):
            last_err = f"Unexpected quotes response: {resp!r}"
        elif resp.get("s") != "ok" or is_fyers_auth_failure(resp):
            last_err = str(resp)
            if is_fyers_auth_failure(resp):
                saw_auth_fail = True
        else:
            rows = resp.get("d") or []
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    n = str(row.get("n") or "")
                    saint = fyers_to_saint.get(n)
                    if not saint:
                        for fk, sk in fyers_to_saint.items():
                            if fk == n or n.endswith(fk.split(":")[-1]):
                                saint = sk
                                break
                    if not saint:
                        continue
                    q = _parse_quote_row(row, saint)
                    if q:
                        out[saint] = q

    if out:
        note_fyers_live_ok()
    elif saw_auth_fail:
        note_fyers_live_fail(last_err or "Fyers token invalid", auth=True)
    elif get_access_token() and last_err:
        note_fyers_live_fail(last_err, auth=False)
    return out


def _poll_loop() -> None:
    """Rotate Fyers batches — one batch per second during market hours."""
    from .session import is_live_data_window

    while _poller.get("running"):
        if not is_live_data_window() or not get_access_token():
            time.sleep(5.0)
            continue
        symbols = list(_poller.get("symbols") or [])
        if not symbols:
            time.sleep(2.0)
            continue

        batch_idx = int(_poller.get("batch_idx") or 0)
        batch_syms = symbols[batch_idx : batch_idx + FYERS_BATCH_SIZE]
        if not batch_syms:
            _poller["batch_idx"] = 0
            continue
        _poller["batch_idx"] = batch_idx + FYERS_BATCH_SIZE

        fyers_to_saint: dict[str, str] = {}
        fyers_syms: list[str] = []
        for s in batch_syms:
            f = to_fyers_eq(s)
            fyers_to_saint[f] = s
            fyers_syms.append(f)

        quotes = _fetch_fyers_batch(fyers_syms, fyers_to_saint)
        _store_live_quotes(quotes)
        time.sleep(FYERS_BATCH_INTERVAL_S)


def ensure_fyers_poller(symbols: list[str]) -> None:
    """Start background batch polling for Nifty constituents (idempotent)."""
    uniq = list(dict.fromkeys(s for s in symbols if s))
    if not uniq:
        return
    with _live_lock:
        _poller["symbols"] = uniq
        if _poller.get("thread") and _poller["thread"].is_alive():
            return
        _poller["running"] = True
        t = threading.Thread(target=_poll_loop, daemon=True, name="fyers-quote-poller")
        t.start()
        _poller["thread"] = t


def stop_fyers_poller() -> None:
    _poller["running"] = False


def live_fyers_quotes(symbols: list[str], *, max_age_s: float = 3.0) -> dict[str, Quote]:
    """Read poller cache; optionally sync-fetch one batch for missing names."""
    if not symbols:
        return {}
    now = time.time()
    out: dict[str, Quote] = {}
    missing: list[str] = []
    with _live_lock:
        for s in symbols:
            hit = _live_cache.get(s)
            if hit and now - hit[0] <= max_age_s:
                out[s] = hit[1]
            else:
                missing.append(s)

    if missing and get_access_token():
        from .session import is_live_data_window

        if is_live_data_window():
            fyers_to_saint = {to_fyers_eq(s): s for s in missing[:FYERS_BATCH_SIZE]}
            batch = _fetch_fyers_batch(list(fyers_to_saint.keys()), fyers_to_saint)
            _store_live_quotes(batch)
            out.update(batch)
    return out


def fetch_fyers_quotes(symbols: list[str]) -> dict[str, Quote]:
    """Batch-fetch LTP/change for Saint symbols. Uses live poller when running."""
    from .session import is_live_data_window

    if not is_live_data_window():
        return {}
    if not get_access_token():
        return {}
    if not symbols:
        return {}

    ensure_fyers_poller(symbols)
    cached = live_fyers_quotes(symbols, max_age_s=5.0)
    if len(cached) >= max(1, len(symbols) // 2):
        return cached

    # Cold start — pull all batches synchronously once.
    fyers_to_saint: dict[str, str] = {}
    fyers_syms: list[str] = []
    for s in symbols:
        f = to_fyers_eq(s)
        fyers_to_saint[f] = s
        fyers_syms.append(f)

    out: dict[str, Quote] = dict(cached)
    for i in range(0, len(fyers_syms), FYERS_BATCH_SIZE):
        batch = fyers_syms[i : i + FYERS_BATCH_SIZE]
        sub_map = {f: fyers_to_saint[f] for f in batch}
        out.update(_fetch_fyers_batch(batch, sub_map))
        if i + FYERS_BATCH_SIZE < len(fyers_syms):
            time.sleep(FYERS_BATCH_INTERVAL_S)
    _store_live_quotes(out)
    return out
