from __future__ import annotations

import threading
import time
from datetime import datetime

from .market_minute_store import record_index_quote
from .nifty_breadth import build_nifty_breadth, build_sensex_breadth
from .quotes import get_quote
from .session import is_cash_session_open, now_ist

_worker: dict[str, object] = {"thread": None, "running": False, "last_minute": ""}
_lock = threading.Lock()


def _tick_once() -> None:
    # Builds use Fyers-first with Yahoo fallback and now persist constituent snapshots.
    build_nifty_breadth(force=True)
    build_sensex_breadth(force=True)

    nq = get_quote("NIFTY")
    sq = get_quote("SENSEX")
    record_index_quote("nifty", nq)
    record_index_quote("sensex", sq)


def _loop() -> None:
    while bool(_worker.get("running")):
        try:
            now = now_ist()
            if is_cash_session_open(now):
                minute_key = now.strftime("%Y-%m-%d %H:%M")
                last_minute = str(_worker.get("last_minute") or "")
                if minute_key != last_minute:
                    _worker["last_minute"] = minute_key
                    _tick_once()
            time.sleep(5)
        except Exception:  # noqa: BLE001
            time.sleep(5)


def ensure_market_minute_worker() -> None:
    with _lock:
        th = _worker.get("thread")
        if isinstance(th, threading.Thread) and th.is_alive():
            return
        _worker["running"] = True
        t = threading.Thread(target=_loop, daemon=True, name="saint-market-minute-worker")
        _worker["thread"] = t
        t.start()


def stop_market_minute_worker() -> None:
    with _lock:
        _worker["running"] = False
