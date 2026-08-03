"""5-minute weighted breadth history for Nifty page."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any

from .config import settings

_lock = Lock()
_HISTORY: list[dict[str, Any]] = []
_MAX = 24
_BUCKET_S = 5 * 60
_loaded = False


def _path() -> Path:
    try:
        p = Path(settings.alerts_db).resolve().parent / "nifty_breadth_history.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:  # noqa: BLE001
        return Path("/tmp/saint_nifty_breadth_history.json")


def _load() -> None:
    global _HISTORY
    path = _path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                _HISTORY = data[-_MAX:]
    except Exception:  # noqa: BLE001
        _HISTORY = []


def _save() -> None:
    try:
        _path().write_text(json.dumps(_HISTORY[-_MAX:], indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _ensure() -> None:
    global _loaded
    if not _loaded:
        _load()
        _loaded = True


def _bucket_ts(now: float | None = None) -> int:
    t = int(now if now is not None else time.time())
    return t - (t % _BUCKET_S)


def record_breadth_snapshot(snap: dict[str, Any]) -> list[dict[str, Any]]:
    """Upsert current 5m bucket; return newest-first history (5 rows for UI)."""
    _ensure()
    bucket = _bucket_ts()
    row = {
        "bucketTs": bucket,
        "t": time.strftime("%H:%M", time.localtime(bucket)),
        "asOf": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weightUp": snap.get("weightUp"),
        "weightDown": snap.get("weightDown"),
        "weightFlat": snap.get("weightFlat"),
        "contributionPct": snap.get("contributionPct"),
        "advances": snap.get("advances"),
        "declines": snap.get("declines"),
        "lean": snap.get("lean"),
    }
    with _lock:
        if _HISTORY and int(_HISTORY[-1].get("bucketTs") or 0) == bucket:
            _HISTORY[-1] = row
        else:
            _HISTORY.append(row)
            if len(_HISTORY) > _MAX:
                del _HISTORY[: len(_HISTORY) - _MAX]
        _save()
        return list(reversed(_HISTORY[-5:]))


def breadth_history(limit: int = 5) -> list[dict[str, Any]]:
    _ensure()
    with _lock:
        return list(reversed(_HISTORY[-limit:]))
