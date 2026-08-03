"""5-minute PCR / Nifty board snapshot history (in-memory + light disk)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any

from .config import settings

_lock = Lock()
_HISTORY: list[dict[str, Any]] = []
_MAX = 48  # ~4 hours of 5m bars
_BUCKET_S = 5 * 60


def _path() -> Path:
    try:
        p = Path(settings.alerts_db).resolve().parent / "nifty_pcr_history.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:  # noqa: BLE001
        return Path("/tmp/saint_nifty_pcr_history.json")


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


_loaded = False


def _ensure() -> None:
    global _loaded
    if not _loaded:
        _load()
        _loaded = True


def _bucket_ts(now: float | None = None) -> int:
    t = int(now if now is not None else time.time())
    return t - (t % _BUCKET_S)


def record_pcr_snapshot(snap: dict[str, Any]) -> list[dict[str, Any]]:
    """Upsert current 5m bucket; return newest-first history (max 10 for UI)."""
    _ensure()
    bucket = _bucket_ts()
    row = {
        "bucketTs": bucket,
        "t": time.strftime("%H:%M", time.localtime(bucket)),
        "asOf": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "oiPcr": snap.get("oiPcr"),
        "volumePcr": snap.get("volumePcr"),
        "putOi": snap.get("putOi"),
        "callOi": snap.get("callOi"),
        "spot": snap.get("spot"),
        "lean": snap.get("lean"),
        "ceOiWing": snap.get("ceOiWing"),
        "peOiWing": snap.get("peOiWing"),
        "ceOiChgWing": snap.get("ceOiChgWing"),
        "peOiChgWing": snap.get("peOiChgWing"),
        "insight": snap.get("insight"),
    }
    with _lock:
        if _HISTORY and int(_HISTORY[-1].get("bucketTs") or 0) == bucket:
            prev_insight = _HISTORY[-1].get("insight")
            _HISTORY[-1] = row
            if not row.get("insight") and prev_insight:
                _HISTORY[-1]["insight"] = prev_insight
        else:
            # insight vs previous bucket
            if len(_HISTORY) >= 1 and not row.get("insight"):
                row["insight"] = _delta_insight(_HISTORY[-1], row)
            _HISTORY.append(row)
            if len(_HISTORY) > _MAX:
                del _HISTORY[: len(_HISTORY) - _MAX]
        _save()
        return list(reversed(_HISTORY[-10:]))


def _delta_insight(prev: dict[str, Any], cur: dict[str, Any]) -> str:
    bits: list[str] = []
    try:
        dp = float(cur.get("oiPcr") or 0) - float(prev.get("oiPcr") or 0)
        if abs(dp) >= 0.01:
            bits.append(f"PCR {dp:+.3f}")
    except (TypeError, ValueError):
        pass
    try:
        if cur.get("spot") is not None and prev.get("spot") is not None:
            move = (float(cur["spot"]) - float(prev["spot"])) / float(prev["spot"]) * 100
            if abs(move) >= 0.03:
                bits.append(f"spot {move:+.2f}%")
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    try:
        dpe = int(cur.get("peOiWing") or 0) - int(prev.get("peOiWing") or 0)
        dce = int(cur.get("ceOiWing") or 0) - int(prev.get("ceOiWing") or 0)
        if abs(dpe) + abs(dce) > 0:
            bits.append(f"wing OI PE {dpe:+,} / CE {dce:+,}")
    except (TypeError, ValueError):
        pass
    if not bits:
        return "Quiet — little change vs prior 5m."
    # Meaning
    meaning = ""
    try:
        dp = float(cur.get("oiPcr") or 0) - float(prev.get("oiPcr") or 0)
        if dp >= 0.02:
            meaning = " → put-side OI relative build (often supportive)."
        elif dp <= -0.02:
            meaning = " → call-side OI relative build (often resistive)."
    except (TypeError, ValueError):
        pass
    return "; ".join(bits) + meaning


def pcr_history(limit: int = 10) -> list[dict[str, Any]]:
    _ensure()
    with _lock:
        return list(reversed(_HISTORY[-limit:]))
