"""Weighted breadth history for Nifty/Sensex (1m + 5m)."""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from zoneinfo import ZoneInfo

from .config import settings
from .session import (
    SESSION_CLOSE,
    SESSION_OPEN,
    is_trading_day,
    now_ist,
    prev_trading_day,
)

_lock = Lock()
_MAX = 3000
_INTERVALS_MIN = (1, 5)
_STORE: dict[str, dict[int, list[dict[str, Any]]]] = {
    "nifty": {1: [], 5: []},
    "sensex": {1: [], 5: []},
}
_loaded = False
IST = ZoneInfo("Asia/Kolkata")
_BACKFILL_ATTEMPTS: dict[tuple[str, str], float] = {}


def _path() -> Path:
    try:
        p = Path(settings.alerts_db).resolve().parent / "nifty_breadth_history.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:  # noqa: BLE001
        return Path("/tmp/saint_nifty_breadth_history.json")


def _load() -> None:
    global _STORE
    path = _path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # Backward compatibility: old schema had only Nifty 5m rows.
                _STORE = {
                    "nifty": {1: [], 5: data[-_MAX:]},
                    "sensex": {1: [], 5: []},
                }
                return
            if isinstance(data, dict):
                next_store = {
                    "nifty": {1: [], 5: []},
                    "sensex": {1: [], 5: []},
                }
                for index_key in ("nifty", "sensex"):
                    payload = data.get(index_key)
                    if not isinstance(payload, dict):
                        continue
                    for interval_m in _INTERVALS_MIN:
                        rows = payload.get(str(interval_m)) or payload.get(interval_m)
                        if isinstance(rows, list):
                            next_store[index_key][interval_m] = rows[-_MAX:]
                _STORE = next_store
                return
    except Exception:  # noqa: BLE001
        _STORE = {
            "nifty": {1: [], 5: []},
            "sensex": {1: [], 5: []},
        }


def _save() -> None:
    try:
        payload = {
            index_key: {
                str(interval_m): rows[-_MAX:]
                for interval_m, rows in interval_map.items()
            }
            for index_key, interval_map in _STORE.items()
        }
        _path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _ensure() -> None:
    global _loaded
    if not _loaded:
        _load()
        _loaded = True


def _bucket_ts(interval_m: int, now: float | None = None) -> int:
    t = int(now if now is not None else time.time())
    bucket_s = max(1, interval_m) * 60
    return t - (t % bucket_s)


def _history_rows(index_key: str, interval_m: int) -> list[dict[str, Any]]:
    k = index_key.lower()
    if k not in _STORE:
        _STORE[k] = {1: [], 5: []}
    if interval_m not in _STORE[k]:
        _STORE[k][interval_m] = []
    return _STORE[k][interval_m]


def _target_session_date() -> date:
    now = now_ist()
    if is_trading_day(now.date()) and now.time() >= SESSION_OPEN:
        return now.date()
    return prev_trading_day(now.date())


def _in_session_window(bucket_ts: int, target_day: date) -> bool:
    ts_ist = datetime.fromtimestamp(int(bucket_ts), tz=timezone.utc).astimezone(IST)
    if ts_ist.date() != target_day:
        return False
    t = ts_ist.time()
    return SESSION_OPEN <= t <= SESSION_CLOSE


def _session_filtered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_day = _target_session_date()
    return [r for r in rows if _in_session_window(int(r.get("bucketTs") or 0), target_day)]


def _rebuild_5m_from_1m(index_key: str) -> None:
    """Derive a complete 5m session tape from stored 1m rows.

    The native 5m writer updates a bucket throughout the 5-minute window and keeps
    the latest minute in that bucket. To mirror this behavior from 1m history,
    keep the last 1m row in each 5m bucket and relabel timestamp to the bucket start.
    """
    target_day = _target_session_date()
    one_min_rows = _session_filtered(_history_rows(index_key, 1))
    if not one_min_rows:
        return
    by_bucket: dict[int, dict[str, Any]] = {}
    for row in one_min_rows:
        ts = int(row.get("bucketTs") or 0)
        if ts <= 0:
            continue
        bucket = ts - (ts % (5 * 60))
        prev = by_bucket.get(bucket)
        # Keep the latest 1m row within the bucket window.
        if prev is None or int(prev.get("bucketTs") or 0) <= ts:
            by_bucket[bucket] = row

    rebuilt: list[dict[str, Any]] = []
    prev_weight_up: float | None = None
    prev_contrib: float | None = None
    prev_status: str | None = None
    for bucket in sorted(by_bucket.keys()):
        source = by_bucket[bucket]
        bucket_dt_ist = datetime.fromtimestamp(bucket, tz=timezone.utc).astimezone(IST)
        if not _in_session_window(bucket, target_day):
            continue
        weight_up = _as_float(source.get("weightUp"))
        weight_down = _as_float(source.get("weightDown"))
        contrib = _as_float(source.get("contributionPct"))
        status, dw_up_1, d_contrib_1 = _breadth_status(
            weight_up=weight_up,
            weight_down=weight_down,
            contrib=contrib,
            prev_weight_up=prev_weight_up,
            prev_contrib=prev_contrib,
            prev_status=prev_status,
        )
        rebuilt.append(
            {
                "bucketTs": bucket,
                "t": bucket_dt_ist.strftime("%H:%M"),
                "asOf": source.get("asOf"),
                "weightUp": weight_up,
                "weightDown": weight_down,
                "weightFlat": source.get("weightFlat"),
                "contributionPct": contrib,
                "advances": source.get("advances"),
                "declines": source.get("declines"),
                "lean": source.get("lean"),
                "dwUp1": dw_up_1,
                "dContrib1": d_contrib_1,
                "breadthStatus": status,
            }
        )
        prev_weight_up = weight_up
        prev_contrib = contrib
        prev_status = status

    replace_session_history(
        index_key=index_key,
        interval_minutes=5,
        session_day=target_day,
        rows=rebuilt,
    )


def replace_session_history(
    *,
    index_key: str,
    interval_minutes: int,
    session_day: date,
    rows: list[dict[str, Any]],
) -> None:
    """Replace one session's rows for an index/interval with supplied snapshots."""
    _ensure()
    wanted_interval = 1 if int(interval_minutes) == 1 else 5
    with _lock:
        store_rows = _history_rows(index_key, wanted_interval)
        keep = [
            r
            for r in store_rows
            if not _in_session_window(int(r.get("bucketTs") or 0), session_day)
        ]
        merged = sorted([*keep, *rows], key=lambda x: int(x.get("bucketTs") or 0))
        if len(merged) > _MAX:
            merged = merged[-_MAX:]
        _STORE[index_key.lower()][wanted_interval] = merged
        _save()


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _breadth_status(
    *,
    weight_up: float | None,
    weight_down: float | None,
    contrib: float | None,
    prev_weight_up: float | None,
    prev_contrib: float | None,
    prev_status: str | None,
) -> tuple[str, float | None, float | None]:
    if (
        weight_up is None
        or weight_down is None
        or contrib is None
        or prev_weight_up is None
        or prev_contrib is None
    ):
        return "Normal", None, None

    dw_up = weight_up - prev_weight_up
    d_contrib = contrib - prev_contrib
    bull_dominant = weight_up >= (weight_down + 5.0)
    bear_dominant = weight_down >= (weight_up + 5.0)

    # Shock tags catch sudden participation breaks that matter intraday even
    # when aggregate contribution hasn't crossed a larger threshold yet.
    if bear_dominant and (d_contrib <= -0.07 or (dw_up <= -6.0 and d_contrib <= -0.04)):
        return "Bearish shock", dw_up, d_contrib
    if bull_dominant and (d_contrib >= 0.07 or (dw_up >= 6.0 and d_contrib >= 0.04)):
        return "Bullish shock", dw_up, d_contrib

    # Build-up tags highlight expanding participation in one direction.
    if bull_dominant and (dw_up >= 2.0 and d_contrib >= 0.05):
        return "Bullish buildup", dw_up, d_contrib
    if bear_dominant and (dw_up <= -2.0 and d_contrib <= -0.05):
        return "Bearish buildup", dw_up, d_contrib

    # Post tags mark likely cooldown/fade after a prior expansion pulse.
    if prev_status == "Bullish buildup" and (dw_up <= -1.0 or d_contrib <= -0.05):
        return "Post bullish", dw_up, d_contrib
    if prev_status == "Bearish buildup" and (dw_up >= 1.0 or d_contrib >= 0.05):
        return "Post bearish", dw_up, d_contrib

    return "Normal", dw_up, d_contrib


def record_breadth_snapshot(
    snap: dict[str, Any],
    *,
    index_key: str = "nifty",
    limit: int = 5,
    interval_minutes: int = 5,
) -> list[dict[str, Any]]:
    """Upsert current 1m/5m buckets; return newest-first rows for requested interval."""
    _ensure()
    now_ist = datetime.now(timezone.utc).astimezone(IST)
    wanted_interval = 1 if int(interval_minutes) == 1 else 5
    with _lock:
        for interval_m in _INTERVALS_MIN:
            bucket = _bucket_ts(interval_m)
            bucket_dt_ist = datetime.fromtimestamp(bucket, tz=timezone.utc).astimezone(IST)
            rows = _history_rows(index_key, interval_m)
            replacing_latest = bool(rows and int(rows[-1].get("bucketTs") or 0) == bucket)
            prev_row = rows[-2] if replacing_latest and len(rows) >= 2 else (rows[-1] if rows else None)

            weight_up = _as_float(snap.get("weightUp"))
            weight_down = _as_float(snap.get("weightDown"))
            contrib = _as_float(snap.get("contributionPct"))
            prev_weight_up = _as_float(prev_row.get("weightUp")) if prev_row else None
            prev_contrib = _as_float(prev_row.get("contributionPct")) if prev_row else None
            prev_status = str(prev_row.get("breadthStatus")) if prev_row and prev_row.get("breadthStatus") else None
            status, dw_up_1, d_contrib_1 = _breadth_status(
                weight_up=weight_up,
                weight_down=weight_down,
                contrib=contrib,
                prev_weight_up=prev_weight_up,
                prev_contrib=prev_contrib,
                prev_status=prev_status,
            )

            row = {
                "bucketTs": bucket,
                "t": bucket_dt_ist.strftime("%H:%M"),
                "asOf": now_ist.isoformat(),
                "weightUp": weight_up,
                "weightDown": weight_down,
                "weightFlat": snap.get("weightFlat"),
                "contributionPct": contrib,
                "advances": snap.get("advances"),
                "declines": snap.get("declines"),
                "lean": snap.get("lean"),
                "dwUp1": dw_up_1,
                "dContrib1": d_contrib_1,
                "breadthStatus": status,
            }
            if rows and int(rows[-1].get("bucketTs") or 0) == bucket:
                rows[-1] = row
            else:
                rows.append(row)
                if len(rows) > _MAX:
                    del rows[: len(rows) - _MAX]
        _save()
        rows = _history_rows(index_key, wanted_interval)
        return list(reversed(rows[-max(1, int(limit)):]))


def breadth_history(
    limit: int = 5,
    *,
    index_key: str = "nifty",
    interval_minutes: int = 5,
) -> list[dict[str, Any]]:
    _ensure()
    wanted_interval = 1 if int(interval_minutes) == 1 else 5
    with _lock:
        raw_rows = _session_filtered(_history_rows(index_key, wanted_interval))

    # Recompute delta/status fields on read so historical rows from older logic
    # are normalized without requiring a full rewrite cycle.
    normalized_rows: list[dict[str, Any]] = []
    prev_weight_up: float | None = None
    prev_contrib: float | None = None
    prev_status: str | None = None
    for row in raw_rows:
        r = dict(row)
        weight_up = _as_float(r.get("weightUp"))
        weight_down = _as_float(r.get("weightDown"))
        contrib = _as_float(r.get("contributionPct"))
        status, dw_up_1, d_contrib_1 = _breadth_status(
            weight_up=weight_up,
            weight_down=weight_down,
            contrib=contrib,
            prev_weight_up=prev_weight_up,
            prev_contrib=prev_contrib,
            prev_status=prev_status,
        )
        r["dwUp1"] = dw_up_1
        r["dContrib1"] = d_contrib_1
        r["breadthStatus"] = status
        normalized_rows.append(r)
        prev_weight_up = weight_up
        prev_contrib = contrib
        prev_status = status

    out = list(reversed(normalized_rows[-max(1, int(limit)):]))

    # Reconstruct from cached constituent bars whenever current-session 1m history is sparse.
    if wanted_interval == 1 and len(out) < 350:
        target_day = _target_session_date().isoformat()
        key = (index_key.lower(), target_day)
        now = time.time()
        last_attempt = float(_BACKFILL_ATTEMPTS.get(key) or 0.0)
        if now - last_attempt > 120.0:
            _BACKFILL_ATTEMPTS[key] = now
            try:
                from .breadth_backfill import rebuild_breadth_1m_from_cached_data

                rebuild_breadth_1m_from_cached_data(index_key=index_key)
            except Exception:  # noqa: BLE001
                pass
            with _lock:
                rows = _session_filtered(_history_rows(index_key, wanted_interval))
            normalized_rows = []
            prev_weight_up = None
            prev_contrib = None
            prev_status = None
            for row in rows:
                r = dict(row)
                weight_up = _as_float(r.get("weightUp"))
                weight_down = _as_float(r.get("weightDown"))
                contrib = _as_float(r.get("contributionPct"))
                status, dw_up_1, d_contrib_1 = _breadth_status(
                    weight_up=weight_up,
                    weight_down=weight_down,
                    contrib=contrib,
                    prev_weight_up=prev_weight_up,
                    prev_contrib=prev_contrib,
                    prev_status=prev_status,
                )
                r["dwUp1"] = dw_up_1
                r["dContrib1"] = d_contrib_1
                r["breadthStatus"] = status
                normalized_rows.append(r)
                prev_weight_up = weight_up
                prev_contrib = contrib
                prev_status = status
            out = list(reversed(normalized_rows[-max(1, int(limit)):]))

    # 5m view should reflect full session too; derive from 1m when sparse.
    if wanted_interval == 5 and len(out) < 70:
        target_day = _target_session_date().isoformat()
        key = (f"{index_key.lower()}_5m", target_day)
        now = time.time()
        last_attempt = float(_BACKFILL_ATTEMPTS.get(key) or 0.0)
        if now - last_attempt > 120.0:
            _BACKFILL_ATTEMPTS[key] = now
            try:
                # Ensure 1m session is complete first, then aggregate into 5m.
                from .breadth_backfill import rebuild_breadth_1m_from_cached_data

                rebuild_breadth_1m_from_cached_data(index_key=index_key)
                _rebuild_5m_from_1m(index_key)
            except Exception:  # noqa: BLE001
                pass
            with _lock:
                rows = _session_filtered(_history_rows(index_key, wanted_interval))
            normalized_rows = []
            prev_weight_up = None
            prev_contrib = None
            prev_status = None
            for row in rows:
                r = dict(row)
                weight_up = _as_float(r.get("weightUp"))
                weight_down = _as_float(r.get("weightDown"))
                contrib = _as_float(r.get("contributionPct"))
                status, dw_up_1, d_contrib_1 = _breadth_status(
                    weight_up=weight_up,
                    weight_down=weight_down,
                    contrib=contrib,
                    prev_weight_up=prev_weight_up,
                    prev_contrib=prev_contrib,
                    prev_status=prev_status,
                )
                r["dwUp1"] = dw_up_1
                r["dContrib1"] = d_contrib_1
                r["breadthStatus"] = status
                normalized_rows.append(r)
                prev_weight_up = weight_up
                prev_contrib = contrib
                prev_status = status
            out = list(reversed(normalized_rows[-max(1, int(limit)):]))

    return out
