"""Short in-memory series for 5m / 15m trend arrows.

Survives only while the process is up — fine for session monitoring.
Samples are deduped to ~30s so dashboard polls don't spam identical points.
"""

from __future__ import annotations

import time
from typing import Literal

TrendDir = Literal["up", "down", "flat"]

_MIN_GAP_S = 25.0
_KEEP_S = 20 * 60
_series: dict[str, list[tuple[float, float]]] = {}


def record(name: str, value: float, *, now: float | None = None) -> None:
    ts = now if now is not None else time.time()
    buf = _series.setdefault(name, [])
    if buf and ts - buf[-1][0] < _MIN_GAP_S:
        # Refresh last point so cold caches still age correctly.
        buf[-1] = (ts, float(value))
    else:
        buf.append((ts, float(value)))
    cutoff = ts - _KEEP_S
    _series[name] = [(t, v) for t, v in buf if t >= cutoff]


def _value_near(name: str, lag_s: float, *, now: float | None = None) -> float | None:
    ts = now if now is not None else time.time()
    buf = _series.get(name) or []
    if len(buf) < 2:
        return None
    target = ts - lag_s
    # Prefer closest sample at or before target; else earliest.
    earlier = [p for p in buf if p[0] <= target]
    if earlier:
        return earlier[-1][1]
    # Not enough age yet — need at least ~70% of the lag covered.
    oldest_t = buf[0][0]
    if ts - oldest_t < lag_s * 0.7:
        return None
    return buf[0][1]


def direction(
    name: str,
    *,
    current: float,
    lag_s: float,
    eps: float,
    now: float | None = None,
) -> TrendDir | None:
    past = _value_near(name, lag_s, now=now)
    if past is None:
        return None
    delta = current - past
    if delta > eps:
        return "up"
    if delta < -eps:
        return "down"
    return "flat"


def trend_pair(
    name: str,
    *,
    current: float,
    eps: float,
    now: float | None = None,
) -> dict:
    """Return `{m5, m15}` each up/down/flat/null for UI arrows."""
    return {
        "m5": direction(name, current=current, lag_s=5 * 60, eps=eps, now=now),
        "m15": direction(name, current=current, lag_s=15 * 60, eps=eps, now=now),
    }
