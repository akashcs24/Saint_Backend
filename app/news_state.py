"""In-memory master switch for cross-source news fetching (default off)."""

from __future__ import annotations

from threading import Lock

_LOCK = Lock()
_ENABLED = False


def news_fetch_enabled(*, force: bool = False) -> bool:
    del force  # kept for API compatibility
    with _LOCK:
        return bool(_ENABLED)


def set_news_fetch_enabled(enabled: bool) -> bool:
    with _LOCK:
        global _ENABLED
        _ENABLED = bool(enabled)
        return bool(_ENABLED)
