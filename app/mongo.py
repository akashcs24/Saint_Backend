"""MongoDB helpers for Saint (Atlas).

Uses SAINT_MONGODB_URI / SAINT_MONGODB_DB. Local macOS Python often needs
certifi's CA bundle for Atlas TLS — we always pass tlsCAFile when available.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from .config import settings

_COLLECTION = "nifty_paper_trades"
_MONGO_OK: dict[str, float | bool] = {"ts": 0.0, "ok": False}
_MONGO_OK_TTL_S = 60.0


def mongodb_configured() -> bool:
    return bool((getattr(settings, "mongodb_uri", "") or "").strip())


@lru_cache(maxsize=1)
def _client():
    uri = (settings.mongodb_uri or "").strip()
    if not uri:
        return None
    from pymongo import MongoClient

    kwargs: dict[str, Any] = {
        "serverSelectionTimeoutMS": 8_000,
        "connectTimeoutMS": 8_000,
    }
    try:
        import certifi

        kwargs["tlsCAFile"] = certifi.where()
    except Exception:  # noqa: BLE001
        pass
    return MongoClient(uri, **kwargs)


def mongo_db():
    client = _client()
    if client is None:
        return None
    name = (getattr(settings, "mongodb_db", "") or "saint").strip() or "saint"
    return client[name]


def paper_trades_collection():
    db = mongo_db()
    if db is None:
        return None
    return db[_COLLECTION]


def mongo_ping() -> dict[str, Any]:
    """Best-effort connectivity check for /health or diagnostics."""
    if not mongodb_configured():
        return {"configured": False, "ok": False}
    try:
        client = _client()
        if client is None:
            return {"configured": True, "ok": False, "error": "no client"}
        client.admin.command("ping")
        return {
            "configured": True,
            "ok": True,
            "db": getattr(settings, "mongodb_db", "saint"),
            "collection": _COLLECTION,
        }
    except Exception as exc:  # noqa: BLE001
        return {"configured": True, "ok": False, "error": str(exc)}


def mongo_is_reachable() -> bool:
    """Cached ping — skip Atlas when Render credentials/network are bad."""
    if not mongodb_configured():
        return False
    now = time.time()
    if now - float(_MONGO_OK["ts"]) < _MONGO_OK_TTL_S:
        return bool(_MONGO_OK["ok"])
    ok = bool(mongo_ping().get("ok"))
    _MONGO_OK["ts"] = now
    _MONGO_OK["ok"] = ok
    return ok


def reset_mongo_client_cache() -> None:
    """Clear cached client (tests / after env change)."""
    _client.cache_clear()
    _MONGO_OK["ts"] = 0.0
    _MONGO_OK["ok"] = False
