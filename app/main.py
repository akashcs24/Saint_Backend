from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .ai_helper import ai_configured, run_ai_helper
from .alerts import alerts_enabled, run_alert_tick, send_test_alert, telegram_configured
from .config import settings
from .service import build_prices, build_stock_detail, get_dashboard

app = FastAPI(title="Saint Infinite Market API", version="0.1.0")

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_alerts_secret(
    key: str | None = None,
    x_saint_alerts_key: str | None = None,
) -> None:
    expected = (settings.alerts_secret or "").strip()
    if not expected:
        return
    provided = (key or x_saint_alerts_key or "").strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid alerts key")


@app.api_route("/health", methods=["GET", "HEAD"])
def health(
    tick: bool = Query(False),
    key: str | None = Query(None),
    x_saint_alerts_key: str | None = Header(None),
):
    """Liveness probe. Optional ``?tick=1&key=...`` kicks alert scan in background.

    Free UptimeRobot HEAD probes time out if we block on Yahoo/dashboard.
    Always return /health quickly; run ENTRY/EXIT on a daemon thread.
    """
    tick_started = False
    if tick:
        _check_alerts_secret(key, x_saint_alerts_key)
        import threading

        threading.Thread(
            target=lambda: run_alert_tick(force_dashboard=False),
            daemon=True,
            name="saint-alert-tick",
        ).start()
        tick_started = True
    from .service import _DASH_CACHE

    return {
        "ok": True,
        "priceCache": str(settings.price_cache),
        "priceCacheExists": settings.price_cache.exists(),
        "openaiConfigured": ai_configured(),
        "aiConfigured": ai_configured(),
        "telegramConfigured": telegram_configured(),
        "alertsEnabled": alerts_enabled(),
        "tickRequested": bool(tick),
        "tickStarted": tick_started,
        "dashboardCached": _DASH_CACHE.get("payload") is not None,
        "dashboardBuilding": bool(_DASH_CACHE.get("building")),
        "dashboardLastError": _DASH_CACHE.get("lastError"),
    }


@app.get("/api/dashboard")
def dashboard(force: bool = Query(False)):
    """Board payload. Never bare-500 on cold start — returns empty board + error."""
    try:
        return get_dashboard(force=force)
    except Exception as exc:  # noqa: BLE001
        # Last-resort shield so Vercel stops spinning on uncaught bugs.
        from .service import _empty_dashboard

        return _empty_dashboard(error=f"{type(exc).__name__}: {exc}")


@app.get("/api/stocks/{symbol}")
def stock_detail(symbol: str):
    data = build_stock_detail(symbol)
    if not data:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")
    return data


@app.post("/api/stocks/{symbol}/ai-helper")
def stock_ai_helper(symbol: str, force: bool = Query(False)):
    """On-demand OpenAI helper grounded in Saint's stock packet."""
    data = build_stock_detail(symbol)
    if not data:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")
    result = run_ai_helper(
        data["stock"],
        data.get("news") or [],
        data.get("context") or [],
        force=force,
    )
    if not result.get("ready") and result.get("error") == "not_configured":
        raise HTTPException(status_code=503, detail=result.get("message") or "OpenAI not configured")
    if not result.get("ready") and result.get("error") in {
        "openai_http",
        "gemini_http",
        "request_failed",
        "bad_json",
    }:
        raise HTTPException(status_code=502, detail=result.get("message") or "AI helper failed")
    return result


@app.get("/api/prices/{symbol}")
def prices(symbol: str, range: str = Query("1M", alias="range")):
    return build_prices(symbol, range_key=range)


@app.api_route("/api/alerts/tick", methods=["GET", "POST", "HEAD"])
def alerts_tick(
    force: bool = Query(False),
    key: str | None = Query(None),
    x_saint_alerts_key: str | None = Header(None),
):
    """Scan board and push high-bar Telegram alerts.

    Point UptimeRobot (every 5–10 min) at this URL during market hours.
    HEAD is supported so free UptimeRobot probes still run the scan.
    """
    _check_alerts_secret(key, x_saint_alerts_key)
    return run_alert_tick(force_dashboard=force)


@app.post("/api/alerts/test")
def alerts_test(
    key: str | None = Query(None),
    x_saint_alerts_key: str | None = Header(None),
):
    """Send a one-line Telegram test message."""
    _check_alerts_secret(key, x_saint_alerts_key)
    if not telegram_configured():
        raise HTTPException(status_code=503, detail="Telegram not configured")
    result = send_test_alert()
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result)
    return result
