from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .ai_helper import ai_configured, run_ai_helper
from .alerts import alerts_enabled, run_alert_tick, send_test_alert, telegram_configured
from .config import settings
from .fyers_auth import (
    auth_login_url,
    clear_access_token,
    exchange_auth_code,
    fyers_configured,
    fyers_status,
    get_access_token,
)
from .service import build_prices, build_stock_detail, get_dashboard
from pydantic import BaseModel, Field

app = FastAPI(title="Saint Infinite Market API", version="0.1.0")


@app.on_event("startup")
def _startup_fyers_poller() -> None:
    """Warm Fyers batch poller when a token is already on disk."""
    try:
        from .fyers_quotes import ensure_fyers_poller
        from .nifty_weights import get_nifty_weights
        from .universe import UNIVERSE

        if get_access_token():
            weights = get_nifty_weights().get("weights") or {}
            symbols = [s for s in weights if s in UNIVERSE]
            ensure_fyers_poller(symbols)
    except Exception:  # noqa: BLE001
        pass


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


def _mongo_health() -> dict:
    try:
        from .mongo import mongo_ping

        return mongo_ping()
    except Exception as exc:  # noqa: BLE001
        return {"configured": bool(getattr(settings, "mongodb_uri", "")), "ok": False, "error": str(exc)}


@app.api_route("/health", methods=["GET", "HEAD"])
def health(
    tick: bool = Query(False),
    key: str | None = Query(None),
    x_saint_alerts_key: str | None = Header(None),
):
    """Liveness probe. Optional ``?tick=1&key=...`` kicks alert + Nifty paper tick in background.

    Free UptimeRobot HEAD probes time out if we block on Yahoo/dashboard.
    Always return /health quickly; run ENTRY/EXIT and Nifty board (paper trades) on a daemon thread.
    """
    tick_started = False
    if tick:
        _check_alerts_secret(key, x_saint_alerts_key)
        import threading

        def _bg_tick() -> None:
            # Alerts + Nifty board/paper trades (paper runs even if Telegram is off).
            try:
                run_alert_tick(force_dashboard=False)
            except Exception:  # noqa: BLE001
                pass
            try:
                from .nifty_board import _kick_rebuild

                _kick_rebuild(force=False)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(
            target=_bg_tick,
            daemon=True,
            name="saint-market-tick",
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
        "fyersConfigured": fyers_configured(),
        "fyersConnected": bool(fyers_status(verify=False).get("connected")),
        "mongo": _mongo_health(),
    }


class FyersExchangeBody(BaseModel):
    code: str = Field(..., min_length=1, description="Auth code or full redirect URL")


@app.get("/api/fyers/status")
def api_fyers_status(force: bool = Query(False)):
    """Live Fyers status. Green only after a successful quote probe (not stale token file)."""
    return fyers_status(verify=True, force_verify=force)


@app.get("/api/fyers/auth-url")
def api_fyers_auth_url():
    try:
        url = auth_login_url()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    # Don't block login URL on a live probe.
    return {"url": url, **fyers_status(verify=False)}


@app.post("/api/fyers/exchange")
def api_fyers_exchange(body: FyersExchangeBody):
    try:
        return exchange_auth_code(body.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/fyers/logout")
def api_fyers_logout():
    clear_access_token()
    return fyers_status(verify=False)


@app.get("/api/dashboard")
def dashboard(force: bool = Query(False)):
    """Board payload. Never bare-500 on cold start — returns empty board + error."""
    try:
        return get_dashboard(force=force)
    except Exception as exc:  # noqa: BLE001
        # Last-resort shield so Vercel stops spinning on uncaught bugs.
        from .service import _empty_dashboard

        return _empty_dashboard(error=f"{type(exc).__name__}: {exc}")


@app.get("/api/nifty")
def nifty_board(force: bool = Query(False)):
    """Nifty 50 page: index, breadth, drivers, OI wings, PCR history, lead/lag."""
    try:
        from .nifty_board import get_nifty_board

        return get_nifty_board(force=force)
    except Exception as exc:  # noqa: BLE001
        from .nifty_board import _empty_nifty_board

        return _empty_nifty_board(error=f"{type(exc).__name__}: {exc}")


@app.get("/api/nifty/paper-trades")
def nifty_paper_trades(limit: int = Query(40, ge=1, le=200)):
    """List paper trades bucketed by strategy (decline, tsl, …)."""
    try:
        from .nifty_paper_trades import paper_trades_board

        return paper_trades_board(limit_per=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/nifty/paper-trades/tick")
def nifty_paper_trades_tick(force: bool = Query(True)):
    """Force-evaluate every paper strategy and capture Fyers ATM CE prices."""
    try:
        from .nifty_paper_trades import tick_paper_trades

        return tick_paper_trades(force=force)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


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
