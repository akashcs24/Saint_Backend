"""Fyers OAuth for Saint — login URL, auth-code exchange, token store.

Used so Nifty breadth can pull realtime quotes when connected.
Credentials: FYERS_APP_ID / FYERS_SECRET_KEY / FYERS_REDIRECT_URI (env).
Access token lives in memory + a small local file (survives process restart on
the same host; Render free /tmp is wiped on redeploy — re-login after deploy).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import settings

_lock = Lock()
_STATE: dict[str, Any] = {
    "accessToken": None,
    "connectedAt": None,
    "lastError": None,
    # After auth failure, ignore disk/env token until a fresh exchange.
    "revoked": False,
}
# Last live probe — green UI only when this is ok (not merely "token file exists").
_PROBE: dict[str, Any] = {"ts": 0.0, "ok": False, "error": None}
_PROBE_TTL_S = 45.0
_DOTENV_LOADED = False

_AUTH_FAIL_MARKERS = (
    "please provide valid token",
    "invalid token",
    "token expired",
    "access token",
    "unauthorized",
    "authentication",
    "code: -16",
    "code:-16",
    '"code":-16',
    '"code": -16',
)


def _ensure_dotenv() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_path = Path(__file__).resolve().parent.parent / ".env"
    try:
        from dotenv import load_dotenv

        if env_path.exists():
            load_dotenv(env_path, override=False)
    except ImportError:
        # Minimal parser if python-dotenv not installed
        if not env_path.exists():
            return
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        except Exception:  # noqa: BLE001
            pass


def _env(name: str, default: str = "") -> str:
    _ensure_dotenv()
    # Prefer FYERS_*; allow SAINT_FYERS_* for Render naming consistency.
    return (
        os.getenv(name, "").strip()
        or os.getenv(f"SAINT_{name}", "").strip()
        or default
    )


def fyers_creds() -> dict[str, str]:
    return {
        "app_id": _env("FYERS_APP_ID"),
        "secret": _env("FYERS_SECRET_KEY"),
        "redirect": _env(
            "FYERS_REDIRECT_URI",
            "https://trade.fyers.in/api-login/redirect-uri/index.html",
        ),
    }


def fyers_configured() -> bool:
    c = fyers_creds()
    return bool(c["app_id"] and c["secret"])


def _token_path() -> Path:
    """Prefer durable backend/data over TMPDIR (macOS TMP is easy to lose on restart)."""
    durable = Path(settings.alerts_db).resolve().parent / "fyers_token.json"
    raw = _env("FYERS_TOKEN_PATH")
    if raw:
        p = Path(raw)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:  # noqa: BLE001
            pass
    try:
        durable.parent.mkdir(parents=True, exist_ok=True)
        return durable
    except Exception:  # noqa: BLE001
        return Path(os.getenv("TMPDIR") or "/tmp") / "saint_fyers_token.json"


def _jwt_unexpired(token: str) -> bool:
    """True if token has no exp claim or exp is still in the future."""
    try:
        parts = (token or "").strip().split(".")
        if len(parts) < 2:
            return True  # opaque token — let Fyers decide
        import base64

        seg = parts[1]
        pad = "=" * (-len(seg) % 4)
        payload = json.loads(base64.urlsafe_b64decode(seg + pad))
        exp = payload.get("exp")
        if exp is None:
            return True
        return float(exp) > time.time() + 30
    except Exception:  # noqa: BLE001
        return True


def _jwt_meta(token: str) -> dict[str, Any]:
    try:
        parts = (token or "").strip().split(".")
        if len(parts) < 2:
            return {}
        import base64

        seg = parts[1]
        pad = "=" * (-len(seg) % 4)
        payload = json.loads(base64.urlsafe_b64decode(seg + pad))
        exp = payload.get("exp")
        iat = payload.get("iat")
        return {
            "exp": (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(exp)) if exp else None
            ),
            "iat": (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(iat)) if iat else None
            ),
            "expired": bool(exp and float(exp) <= time.time()),
            "fyId": payload.get("fy_id"),
        }
    except Exception:  # noqa: BLE001
        return {}


def _load_token_from_disk() -> str | None:
    """Prefer fresh token file; ignore expired JWT sitting in .env."""
    path = _token_path()
    file_tok = None
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            file_tok = (data.get("accessToken") or "").strip() or None
    except Exception:  # noqa: BLE001
        file_tok = None

    if file_tok and _jwt_unexpired(file_tok):
        return file_tok

    env_tok = _env("FYERS_ACCESS_TOKEN")
    if env_tok and _jwt_unexpired(env_tok):
        return env_tok

    # Both missing/expired — do not resurrect a dead .env JWT.
    return None


def _save_token(token: str) -> None:
    path = _token_path()
    meta = _jwt_meta(token)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "accessToken": token,
                    "savedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    **{k: v for k, v in meta.items() if k != "fyId"},
                    "fyId": meta.get("fyId"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass


def get_access_token() -> str | None:
    with _lock:
        if _STATE.get("revoked"):
            return None
        if _STATE["accessToken"]:
            tok = str(_STATE["accessToken"])
            if _jwt_unexpired(tok):
                return tok
            # In-memory token expired — fall through to disk/env refresh.
            _STATE["accessToken"] = None
    tok = _load_token_from_disk()
    if tok:
        with _lock:
            if _STATE.get("revoked"):
                return None
            _STATE["accessToken"] = tok
            if not _STATE["connectedAt"]:
                _STATE["connectedAt"] = time.time()
        return tok
    return None


def set_access_token(token: str) -> None:
    token = (token or "").strip()
    if not token:
        raise ValueError("Empty access token")
    with _lock:
        _STATE["accessToken"] = token
        _STATE["connectedAt"] = time.time()
        _STATE["lastError"] = None
        _STATE["revoked"] = False
        _PROBE["ts"] = 0.0
        _PROBE["ok"] = False
        _PROBE["error"] = None
    _save_token(token)
    os.environ["FYERS_ACCESS_TOKEN"] = token


def clear_access_token() -> None:
    with _lock:
        _STATE["accessToken"] = None
        _STATE["connectedAt"] = None
        _STATE["lastError"] = None
        _STATE["revoked"] = True
        _PROBE["ts"] = 0.0
        _PROBE["ok"] = False
        _PROBE["error"] = None
    os.environ.pop("FYERS_ACCESS_TOKEN", None)
    path = _token_path()
    try:
        if path.exists():
            path.unlink()
    except Exception:  # noqa: BLE001
        pass


def _resp_text(resp: Any) -> str:
    try:
        return str(resp).lower()
    except Exception:  # noqa: BLE001
        return ""


def is_fyers_auth_failure(resp_or_msg: Any) -> bool:
    text = _resp_text(resp_or_msg)
    if not text:
        return False
    return any(m in text for m in _AUTH_FAIL_MARKERS)


def mark_token_invalid(reason: str) -> None:
    """Drop stored token so UI goes grey — stale file must not look 'live'."""
    msg = (reason or "Fyers token invalid or expired").strip()
    with _lock:
        _STATE["accessToken"] = None
        _STATE["connectedAt"] = None
        _STATE["lastError"] = msg
        _STATE["revoked"] = True
        _PROBE["ts"] = time.time()
        _PROBE["ok"] = False
        _PROBE["error"] = msg
    os.environ.pop("FYERS_ACCESS_TOKEN", None)
    path = _token_path()
    try:
        if path.exists():
            path.unlink()
    except Exception:  # noqa: BLE001
        pass


def note_fyers_live_ok() -> None:
    with _lock:
        _PROBE["ts"] = time.time()
        _PROBE["ok"] = True
        _PROBE["error"] = None
        _STATE["lastError"] = None


def note_fyers_live_fail(reason: str, *, auth: bool = False) -> None:
    """Record a failed live call. Auth failures clear the token."""
    msg = (reason or "Fyers realtime failed").strip()
    if auth or is_fyers_auth_failure(msg):
        mark_token_invalid(msg)
        return
    with _lock:
        _PROBE["ts"] = time.time()
        _PROBE["ok"] = False
        _PROBE["error"] = msg
        _STATE["lastError"] = msg


def verify_fyers_token(*, force: bool = False) -> bool:
    """Lightweight quote probe. Green only after a recent successful call."""
    import threading

    from .session import is_live_data_window

    tok = get_access_token()
    if not tok:
        with _lock:
            _PROBE["ok"] = False
            if not _PROBE.get("error"):
                _PROBE["error"] = "No valid Fyers token (reconnect — expired .env tokens are ignored)"
        return False

    # Outside market hours: keep token, skip routine probes (no overnight burn).
    # force=True still probes so we can validate a fresh afternoon login after close.
    # Do NOT clear a successful after-hours login probe — that made the UI flip to "paused/disconnected".
    if not is_live_data_window() and not force:
        with _lock:
            if not _STATE.get("revoked"):
                _STATE["lastError"] = None
            return bool(_PROBE.get("ok")) and bool(tok)

    now = time.time()
    with _lock:
        age = now - float(_PROBE.get("ts") or 0)
        if not force and age < _PROBE_TTL_S and _PROBE.get("ts"):
            return bool(_PROBE.get("ok"))

    box: dict[str, Any] = {"resp": None, "exc": None}

    def _call() -> None:
        try:
            client = fyers_client()
            box["resp"] = client.quotes(data={"symbols": "NSE:SBIN-EQ"})
        except Exception as exc:  # noqa: BLE001
            box["exc"] = exc

    th = threading.Thread(target=_call, daemon=True, name="fyers-probe")
    th.start()
    th.join(8.0)
    if th.is_alive():
        note_fyers_live_fail("Fyers probe timed out (no response in 8s)")
        return False

    if box["exc"] is not None:
        exc = box["exc"]
        note_fyers_live_fail(f"{type(exc).__name__}: {exc}", auth=is_fyers_auth_failure(exc))
        return False

    resp = box["resp"]
    if isinstance(resp, dict) and resp.get("s") == "ok":
        rows = resp.get("d") or []
        if isinstance(rows, list) and rows:
            note_fyers_live_ok()
            return True
        # Empty ok payload is suspicious but not always auth — treat as fail, keep token.
        note_fyers_live_fail("Fyers quotes returned empty payload")
        return False

    if is_fyers_auth_failure(resp):
        note_fyers_live_fail(f"Fyers auth failed: {resp}", auth=True)
        return False

    note_fyers_live_fail(f"Fyers probe failed: {resp}")
    return False


def fyers_status(*, verify: bool = True, force_verify: bool = False) -> dict[str, Any]:
    from .session import is_live_data_window, live_data_window_label

    in_hours = is_live_data_window()
    tok = get_access_token()
    live = False
    if tok:
        # Probe during market hours, or whenever force_verify (e.g. right after login).
        if verify and (in_hours or force_verify):
            live = verify_fyers_token(force=force_verify)
            tok = get_access_token()  # may have been cleared on auth fail
        else:
            with _lock:
                live = bool(_PROBE.get("ok")) and bool(tok)
                never_probed = not _PROBE.get("ts")
            # After process restart overnight, probe memory is empty — one force check
            # so a valid saved token still shows Connected (not grey "paused").
            if tok and not live and never_probed and verify and not in_hours:
                live = verify_fyers_token(force=True)
                tok = get_access_token()

    with _lock:
        connected_at = _STATE.get("connectedAt")
        last_error = _STATE.get("lastError") or _PROBE.get("error")
        has_token = bool(_STATE.get("accessToken")) or bool(_load_token_from_disk())

    if not tok:
        has_token = False
        live = False
    elif not in_hours:
        last_error = None  # pause / after-hours is not an auth error

    return {
        "configured": fyers_configured(),
        # Token proven with Fyers (including after-hours login probe).
        "connected": bool(live),
        "hasToken": bool(has_token),
        "marketHours": in_hours,
        "marketHoursLabel": live_data_window_label(),
        # Polling paused — not the same as disconnected.
        "pausedOutsideHours": bool(live or has_token) and not in_hours,
        "tokenMeta": _jwt_meta(tok) if tok else None,
        "connectedAt": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(connected_at))
            if connected_at
            else None
        ),
        "appIdSuffix": (fyers_creds()["app_id"][-6:] if fyers_creds()["app_id"] else None),
        "redirectUri": fyers_creds()["redirect"],
        "lastError": last_error,
        "breadthSourceHint": (
            "fyers" if (live and in_hours) else ("fyers(paused)" if live else "yahoo")
        ),
    }


def auth_login_url() -> str:
    if not fyers_configured():
        raise RuntimeError(
            "Fyers not configured. Set FYERS_APP_ID and FYERS_SECRET_KEY on the API."
        )
    from fyers_apiv3 import fyersModel

    c = fyers_creds()
    session = fyersModel.SessionModel(
        client_id=c["app_id"],
        secret_key=c["secret"],
        redirect_uri=c["redirect"],
        response_type="code",
        state="saint",
        grant_type="authorization_code",
    )
    return session.generate_authcode()


def _parse_auth_code(auth_code_or_url: str) -> str:
    raw = (auth_code_or_url or "").strip()
    if not raw:
        raise ValueError("Paste the auth code (or full redirect URL)")
    if "auth_code=" in raw or raw.startswith("http"):
        qs = parse_qs(urlparse(raw).query)
        code = (qs.get("auth_code") or qs.get("code") or [None])[0]
        if not code:
            raise ValueError("Could not find auth_code in the pasted URL")
        return str(code).strip()
    return raw


def exchange_auth_code(auth_code_or_url: str) -> dict[str, Any]:
    if not fyers_configured():
        raise RuntimeError(
            "Fyers not configured. Set FYERS_APP_ID and FYERS_SECRET_KEY on the API."
        )
    from fyers_apiv3 import fyersModel

    code = _parse_auth_code(auth_code_or_url)
    c = fyers_creds()
    session = fyersModel.SessionModel(
        client_id=c["app_id"],
        secret_key=c["secret"],
        redirect_uri=c["redirect"],
        response_type="code",
        grant_type="authorization_code",
        state="saint",
    )
    session.set_token(code)
    resp = session.generate_token()
    token = None
    if isinstance(resp, dict):
        token = resp.get("access_token")
    if not token:
        with _lock:
            _STATE["lastError"] = f"Token exchange failed: {resp}"
        raise RuntimeError(f"Token exchange failed: {resp}")
    set_access_token(str(token))
    # Force breadth to rebuild on next dashboard pull.
    try:
        from .nifty_breadth import _BREADTH_CACHE

        _BREADTH_CACHE["ts"] = 0.0
        _BREADTH_CACHE["payload"] = None
    except Exception:  # noqa: BLE001
        pass
    # Prove the new token works before returning green (also after hours).
    verify_fyers_token(force=True)
    return fyers_status(verify=False, force_verify=False)


def fyers_client():
    """Authenticated FyersModel or raise."""
    from fyers_apiv3 import fyersModel

    if not fyers_configured():
        raise RuntimeError("Fyers app credentials missing")
    token = get_access_token()
    if not token:
        raise RuntimeError("Fyers not connected — login from the app header")
    c = fyers_creds()
    log_dir = Path(settings.alerts_db).resolve().parent / "fyers_logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir) + "/"
    except Exception:  # noqa: BLE001
        log_path = str(Path("/tmp")) + "/"
    return fyersModel.FyersModel(
        client_id=c["app_id"],
        token=token,
        is_async=False,
        log_path=log_path,
    )
