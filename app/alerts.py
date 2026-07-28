"""High-bar Telegram alerts — ENTRY and EXIT.

ENTRY: news + strong conviction + executable action + optional technicals.
EXIT: after we alerted an entry for the symbol today, when thesis fades
(giveback / trailing stop / reverse / invalidated) → push exit/watch.

Deduped so each symbol gets at most one entry and one exit per session day.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from .config import settings
from .session import now_ist
from .technicals import get_technicals

IST = ZoneInfo("Asia/Kolkata")

_BUY_LONG = {"buy long", "buy"}
_BUY_SHORT = {"buy short", "short"}
_EXIT_THESIS = {"invalidated", "fading"}


def telegram_configured() -> bool:
    return bool(settings.telegram_bot_token.strip() and settings.telegram_chat_id.strip())


def alerts_enabled() -> bool:
    return bool(settings.alerts_enabled) and telegram_configured()


def _db_path() -> Path:
    path = Path(settings.alerts_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(_db_path()))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS sent_alerts (
            dedupe_key TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            sent_at TEXT NOT NULL,
            payload TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS open_entries (
            session_day TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            conviction INTEGER,
            headline TEXT,
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            PRIMARY KEY (session_day, symbol)
        )
        """
    )
    return con


def _already_sent(dedupe_key: str) -> bool:
    with _conn() as con:
        row = con.execute(
            "SELECT 1 FROM sent_alerts WHERE dedupe_key = ?",
            (dedupe_key,),
        ).fetchone()
    return row is not None


def _mark_sent(dedupe_key: str, symbol: str, payload: dict) -> None:
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO sent_alerts (dedupe_key, symbol, sent_at, payload) VALUES (?, ?, ?, ?)",
            (
                dedupe_key,
                symbol,
                datetime.now(IST).isoformat(),
                json.dumps(payload, ensure_ascii=False)[:4000],
            ),
        )
        con.commit()


def _open_entry(session_day: str, symbol: str, *, side: str, conviction: int | None, headline: str | None) -> None:
    with _conn() as con:
        con.execute(
            """
            INSERT INTO open_entries (session_day, symbol, side, conviction, headline, opened_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(session_day, symbol) DO UPDATE SET
                side = excluded.side,
                conviction = excluded.conviction,
                headline = excluded.headline,
                opened_at = excluded.opened_at,
                closed_at = NULL
            """,
            (
                session_day,
                symbol.upper(),
                side,
                conviction,
                (headline or "")[:240],
                datetime.now(IST).isoformat(),
            ),
        )
        con.commit()


def _get_open_entry(session_day: str, symbol: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            """
            SELECT session_day, symbol, side, conviction, headline, opened_at, closed_at
            FROM open_entries
            WHERE session_day = ? AND symbol = ? AND closed_at IS NULL
            """,
            (session_day, symbol.upper()),
        ).fetchone()
    if not row:
        return None
    keys = ["session_day", "symbol", "side", "conviction", "headline", "opened_at", "closed_at"]
    return dict(zip(keys, row))


def _close_entry(session_day: str, symbol: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE open_entries SET closed_at = ? WHERE session_day = ? AND symbol = ? AND closed_at IS NULL",
            (datetime.now(IST).isoformat(), session_day, symbol.upper()),
        )
        con.commit()


def send_telegram(text: str, *, disable_preview: bool = True) -> dict[str, Any]:
    if not telegram_configured():
        return {"ok": False, "error": "not_configured"}
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token.strip()}/sendMessage"
    try:
        r = requests.post(
            url,
            json={
                "chat_id": settings.telegram_chat_id.strip(),
                "text": text,
                "disable_web_page_preview": disable_preview,
            },
            timeout=20,
        )
        data = r.json() if r.content else {}
        if r.status_code >= 400 or not data.get("ok"):
            return {
                "ok": False,
                "error": "telegram_http",
                "status": r.status_code,
                "body": data,
            }
        return {"ok": True, "messageId": (data.get("result") or {}).get("message_id")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "request_failed", "message": str(exc)[:200]}


def _tech_aligns(symbol: str, action: str) -> tuple[bool, str]:
    """Lightweight technical gate — no AI call."""
    try:
        tech = get_technicals(symbol, force=False)
    except Exception as exc:  # noqa: BLE001
        return False, f"tech_error:{exc.__class__.__name__}"
    if not tech or not tech.get("ready"):
        return False, "tech_unavailable"

    trend = str(tech.get("trend") or "")
    rsi_zone = str(tech.get("rsiZone") or "")
    macd = str(tech.get("macdState") or "")
    long = action in _BUY_LONG

    if long:
        if rsi_zone == "overbought":
            return False, "rsi_overbought"
        if trend in {"downtrend", "short_down"}:
            return False, f"trend_{trend}"
        if macd in {"bearish_cross", "bearish_hist"}:
            return False, "macd_bearish"
        if trend in {"uptrend", "short_up"} or rsi_zone in {"bullish_bias", "neutral", "oversold"}:
            return True, f"{trend}/{rsi_zone}/{macd or 'macd_na'}"
        return False, f"weak_long:{trend}/{rsi_zone}"

    if rsi_zone == "oversold":
        return False, "rsi_oversold"
    if trend in {"uptrend", "short_up"}:
        return False, f"trend_{trend}"
    if macd in {"bullish_cross", "bullish_hist"}:
        return False, "macd_bullish"
    if trend in {"downtrend", "short_down"} or rsi_zone in {"bearish_bias", "neutral", "overbought"}:
        return True, f"{trend}/{rsi_zone}/{macd or 'macd_na'}"
    return False, f"weak_short:{trend}/{rsi_zone}"


def _entry_skip_reason(row: dict) -> str | None:
    action = (row.get("action") or "").strip().lower()
    if action not in _BUY_LONG | _BUY_SHORT:
        return "action_not_executable"

    conv = int(row.get("conviction") or 0)
    if conv < int(settings.alerts_min_conviction):
        return "conviction_low"

    thesis = row.get("thesisHealth")
    if thesis in _EXIT_THESIS:
        return f"thesis_{thesis}"

    mins = row.get("latestNewsMins")
    if mins is None:
        mins = row.get("anchorMinutesAgo")
    if mins is not None and int(mins) > int(settings.alerts_max_news_age_mins):
        return "news_stale"

    if row.get("bucket") == "already_reacted" and action in {"already priced", "already fallen"}:
        return "already_reacted"

    return None


def _format_entry(row: dict, *, tech_note: str | None) -> str:
    sym = row.get("symbol") or "?"
    name = row.get("name") or sym
    action = row.get("action") or "watch"
    conv = row.get("conviction")
    conf = row.get("confidence")
    thesis = row.get("thesisLabel") or row.get("thesisHealth") or "n/a"
    headline = row.get("anchorHeadline") or row.get("latestHeadline") or ""
    note = row.get("actionNote") or ""
    side = "BUY" if str(action).lower() in _BUY_LONG else "SHORT"

    lines = [
        f"Saint ENTRY · {side}",
        f"{sym} — {name}",
        f"Conviction {conv}" + (f" · conf {conf}" if conf is not None else ""),
        f"Thesis: {thesis}",
    ]
    if tech_note:
        lines.append(f"Technicals: {tech_note}")
    if note:
        lines.append(f"Note: {note}")
    if headline:
        lines.append(f"News: {headline[:220]}")
    lines.append("Exit alert will fire on giveback / trail / reverse.")
    lines.append("Not advice — check levels before entry.")
    return "\n".join(lines)


def _format_exit(row: dict, *, open_side: str) -> str:
    sym = row.get("symbol") or "?"
    name = row.get("name") or sym
    thesis = row.get("thesisHealth") or "fading"
    label = row.get("thesisLabel") or thesis
    trigger = row.get("thesisExitTrigger") or ""
    peak = row.get("thesisPeakFavPct")
    last = row.get("thesisLastMovePct")
    note = row.get("actionNote") or ""

    why = "giveback/trail" if trigger in {"giveback", "trail"} else str(thesis)
    lines = [
        f"Saint EXIT · {open_side.upper()} → WATCH",
        f"{sym} — {name}",
        f"Reason: {why}",
        f"Detail: {label}",
    ]
    if peak is not None or last is not None:
        lines.append(f"Peak fav {peak}% · now {last}%")
    if note:
        lines.append(f"Note: {note}")
    lines.append("Not advice — manage residual risk.")
    return "\n".join(lines)


def _entry_dedupe_key(row: dict, session_day: str) -> str:
    sym = str(row.get("symbol") or "").upper()
    action = str(row.get("action") or "").lower()
    headline = str(row.get("anchorHeadline") or row.get("latestHeadline") or "")[:120]
    h = hashlib.sha1(headline.encode("utf-8")).hexdigest()[:10]
    return f"entry:{session_day}:{sym}:{action}:{h}"


def _exit_dedupe_key(symbol: str, session_day: str, thesis: str) -> str:
    return f"exit:{session_day}:{symbol.upper()}:{thesis}"


def evaluate_entries(stocks: list[dict], *, session_day: str) -> list[dict]:
    out: list[dict] = []
    for row in stocks or []:
        skip = _entry_skip_reason(row)
        if skip:
            continue
        action = (row.get("action") or "").strip().lower()
        tech_note = None
        if settings.alerts_require_technicals:
            ok, tech_note = _tech_aligns(str(row.get("symbol") or ""), action)
            if not ok:
                continue
        key = _entry_dedupe_key(row, session_day)
        if _already_sent(key):
            continue
        # Already have an open entry today — don't re-enter spam.
        if _get_open_entry(session_day, str(row.get("symbol") or "")):
            continue
        side = "long" if action in _BUY_LONG else "short"
        out.append(
            {
                "kind": "entry",
                "dedupeKey": key,
                "symbol": row.get("symbol"),
                "side": side,
                "action": row.get("action"),
                "conviction": row.get("conviction"),
                "techNote": tech_note,
                "text": _format_entry(row, tech_note=tech_note),
                "row": {
                    "symbol": row.get("symbol"),
                    "action": row.get("action"),
                    "conviction": row.get("conviction"),
                    "anchorHeadline": row.get("anchorHeadline"),
                    "side": side,
                },
            }
        )
    return out


def evaluate_exits(stocks: list[dict], *, session_day: str) -> list[dict]:
    """EXIT only for symbols we previously ENTRY-alerted today."""
    out: list[dict] = []
    by_sym = {str(s.get("symbol") or "").upper(): s for s in (stocks or []) if s.get("symbol")}

    with _conn() as con:
        opens = con.execute(
            """
            SELECT symbol, side, conviction, headline
            FROM open_entries
            WHERE session_day = ? AND closed_at IS NULL
            """,
            (session_day,),
        ).fetchall()

    for symbol, side, _conv, _headline in opens:
        row = by_sym.get(symbol)
        if not row:
            continue
        thesis = row.get("thesisHealth")
        # Exit when path kills the thesis (giveback / trail / reverse / invalid).
        # Also when action was demoted to watch with an exit note.
        action = (row.get("action") or "").strip().lower()
        note = (row.get("actionNote") or "").lower()
        exit_note = "exit" in note or "fading" in note or "giveback" in note or "trail" in note
        if thesis not in _EXIT_THESIS and not (action == "watch" and exit_note):
            continue
        key = _exit_dedupe_key(symbol, session_day, str(thesis or "watch"))
        if _already_sent(key):
            continue
        out.append(
            {
                "kind": "exit",
                "dedupeKey": key,
                "symbol": symbol,
                "side": side,
                "thesisHealth": thesis,
                "text": _format_exit(row, open_side=str(side)),
                "row": {
                    "symbol": symbol,
                    "side": side,
                    "thesisHealth": thesis,
                    "thesisExitTrigger": row.get("thesisExitTrigger"),
                },
            }
        )
    return out


# Back-compat for older imports / tests
def evaluate_board(stocks: list[dict]) -> list[dict]:
    session_day = now_ist().date().isoformat()
    return evaluate_entries(stocks, session_day=session_day)


def run_alert_tick(*, force_dashboard: bool = False) -> dict[str, Any]:
    """Scan board → send new ENTRY and EXIT Telegram alerts."""
    started = time.time()
    if not alerts_enabled():
        return {
            "ok": True,
            "enabled": False,
            "reason": "alerts_disabled_or_telegram_not_configured",
            "sent": [],
            "elapsedMs": int((time.time() - started) * 1000),
        }

    from .service import get_dashboard

    dash = get_dashboard(force=force_dashboard)
    stocks = dash.get("stocks") or []
    session_day = now_ist().date().isoformat()

    entries = evaluate_entries(stocks, session_day=session_day)
    exits = evaluate_exits(stocks, session_day=session_day)
    candidates = entries + exits

    sent: list[dict] = []
    errors: list[dict] = []

    for c in candidates:
        result = send_telegram(c["text"])
        if not result.get("ok"):
            errors.append({"symbol": c.get("symbol"), "kind": c.get("kind"), **result})
            continue

        _mark_sent(c["dedupeKey"], str(c.get("symbol") or ""), c.get("row") or {})
        sym = str(c.get("symbol") or "").upper()
        if c.get("kind") == "entry":
            _open_entry(
                session_day,
                sym,
                side=str(c.get("side") or "long"),
                conviction=int(c.get("conviction") or 0) if c.get("conviction") is not None else None,
                headline=(c.get("row") or {}).get("anchorHeadline"),
            )
        elif c.get("kind") == "exit":
            _close_entry(session_day, sym)

        sent.append(
            {
                "kind": c.get("kind"),
                "symbol": c.get("symbol"),
                "side": c.get("side"),
                "action": c.get("action"),
                "thesisHealth": c.get("thesisHealth"),
                "messageId": result.get("messageId"),
            }
        )

    return {
        "ok": True,
        "enabled": True,
        "boardSize": len(stocks),
        "entryCandidates": len(entries),
        "exitCandidates": len(exits),
        "sent": sent,
        "errors": errors,
        "elapsedMs": int((time.time() - started) * 1000),
        "asOf": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    }


def send_test_alert() -> dict[str, Any]:
    text = (
        "Saint test alert\n"
        f"Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}\n"
        "ENTRY + EXIT wiring ready (enable SAINT_ALERTS_ENABLED + Telegram keys)."
    )
    result = send_telegram(text)
    return {"ok": bool(result.get("ok")), **result}
