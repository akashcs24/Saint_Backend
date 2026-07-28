"""Append-only prediction snapshots + checkpoint verification.

Each row is keyed by ``(news_id, symbol, target_session)`` so later price
updates never rewrite the original call. Outcomes are filled at open, open+30m,
and close for next-session stories, or at close for live-session stories.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from threading import Lock

from .config import settings
from .prices import price_at_checkpoint
from .session import (
    IST,
    checkpoint_times,
    classify_published_at,
    now_ist,
    session_bounds,
    target_session_date,
)

_lock = Lock()

OutcomeStatus = str  # pending | confirmed | wrong | flat


def _db_path() -> Path:
    path = Path(settings.predictions_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            news_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            target_session TEXT NOT NULL,
            bucket TEXT NOT NULL,
            expected_direction INTEGER NOT NULL,
            sentiment TEXT,
            conviction INTEGER,
            confidence TEXT,
            headline TEXT,
            reason TEXT,
            link_type TEXT,
            relevance REAL,
            credibility REAL,
            published_at TEXT,
            session_phase TEXT,
            baseline_price REAL,
            baseline_label TEXT,
            scorer TEXT,
            predicted_at TEXT NOT NULL,
            verify_open_at TEXT,
            verify_open_plus_at TEXT,
            verify_close_at TEXT,
            open_move_pct REAL,
            open_plus_move_pct REAL,
            close_move_pct REAL,
            outcome_status TEXT NOT NULL DEFAULT 'pending',
            outcome_move_pct REAL,
            resolved_at TEXT,
            UNIQUE(news_id, symbol, target_session)
        )
        """
    )
    # One living overnight call per symbol × target session.
    # Revises freely until cash open (09:15), then freezes for verification.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS open_calls (
            symbol TEXT NOT NULL,
            target_session TEXT NOT NULL,
            expected_direction INTEGER NOT NULL,
            sentiment TEXT,
            conviction INTEGER,
            confidence TEXT,
            headline TEXT,
            reason TEXT,
            news_id TEXT,
            link_type TEXT,
            baseline_price REAL,
            baseline_label TEXT,
            bucket TEXT,
            session_phase TEXT,
            scorer TEXT,
            revised_at TEXT NOT NULL,
            frozen_at TEXT,
            verify_open_at TEXT,
            verify_open_plus_at TEXT,
            verify_close_at TEXT,
            open_move_pct REAL,
            open_plus_move_pct REAL,
            close_move_pct REAL,
            outcome_status TEXT NOT NULL DEFAULT 'pending',
            outcome_move_pct REAL,
            resolved_at TEXT,
            PRIMARY KEY (symbol, target_session)
        )
        """
    )
    conn.commit()


def _row_id(news_id: str, symbol: str, target_session: str) -> str:
    return f"{news_id}|{symbol.upper()}|{target_session}"


def upsert_prediction(row: dict) -> dict:
    """Insert a prediction once; never overwrite the original call fields."""
    news_id = row["newsId"]
    symbol = row["symbol"].upper()
    target = row["targetSession"]
    pid = _row_id(news_id, symbol, target)
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            existing = conn.execute("SELECT * FROM predictions WHERE id = ?", (pid,)).fetchone()
            if existing:
                return dict(existing)
            conn.execute(
                """
                INSERT INTO predictions (
                    id, news_id, symbol, target_session, bucket, expected_direction,
                    sentiment, conviction, confidence, headline, reason, link_type,
                    relevance, credibility, published_at, session_phase,
                    baseline_price, baseline_label, scorer, predicted_at,
                    verify_open_at, verify_open_plus_at, verify_close_at,
                    outcome_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    pid,
                    news_id,
                    symbol,
                    target,
                    row.get("bucket") or "next_session",
                    int(row.get("expectedDirection") or 0),
                    row.get("sentiment"),
                    row.get("conviction"),
                    row.get("confidence"),
                    row.get("headline"),
                    row.get("reason"),
                    row.get("linkType"),
                    row.get("relevance"),
                    row.get("credibility"),
                    row.get("publishedAt"),
                    row.get("sessionPhase"),
                    row.get("baselinePrice"),
                    row.get("baselineLabel"),
                    row.get("scorer") or "rules",
                    row.get("predictedAt") or datetime.now(timezone.utc).isoformat(),
                    row.get("verifyOpenAt"),
                    row.get("verifyOpenPlusAt"),
                    row.get("verifyCloseAt"),
                    "pending",
                ),
            )
            conn.commit()
            return dict(conn.execute("SELECT * FROM predictions WHERE id = ?", (pid,)).fetchone())
        finally:
            conn.close()


def _public_open_call(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        "symbol": row.get("symbol"),
        "targetSession": row.get("target_session"),
        "expectedDirection": int(row.get("expected_direction") or 0),
        "sentiment": row.get("sentiment"),
        "conviction": row.get("conviction"),
        "confidence": row.get("confidence"),
        "headline": row.get("headline"),
        "reason": row.get("reason"),
        "newsId": row.get("news_id"),
        "baselinePrice": row.get("baseline_price"),
        "baselineLabel": row.get("baseline_label"),
        "revisedAt": row.get("revised_at"),
        "frozenAt": row.get("frozen_at"),
        "locked": bool(row.get("frozen_at")),
        "outcomeStatus": row.get("outcome_status") or "pending",
        "openMovePct": row.get("open_move_pct"),
        "openPlusMovePct": row.get("open_plus_move_pct"),
        "closeMovePct": row.get("close_move_pct"),
        "outcome": public_outcome(row),
    }


def get_open_call(symbol: str, target_session: str | date) -> dict | None:
    """Frozen or living overnight call for symbol × target session."""
    target = target_session.isoformat() if isinstance(target_session, date) else str(target_session)
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            row = conn.execute(
                "SELECT * FROM open_calls WHERE symbol = ? AND target_session = ?",
                (symbol.upper(), target),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def upsert_open_call(row: dict, *, now: datetime | None = None) -> dict:
    """Living overnight call: revise until 09:15 of target session, then freeze.

    Before cash open the call may flip bias/conviction/anchor as new pre-open
    news arrives. At or after open the snapshot is locked for verification.
    """
    symbol = row["symbol"].upper()
    target = row["targetSession"]
    if isinstance(target, date):
        target = target.isoformat()
    target_date = date.fromisoformat(str(target)[:10])
    open_dt, _ = session_bounds(target_date)
    cps = checkpoint_times(target_date)
    now = now_ist(now)
    revised_at = datetime.now(timezone.utc).isoformat()

    fields = {
        "expected_direction": int(row.get("expectedDirection") or 0),
        "sentiment": row.get("sentiment"),
        "conviction": row.get("conviction"),
        "confidence": row.get("confidence"),
        "headline": row.get("headline"),
        "reason": row.get("reason"),
        "news_id": row.get("newsId"),
        "link_type": row.get("linkType"),
        "baseline_price": row.get("baselinePrice"),
        "baseline_label": row.get("baselineLabel"),
        "bucket": row.get("bucket") or "next_session",
        "session_phase": row.get("sessionPhase"),
        "scorer": row.get("scorer") or "rules",
        "revised_at": revised_at,
        "verify_open_at": cps["open"].isoformat(),
        "verify_open_plus_at": cps["open_plus"].isoformat(),
        "verify_close_at": cps["close"].isoformat(),
    }

    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            existing = conn.execute(
                "SELECT * FROM open_calls WHERE symbol = ? AND target_session = ?",
                (symbol, target),
            ).fetchone()
            existing_d = dict(existing) if existing else None

            # Already frozen — never revise call fields.
            if existing_d and existing_d.get("frozen_at"):
                return existing_d

            # Past open → freeze (keep latest living values, or insert then freeze).
            if now >= open_dt:
                frozen_at = revised_at
                if existing_d:
                    conn.execute(
                        "UPDATE open_calls SET frozen_at = ?, revised_at = ? WHERE symbol = ? AND target_session = ?",
                        (frozen_at, revised_at, symbol, target),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO open_calls (
                            symbol, target_session, expected_direction, sentiment, conviction,
                            confidence, headline, reason, news_id, link_type,
                            baseline_price, baseline_label, bucket, session_phase, scorer,
                            revised_at, frozen_at, verify_open_at, verify_open_plus_at, verify_close_at,
                            outcome_status
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            symbol,
                            target,
                            fields["expected_direction"],
                            fields["sentiment"],
                            fields["conviction"],
                            fields["confidence"],
                            fields["headline"],
                            fields["reason"],
                            fields["news_id"],
                            fields["link_type"],
                            fields["baseline_price"],
                            fields["baseline_label"],
                            fields["bucket"],
                            fields["session_phase"],
                            fields["scorer"],
                            revised_at,
                            frozen_at,
                            fields["verify_open_at"],
                            fields["verify_open_plus_at"],
                            fields["verify_close_at"],
                            "pending",
                        ),
                    )
                conn.commit()
                return dict(
                    conn.execute(
                        "SELECT * FROM open_calls WHERE symbol = ? AND target_session = ?",
                        (symbol, target),
                    ).fetchone()
                )

            # Before open — insert or revise freely.
            if existing_d:
                conn.execute(
                    """
                    UPDATE open_calls SET
                        expected_direction = ?, sentiment = ?, conviction = ?, confidence = ?,
                        headline = ?, reason = ?, news_id = ?, link_type = ?,
                        baseline_price = ?, baseline_label = ?, bucket = ?, session_phase = ?,
                        scorer = ?, revised_at = ?,
                        verify_open_at = ?, verify_open_plus_at = ?, verify_close_at = ?
                    WHERE symbol = ? AND target_session = ?
                    """,
                    (
                        fields["expected_direction"],
                        fields["sentiment"],
                        fields["conviction"],
                        fields["confidence"],
                        fields["headline"],
                        fields["reason"],
                        fields["news_id"],
                        fields["link_type"],
                        fields["baseline_price"],
                        fields["baseline_label"],
                        fields["bucket"],
                        fields["session_phase"],
                        fields["scorer"],
                        fields["revised_at"],
                        fields["verify_open_at"],
                        fields["verify_open_plus_at"],
                        fields["verify_close_at"],
                        symbol,
                        target,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO open_calls (
                        symbol, target_session, expected_direction, sentiment, conviction,
                        confidence, headline, reason, news_id, link_type,
                        baseline_price, baseline_label, bucket, session_phase, scorer,
                        revised_at, frozen_at, verify_open_at, verify_open_plus_at, verify_close_at,
                        outcome_status
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,?,?,?)
                    """,
                    (
                        symbol,
                        target,
                        fields["expected_direction"],
                        fields["sentiment"],
                        fields["conviction"],
                        fields["confidence"],
                        fields["headline"],
                        fields["reason"],
                        fields["news_id"],
                        fields["link_type"],
                        fields["baseline_price"],
                        fields["baseline_label"],
                        fields["bucket"],
                        fields["session_phase"],
                        fields["scorer"],
                        fields["revised_at"],
                        fields["verify_open_at"],
                        fields["verify_open_plus_at"],
                        fields["verify_close_at"],
                        "pending",
                    ),
                )
            conn.commit()
            return dict(
                conn.execute(
                    "SELECT * FROM open_calls WHERE symbol = ? AND target_session = ?",
                    (symbol, target),
                ).fetchone()
            )
        finally:
            conn.close()


def freeze_due_open_calls(*, now: datetime | None = None) -> int:
    """Lock any still-living open calls whose target session has opened."""
    now = now_ist(now)
    frozen = 0
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            rows = conn.execute("SELECT * FROM open_calls WHERE frozen_at IS NULL").fetchall()
            for row in rows:
                try:
                    target_date = date.fromisoformat(str(row["target_session"])[:10])
                except ValueError:
                    continue
                open_dt, _ = session_bounds(target_date)
                if now < open_dt:
                    continue
                ts = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "UPDATE open_calls SET frozen_at = ?, revised_at = ? WHERE symbol = ? AND target_session = ?",
                    (ts, ts, row["symbol"], row["target_session"]),
                )
                frozen += 1
            conn.commit()
        finally:
            conn.close()
    return frozen


def _judge(expected: int, move_pct: float | None, threshold: float) -> OutcomeStatus:
    if move_pct is None:
        return "pending"
    if abs(move_pct) < threshold:
        return "flat"
    if expected == 0:
        return "flat"
    return "confirmed" if (move_pct > 0) == (expected > 0) else "wrong"


def resolve_due_predictions(*, now: datetime | None = None, threshold: float | None = None) -> int:
    """Fill open / +30m / close checkpoints that are due. Returns rows updated."""
    from .config import settings as cfg

    threshold = threshold if threshold is not None else float(cfg.reaction_threshold_pct)
    now = now_ist(now)
    # Lock living overnight calls the moment their target session opens.
    freeze_due_open_calls(now=now)
    updated = 0
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            rows = conn.execute(
                "SELECT * FROM predictions WHERE outcome_status = 'pending'"
            ).fetchall()
            for row in rows:
                changes = _resolve_one(dict(row), now, threshold)
                if not changes:
                    continue
                sets = ", ".join(f"{k} = ?" for k in changes)
                conn.execute(
                    f"UPDATE predictions SET {sets} WHERE id = ?",
                    [*changes.values(), row["id"]],
                )
                updated += 1

            # Resolve frozen open_calls the same way (open / +30 / close).
            open_rows = conn.execute(
                "SELECT * FROM open_calls WHERE outcome_status = 'pending' AND frozen_at IS NOT NULL"
            ).fetchall()
            for row in open_rows:
                changes = _resolve_one(dict(row), now, threshold)
                if not changes:
                    continue
                sets = ", ".join(f"{k} = ?" for k in changes)
                conn.execute(
                    f"UPDATE open_calls SET {sets} WHERE symbol = ? AND target_session = ?",
                    [*changes.values(), row["symbol"], row["target_session"]],
                )
                updated += 1
            conn.commit()
        finally:
            conn.close()
    return updated


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        return dt.astimezone(IST)
    except ValueError:
        return None


def _resolve_one(row: dict, now: datetime, threshold: float) -> dict:
    expected = int(row["expected_direction"] or 0)
    symbol = row["symbol"]
    changes: dict = {}

    for field, col in (
        ("verify_open_at", "open_move_pct"),
        ("verify_open_plus_at", "open_plus_move_pct"),
        ("verify_close_at", "close_move_pct"),
    ):
        if row.get(col) is not None:
            continue
        when = _parse_iso(row.get(field))
        if when is None or now < when:
            continue
        _label, price = price_at_checkpoint(symbol, when)
        baseline = row.get("baseline_price")
        if price is None or not baseline:
            continue
        move = round((price - float(baseline)) / float(baseline) * 100.0, 2)
        changes[col] = move

    # Prefer the most complete checkpoint for the final outcome.
    open_m = changes.get("open_move_pct", row.get("open_move_pct"))
    plus_m = changes.get("open_plus_move_pct", row.get("open_plus_move_pct"))
    close_m = changes.get("close_move_pct", row.get("close_move_pct"))

    final_move = close_m if close_m is not None else plus_m if plus_m is not None else open_m
    close_due = _parse_iso(row.get("verify_close_at"))
    # Resolve fully only once the close checkpoint has passed (or for live
    # stories that only schedule a close verify).
    if final_move is not None and close_due is not None and now >= close_due:
        status = _judge(expected, float(final_move), threshold)
        changes["outcome_status"] = status
        changes["outcome_move_pct"] = float(final_move)
        changes["resolved_at"] = datetime.now(timezone.utc).isoformat()
    elif final_move is not None and close_due is None:
        status = _judge(expected, float(final_move), threshold)
        if status != "pending":
            changes["outcome_status"] = status
            changes["outcome_move_pct"] = float(final_move)
            changes["resolved_at"] = datetime.now(timezone.utc).isoformat()

    return changes


def get_prediction(news_id: str, symbol: str, target_session: str) -> dict | None:
    pid = _row_id(news_id, symbol, target_session)
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pid,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def accuracy_summary(*, min_resolved: int = 10) -> dict:
    """Accuracy stats with selectivity slices for the dashboard scorecard.

    Slices (open / close / high-conv / direct / sector) are how we steer toward
    ~80% — the headline rate alone is too noisy.
    """
    from .config import settings as cfg

    threshold = float(cfg.reaction_threshold_pct)
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            rows = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT expected_direction, conviction, link_type,
                           open_move_pct, close_move_pct, outcome_status, outcome_move_pct
                    FROM predictions
                    WHERE expected_direction != 0
                    """
                ).fetchall()
            ]
        finally:
            conn.close()

    def _slice_stats(
        pool: list[dict],
        *,
        status_key: str = "outcome_status",
        move_key: str | None = None,
    ) -> dict:
        confirmed = wrong = flat = 0
        for r in pool:
            if move_key:
                st = _judge(int(r["expected_direction"]), r.get(move_key), threshold)
            else:
                st = r.get(status_key) or "pending"
            if st == "confirmed":
                confirmed += 1
            elif st == "wrong":
                wrong += 1
            elif st == "flat":
                flat += 1
        decided = confirmed + wrong
        resolved = decided + flat
        hit = round(confirmed / decided, 3) if decided else None
        return {
            "resolved": resolved,
            "confirmed": confirmed,
            "wrong": wrong,
            "flat": flat,
            "decided": decided,
            "hitRate": hit,
            "ready": resolved >= max(5, min_resolved // 2) and decided >= 5,
        }

    # Final close-style outcome (stored).
    closed = [r for r in rows if (r.get("outcome_status") or "pending") != "pending"]
    overall = _slice_stats(closed)

    # Open checkpoint — only rows that have an open print.
    open_pool = [r for r in rows if r.get("open_move_pct") is not None]
    at_open = _slice_stats(open_pool, move_key="open_move_pct")

    # Close checkpoint — only rows that have a close print.
    close_pool = [r for r in rows if r.get("close_move_pct") is not None]
    at_close = _slice_stats(close_pool, move_key="close_move_pct")

    high_conv = _slice_stats(
        [r for r in closed if int(r.get("conviction") or 0) >= 60]
    )
    mid_conv = _slice_stats(
        [r for r in closed if 40 <= int(r.get("conviction") or 0) < 60]
    )
    direct = _slice_stats(
        [r for r in closed if (r.get("link_type") or "direct") == "direct"]
    )
    sector = _slice_stats(
        [r for r in closed if (r.get("link_type") or "") in {"peer", "sector"}]
    )

    return {
        "resolved": overall["resolved"],
        "confirmed": overall["confirmed"],
        "wrong": overall["wrong"],
        "flat": overall["flat"],
        "hitRate": overall["hitRate"],
        "ready": overall["resolved"] >= min_resolved
        and overall["decided"] >= max(5, min_resolved // 2),
        "slices": {
            "overall": overall,
            "atOpen": at_open,
            "atClose": at_close,
            "highConviction": high_conv,
            "midConviction": mid_conv,
            "direct": direct,
            "sector": sector,
        },
    }


def schedule_for_news(
    *,
    news_item: dict,
    symbol: str,
    expected_direction: int,
    bucket: str,
    baseline_price: float | None,
    baseline_label: str | None,
    sentiment: str,
    conviction: int,
    confidence: str,
    reason: str,
    scorer: str = "rules",
) -> dict:
    """Build + persist a prediction row for one stock×story."""
    published = news_item.get("publishedAt")
    phase = classify_published_at(published)
    target = target_session_date(published)
    cps = checkpoint_times(target)

    # Live-session stories verify through the same day's close; overnight
    # stories use all three checkpoints on the next session.
    if bucket == "live_session":
        verify = {
            "verifyOpenAt": None,
            "verifyOpenPlusAt": None,
            "verifyCloseAt": cps["close"].isoformat(),
        }
    else:
        verify = {
            "verifyOpenAt": cps["open"].isoformat(),
            "verifyOpenPlusAt": cps["open_plus"].isoformat(),
            "verifyCloseAt": cps["close"].isoformat(),
        }

    return upsert_prediction(
        {
            "newsId": news_item["id"],
            "symbol": symbol,
            "targetSession": target.isoformat(),
            "bucket": bucket,
            "expectedDirection": expected_direction,
            "sentiment": sentiment,
            "conviction": conviction,
            "confidence": confidence,
            "headline": news_item.get("headline"),
            "reason": reason,
            "linkType": news_item.get("linkType"),
            "relevance": news_item.get("relevance"),
            "credibility": news_item.get("credibility"),
            "publishedAt": published,
            "sessionPhase": phase,
            "baselinePrice": baseline_price,
            "baselineLabel": baseline_label,
            "scorer": scorer,
            "predictedAt": datetime.now(timezone.utc).isoformat(),
            **verify,
        }
    )


def public_outcome(row: dict | None) -> dict | None:
    if not row:
        return None
    status = row.get("outcome_status") or row.get("outcomeStatus") or "pending"
    label = {
        "pending": "Pending",
        "confirmed": "Confirmed",
        "wrong": "Wrong",
        "flat": "No meaningful move",
    }.get(status, "Pending")
    move = row.get("outcome_move_pct")
    if move is None:
        move = row.get("close_move_pct") or row.get("open_plus_move_pct") or row.get("open_move_pct")
    return {
        "status": status,
        "label": label,
        "movePct": move,
        "openMovePct": row.get("open_move_pct"),
        "openPlusMovePct": row.get("open_plus_move_pct"),
        "closeMovePct": row.get("close_move_pct"),
        "scorer": row.get("scorer"),
    }
