"""Open-window refresh + overnight thesis live health."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IST = ZoneInfo("Asia/Kolkata")


def test_open_window_is_first_30_minutes():
    from app.session import is_open_window, effective_quote_ttl_s

    open_moment = datetime(2026, 7, 24, 9, 20, tzinfo=IST)  # Friday
    mid_day = datetime(2026, 7, 24, 11, 0, tzinfo=IST)
    assert is_open_window(open_moment) is True
    assert is_open_window(mid_day) is False
    assert effective_quote_ttl_s(open_moment) < effective_quote_ttl_s(mid_day)


def test_gap_with_early_is_confirming():
    from app.thesis import classify_health

    h = classify_health(
        expected_direction=1,
        open_move_pct=0.8,
        plus15_move_pct=None,
        market_open=True,
    )
    assert h["thesisHealth"] == "confirming"
    assert h["gapState"] == "with"


def test_gap_against_is_invalidated():
    from app.thesis import classify_health

    h = classify_health(
        expected_direction=1,
        open_move_pct=-1.2,
        market_open=True,
    )
    assert h["thesisHealth"] == "invalidated"


def test_confirming_then_reverse_becomes_fading():
    from app.thesis import classify_health

    early = classify_health(
        expected_direction=1,
        open_move_pct=1.0,
        plus15_move_pct=0.9,
        market_open=True,
    )
    later = classify_health(
        expected_direction=1,
        open_move_pct=1.0,
        plus15_move_pct=0.9,
        plus30_move_pct=-0.8,
        market_open=True,
    )
    assert early["thesisHealth"] == "confirming"
    assert later["thesisHealth"] == "fading"


def test_flat_open_is_cooling():
    from app.thesis import classify_health

    h = classify_health(
        expected_direction=-1,
        open_move_pct=0.1,
        market_open=True,
    )
    assert h["thesisHealth"] == "cooling"


def test_thesis_exit_demotes_buy_when_invalidated():
    from app.thesis import apply_thesis_exit

    action, note = apply_thesis_exit("buy long", "invalidated", action_note="+0.5% · breakout")
    assert action == "watch"
    assert note and "exit" in note

    action_f, note_f = apply_thesis_exit("buy short", "fading")
    assert action_f == "watch"
    assert note_f and "fading" in note_f

    # Confirming leaves the published action alone.
    action_ok, note_ok = apply_thesis_exit("buy long", "confirming", action_note="keep")
    assert action_ok == "buy long"
    assert note_ok == "keep"


def test_peak_giveback_fades_while_still_green():
    """+5% high → +1% now should fade before crossing below 0."""
    from app.thesis import classify_health

    still_green_old_rule = classify_health(
        expected_direction=1,
        open_move_pct=1.0,
        plus15_move_pct=2.0,
        plus30_move_pct=1.5,
        last_move_pct=1.0,
        session_high_pct=5.0,
        session_low_pct=-0.2,
        market_open=True,
    )
    assert still_green_old_rule["thesisHealth"] == "fading"
    assert still_green_old_rule["givebackFrac"] is not None
    assert still_green_old_rule["givebackFrac"] >= 0.5
    assert "Gave back" in still_green_old_rule["label"]


def test_small_run_does_not_peak_fade():
    """Noise under 2% peak should not force fading via giveback."""
    from app.thesis import classify_health

    h = classify_health(
        expected_direction=1,
        open_move_pct=0.8,
        plus30_move_pct=0.6,
        last_move_pct=0.4,
        session_high_pct=1.2,
        market_open=True,
    )
    assert h["thesisHealth"] == "confirming"
    assert (h.get("peakFavPct") or 0) < 2.0


def test_pending_before_open():
    from app.thesis import classify_health

    h = classify_health(
        expected_direction=1,
        open_move_pct=None,
        market_open=False,
    )
    assert h["thesisHealth"] == "pending"


def test_short_peak_giveback_from_low():
    from app.thesis import classify_health

    h = classify_health(
        expected_direction=-1,
        open_move_pct=-1.0,
        plus30_move_pct=-2.0,
        last_move_pct=-1.0,
        session_high_pct=0.5,
        session_low_pct=-5.0,
        market_open=True,
    )
    assert h["thesisHealth"] == "fading"
    assert h["peakFavPct"] >= 5.0


def test_trail_stop_fades_after_arm():
    """Peak +1.5% (<2% giveback gate), now −0.6% → 2.1pp drop → trail fade."""
    from app.thesis import classify_health

    h = classify_health(
        expected_direction=1,
        open_move_pct=0.8,
        plus30_move_pct=1.2,
        last_move_pct=-0.6,
        session_high_pct=1.5,
        market_open=True,
    )
    assert h["thesisHealth"] == "fading"
    assert h.get("exitTrigger") == "trail"


def test_live_news_gets_no_thesis_health():
    from app.thesis import thesis_health_for_stock

    assert (
        thesis_health_for_stock(
            "RELIANCE",
            expected_direction=1,
            baseline_price=100.0,
            session_phase="during_market",
            current_ltp=101.0,
        )
        is None
    )


def test_session_snapshot_exposes_refresh_hint():
    from app.session import session_snapshot

    with patch("app.session.is_open_window", return_value=True):
        snap = session_snapshot(datetime(2026, 7, 24, 9, 20, tzinfo=IST))
    assert snap["openWindow"] is True
    assert snap["refreshHintMs"] == 30_000
    assert snap["quoteTtlS"] <= 30
