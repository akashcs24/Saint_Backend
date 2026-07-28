"""Phase D: overnight volume/VWAP tape gates (closed-session only)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IST = ZoneInfo("Asia/Kolkata")


def test_tape_factor_neutral_on_live():
    from app.tape import tape_conviction_factor, tape_blocks_buy

    tape = {"extendedAboveVwap": True, "volSurge": True, "closeBelowVwap": False}
    assert tape_conviction_factor("Positive", tape, closed_session=False) == 1.0
    assert not tape_blocks_buy("Positive", tape, closed_session=False)


def test_extended_above_vwap_demotes_overnight_long():
    from app.tape import tape_conviction_factor, tape_blocks_buy
    from app.sentiment import conviction_score, bias_and_action

    tape = {"extendedAboveVwap": True, "volSurge": False, "closeBelowVwap": False}
    assert tape_conviction_factor("Positive", tape, closed_session=True) < 0.6
    assert tape_blocks_buy("Positive", tape, closed_session=True)

    base, _, _ = conviction_score(
        evidence_weight=2.0, agreement=0.8, source_count=2,
        direct_share=1.0, volume_ratio=0.0, price_agrees=None,
    )
    faded, _, drivers = conviction_score(
        evidence_weight=2.0, agreement=0.8, source_count=2,
        direct_share=1.0, volume_ratio=0.0, price_agrees=None,
        tape_factor=0.45,
    )
    assert faded < base
    assert any("VWAP" in d or "fade" in d for d in drivers)

    _, action, note = bias_and_action(
        "Positive", 7, change_pct=0.1, move_since_news_pct=0.0,
        breakout_long=True, tape_blocks_buy=True,
    )
    assert action == "watch"
    assert note and "VWAP" in note


def test_vol_and_below_boost_overnight_long():
    from app.tape import tape_conviction_factor, tape_supports_buy
    from app.sentiment import conviction_score

    tape = {"extendedAboveVwap": False, "volSurge": True, "closeBelowVwap": True}
    factor = tape_conviction_factor("Positive", tape, closed_session=True)
    assert factor >= 1.25
    assert tape_supports_buy("Positive", tape, closed_session=True)

    base, _, _ = conviction_score(
        evidence_weight=2.0, agreement=0.8, source_count=2,
        direct_share=1.0, volume_ratio=0.0, price_agrees=None,
    )
    boosted, _, drivers = conviction_score(
        evidence_weight=2.0, agreement=0.8, source_count=2,
        direct_share=1.0, volume_ratio=0.0, price_agrees=None,
        tape_factor=factor,
    )
    assert boosted > base
    assert any("tape" in d or "vol" in d or "VWAP" in d for d in drivers)


def test_shorts_unaffected_by_tape():
    from app.tape import tape_conviction_factor, tape_blocks_buy

    tape = {"extendedAboveVwap": True, "volSurge": True, "closeBelowVwap": False}
    assert tape_conviction_factor("Negative", tape, closed_session=True) == 1.0
    assert not tape_blocks_buy("Negative", tape, closed_session=True)


def test_news_is_closed_session_majority():
    from app.tape import news_is_closed_session

    overnight = [{
        "impact": 7, "relevance": 0.9, "credibility": 0.9,
        "publishedAt": datetime(2026, 7, 24, 18, 0, tzinfo=IST).isoformat(),
    }]
    live = [{
        "impact": 7, "relevance": 0.9, "credibility": 0.9,
        "publishedAt": datetime(2026, 7, 24, 11, 0, tzinfo=IST).isoformat(),
    }]
    assert news_is_closed_session(overnight) is True
    assert news_is_closed_session(live) is False
