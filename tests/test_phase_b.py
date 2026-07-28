"""Phase B: session timing + event-type conviction calibration."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IST = ZoneInfo("Asia/Kolkata")


def test_merger_classified_as_stress_event():
    from app.events import classify_event, is_hype_event, is_stress_event

    key, bias = classify_event("Adani Ports to acquire controlling stake in logistics firm")
    assert key == "merger_acquisition"
    assert bias > 0
    assert is_stress_event(key)
    assert not is_hype_event(key)


def test_order_win_is_hype_and_demoted():
    from app.events import classify_event, event_evidence_multiplier, is_hype_event

    key, _ = classify_event("L&T bags Rs 5,000 crore order from Middle East client")
    assert key == "order_win"
    assert is_hype_event(key)
    assert event_evidence_multiplier(key) < 1.0


def test_earnings_miss_is_boosted():
    from app.events import classify_event, event_evidence_multiplier, is_stress_event

    key, bias = classify_event("Infosys profit falls 12% as revenue misses estimates")
    assert key == "earnings_miss"
    assert bias < 0
    assert is_stress_event(key)
    assert event_evidence_multiplier(key) > 1.0


def test_conviction_session_and_event_adjustments():
    from app.sentiment import conviction_score

    base, _, _ = conviction_score(
        evidence_weight=2.0,
        agreement=0.8,
        source_count=2,
        direct_share=1.0,
        volume_ratio=0.0,
        price_agrees=None,
    )
    overnight, _, overnight_drivers = conviction_score(
        evidence_weight=2.0,
        agreement=0.8,
        source_count=2,
        direct_share=1.0,
        volume_ratio=0.0,
        price_agrees=None,
        session_factor=1.25,
        event_factor=1.4,
    )
    live_hype, _, live_drivers = conviction_score(
        evidence_weight=2.0,
        agreement=0.8,
        source_count=2,
        direct_share=1.0,
        volume_ratio=0.0,
        price_agrees=None,
        session_factor=0.7,
        event_factor=0.5,
    )
    assert overnight > base > live_hype
    assert any("overnight" in d for d in overnight_drivers)
    assert any("stress" in d or "M&A" in d for d in overnight_drivers)
    assert any("live-session" in d for d in live_drivers)
    assert any("hype" in d for d in live_drivers)


def test_evidence_weight_prefers_overnight_stress_over_live_hype():
    from app.service import _evidence_weight

    stress = {
        "impact": 7,
        "relevance": 0.9,
        "credibility": 0.9,
        "minutesAgo": 30,
        "event": "earnings_miss",
        "publishedAt": datetime(2026, 7, 24, 18, 30, tzinfo=IST).isoformat(),
    }
    hype = {
        "impact": 7,
        "relevance": 0.9,
        "credibility": 0.9,
        "minutesAgo": 30,
        "event": "order_win",
        "publishedAt": datetime(2026, 7, 24, 11, 30, tzinfo=IST).isoformat(),
    }
    assert _evidence_weight(stress) > _evidence_weight(hype)


def test_aggregate_demotes_hype_relative_to_miss():
    from app.service import aggregate_stock_sentiment

    common = {
        "impact": 7,
        "relevance": 0.95,
        "credibility": 0.9,
        "minutesAgo": 45,
        "source": "Moneycontrol",
        "linkType": "direct",
        "summary": "",
        "publishedAt": datetime(2026, 7, 24, 18, 0, tzinfo=IST).isoformat(),
    }
    miss = {
        **common,
        "headline": "Wipro profit falls as revenue misses estimates",
        "sentiment": "Negative",
        "event": "earnings_miss",
        "expectedDirection": -1,
    }
    beat = {
        **common,
        "headline": "Wipro profit surges as revenue beats estimates",
        "sentiment": "Positive",
        "event": "earnings_beat",
        "expectedDirection": 1,
        "publishedAt": datetime(2026, 7, 24, 11, 0, tzinfo=IST).isoformat(),
    }
    miss_read = aggregate_stock_sentiment("WIPRO", [miss], change_pct=0.0, ltp=100.0)
    beat_read = aggregate_stock_sentiment("WIPRO", [beat], change_pct=0.0, ltp=100.0)
    assert miss_read["conviction"] > beat_read["conviction"]
    assert miss_read["expectedDirection"] == -1
    assert beat_read["expectedDirection"] == 1
