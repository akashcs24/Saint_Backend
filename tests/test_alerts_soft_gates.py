"""ENTRY alerts: board rules fire; tech/AI are commentary only."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_entry_fires_even_when_technicals_disagree():
    from app import alerts
    from app.config import settings

    row = {
        "symbol": "HDFCBANK",
        "name": "HDFC Bank",
        "action": "buy short",
        "conviction": 67,
        "thesisHealth": "confirming",
        "latestNewsMins": 30,
        "anchorHeadline": "HDFC Bank weak overnight",
        "bucket": "next_session",
    }

    with (
        patch.object(settings, "alerts_require_technicals", False),
        patch.object(settings, "alerts_ai_comment", False),
        patch.object(alerts, "_tech_aligns", return_value=(False, "trend_uptrend")),
        patch.object(alerts, "_already_sent", return_value=False),
        patch.object(alerts, "_get_open_entry", return_value=None),
    ):
        out = alerts.evaluate_entries([row], session_day="2026-07-28")

    assert len(out) == 1
    assert out[0]["symbol"] == "HDFCBANK"
    assert out[0]["techAligned"] is False
    assert "Technicals (soft)" in out[0]["text"]
    assert "Saint ENTRY" in out[0]["text"]


def test_hard_tech_gate_still_optional():
    from app import alerts
    from app.config import settings

    row = {
        "symbol": "HDFCBANK",
        "action": "buy short",
        "conviction": 67,
        "thesisHealth": "confirming",
        "latestNewsMins": 30,
        "anchorHeadline": "x",
    }

    with (
        patch.object(settings, "alerts_require_technicals", True),
        patch.object(settings, "alerts_ai_comment", False),
        patch.object(alerts, "_tech_aligns", return_value=(False, "trend_uptrend")),
        patch.object(alerts, "_already_sent", return_value=False),
        patch.object(alerts, "_get_open_entry", return_value=None),
    ):
        out = alerts.evaluate_entries([row], session_day="2026-07-28")

    assert out == []


def test_stale_news_still_blocks():
    from app import alerts
    from app.config import settings

    row = {
        "symbol": "HDFCBANK",
        "action": "buy short",
        "conviction": 67,
        "thesisHealth": "confirming",
        "latestNewsMins": 400,
        "anchorHeadline": "old",
    }

    with (
        patch.object(settings, "alerts_require_technicals", False),
        patch.object(settings, "alerts_ai_comment", False),
    ):
        assert alerts._entry_skip_reason(row) == "news_stale"
