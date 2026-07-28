"""Action confirmation layer — demote buy labels when tape fails."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_no_demand_demotes_overnight_long():
    from app.action_confirm import apply_action_confirmation

    tape = {
        "volRatio20": 0.5,
        "volSurge": False,
        "priorDayChangePct": 1.2,
        "closeBelowVwap": False,
        "extendedAboveVwap": False,
    }
    out = apply_action_confirmation(
        "HDFCBANK",
        action="buy long",
        action_note="+0.5% since headline",
        closed_session_news=True,
        tape=tape,
        anchor=None,
    )
    assert out["action"] == "watch"
    assert out["actionConfirm"] == "demoted"
    assert out["actionConfirmOk"] is False
    assert any("no demand" in r for r in out["actionConfirmReasons"])


def test_elevated_prior_tape_keeps_long_preopen():
    from app.action_confirm import apply_action_confirmation

    tape = {
        "volRatio20": 1.6,
        "volSurge": True,
        "priorDayChangePct": -0.4,
        "closeBelowVwap": True,
        "extendedAboveVwap": False,
    }
    with patch("app.action_confirm.is_cash_session_open", return_value=False):
        out = apply_action_confirmation(
            "INFY",
            action="buy long",
            action_note=None,
            closed_session_news=True,
            tape=tape,
            anchor=None,
        )
    assert out["action"] == "buy long"
    assert out["actionConfirm"] == "confirmed"
    assert out["actionConfirmOk"] is True


def test_quiet_prior_awaits_open_preopen():
    from app.action_confirm import apply_action_confirmation

    tape = {
        "volRatio20": 1.0,
        "volSurge": False,
        "priorDayChangePct": 0.0,
        "closeBelowVwap": False,
        "extendedAboveVwap": False,
    }
    with patch("app.action_confirm.is_cash_session_open", return_value=False):
        out = apply_action_confirmation(
            "TCS",
            action="buy long",
            action_note=None,
            closed_session_news=True,
            tape=tape,
            anchor=None,
        )
    assert out["action"] == "watch"
    assert out["actionConfirm"] == "awaiting"


def test_live_quiet_awaits():
    from app.action_confirm import apply_action_confirmation

    with patch(
        "app.action_confirm.live_window_confirm",
        return_value={
            "ready": True,
            "confirm": False,
            "confirmRealtime": False,
            "priceAlignedLong": None,
            "priceAlignedShort": None,
        },
    ):
        out = apply_action_confirmation(
            "RELIANCE",
            action="buy long",
            action_note=None,
            closed_session_news=False,
            tape={},
            anchor={"publishedAt": "2026-07-28T05:00:00+00:00"},
        )
    assert out["action"] == "watch"
    assert out["actionConfirm"] == "awaiting"


def test_live_vol_and_price_confirms():
    from app.action_confirm import apply_action_confirmation

    with patch(
        "app.action_confirm.live_window_confirm",
        return_value={
            "ready": True,
            "confirm": True,
            "confirmRealtime": True,
            "barsAfter": 2,
            "priceAlignedLong": True,
            "priceAlignedShort": False,
        },
    ):
        out = apply_action_confirmation(
            "RELIANCE",
            action="buy long",
            action_note=None,
            closed_session_news=False,
            tape={},
            anchor={"publishedAt": "2026-07-28T05:00:00+00:00"},
        )
    assert out["action"] == "buy long"
    assert out["actionConfirm"] == "confirmed"


def test_live_awaits_next_bar_confirm_delay():
    from app.action_confirm import apply_action_confirmation

    with patch(
        "app.action_confirm.live_window_confirm",
        return_value={
            "ready": True,
            "confirm": True,
            "confirmRealtime": True,
            "barsAfter": 0,
            "priceAlignedLong": True,
            "priceAlignedShort": False,
        },
    ):
        out = apply_action_confirmation(
            "RELIANCE",
            action="buy long",
            action_note=None,
            closed_session_news=False,
            tape={},
            anchor={"publishedAt": "2026-07-28T05:00:00+00:00"},
        )
    assert out["action"] == "watch"
    assert out["actionConfirm"] == "awaiting"



def test_alerts_skip_awaiting_confirm():
    from app.alerts import _entry_skip_reason

    row = {
        "action": "buy long",
        "conviction": 70,
        "thesisHealth": "confirming",
        "latestNewsMins": 30,
        "actionConfirm": "awaiting",
        "actionConfirmOk": False,
    }
    assert _entry_skip_reason(row) == "action_confirm_awaiting"
