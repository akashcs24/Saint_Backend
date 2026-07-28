"""Phase C: support/resistance structure gate for bullish calls."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_breakout_long_boosts_conviction_vs_midrange():
    from app.sentiment import conviction_score

    kwargs = dict(
        evidence_weight=2.0,
        agreement=0.8,
        source_count=2,
        direct_share=1.0,
        volume_ratio=0.0,
        price_agrees=None,
    )
    breakout, _, breakout_drivers = conviction_score(**kwargs, sr_factor=1.25)
    midrange, _, midrange_drivers = conviction_score(**kwargs, sr_factor=0.6)
    assert breakout > midrange
    assert any("breakout" in d for d in breakout_drivers)
    assert any("mid-range" in d for d in midrange_drivers)


def test_sr_factor_reads_position():
    from app.levels import sr_conviction_factor, is_breakout_long

    at_res = {"atResistance": True, "breakout20": False}
    midrange = {"atResistance": False, "breakout20": False}
    assert sr_conviction_factor("Positive", at_res) > 1.0
    assert sr_conviction_factor("Positive", midrange) < 1.0
    # Shorts are not penalised for mid-range position.
    assert sr_conviction_factor("Negative", midrange) == 1.0
    assert is_breakout_long("Positive", at_res)
    assert not is_breakout_long("Positive", midrange)
    # No structure = neutral, no gate.
    assert sr_conviction_factor("Positive", {}) == 1.0


def test_bullish_action_is_buy_long():
    from app.sentiment import bias_and_action

    bias_b, action_b, note_b = bias_and_action(
        "Positive", 6, change_pct=0.2, move_since_news_pct=0.1, breakout_long=True,
        structure={"nearestResistance": 101.0},
    )
    bias_m, action_m, note_m = bias_and_action(
        "Positive", 6, change_pct=0.2, move_since_news_pct=0.1, breakout_long=False,
    )
    assert bias_b == bias_m == "bullish"
    assert action_b == "buy long"
    assert action_m == "buy long"
    assert note_b and "breakout" in note_b
    assert note_m and "mid-range" in note_m


def test_midrange_long_stays_annotated():
    """Mid-range still surfaces as buy long with a mid-range note (tier colours the risk)."""
    from app.sentiment import bias_and_action

    _, action, note = bias_and_action(
        "Positive", 9, change_pct=0.1, move_since_news_pct=0.0, breakout_long=False,
    )
    assert action == "buy long"
    assert note and "mid-range" in note


def test_publish_signal_gates_by_conviction():
    from app.sentiment import publish_signal

    assert publish_signal("bullish", "buy long", 72) == "buy long"
    assert publish_signal("bullish", "buy long", 60) == "buy long"
    # Under 60 is Watch even with a directional lean (gold/grey on the board).
    assert publish_signal("bearish", "buy short", 55) == "watch"
    assert publish_signal("bullish", "buy long", 30) == "watch"
    assert publish_signal("bullish", "buy long", 80, conflict=True) == "watch"
    assert publish_signal("bullish", "already priced", 20) == "already priced"
    # Tape / structure can leave action as watch — do not revive into a buy.
    assert publish_signal("bullish", "watch", 80) == "watch"
