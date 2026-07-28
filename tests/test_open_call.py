"""Living overnight open-call: revise until 09:15, freeze after."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IST = ZoneInfo("Asia/Kolkata")


def test_open_call_revises_before_cash_open():
    from app import config, predictions

    with tempfile.TemporaryDirectory() as td:
        config.settings.predictions_db = Path(td) / "pred.sqlite3"
        pre = datetime(2026, 7, 27, 8, 0, tzinfo=IST)  # Monday before open
        first = predictions.upsert_open_call(
            {
                "symbol": "TCS",
                "targetSession": "2026-07-27",
                "expectedDirection": -1,
                "sentiment": "Negative",
                "conviction": 70,
                "confidence": "high",
                "headline": "Rupee opens higher",
                "reason": "IT headwind",
                "newsId": "news-a",
                "baselinePrice": 2200.0,
                "baselineLabel": "close@2026-07-24",
                "bucket": "next_session",
                "sessionPhase": "before_open",
            },
            now=pre,
        )
        assert first["expected_direction"] == -1
        assert first["frozen_at"] is None

        revised = predictions.upsert_open_call(
            {
                "symbol": "TCS",
                "targetSession": "2026-07-27",
                "expectedDirection": 1,
                "sentiment": "Positive",
                "conviction": 55,
                "confidence": "medium",
                "headline": "Oil relief lifts IT",
                "reason": "Risk-on",
                "newsId": "news-b",
                "baselinePrice": 2200.0,
                "baselineLabel": "close@2026-07-24",
                "bucket": "next_session",
                "sessionPhase": "before_open",
            },
            now=datetime(2026, 7, 27, 9, 10, tzinfo=IST),
        )
        assert revised["expected_direction"] == 1
        assert revised["conviction"] == 55
        assert revised["headline"] == "Oil relief lifts IT"
        assert revised["news_id"] == "news-b"
        assert revised["frozen_at"] is None


def test_open_call_freezes_at_and_after_open():
    from app import config, predictions

    with tempfile.TemporaryDirectory() as td:
        config.settings.predictions_db = Path(td) / "pred.sqlite3"
        predictions.upsert_open_call(
            {
                "symbol": "INFY",
                "targetSession": "2026-07-27",
                "expectedDirection": 1,
                "sentiment": "Positive",
                "conviction": 60,
                "confidence": "medium",
                "headline": "Pre-open bullish",
                "reason": "oil",
                "newsId": "n1",
                "baselinePrice": 1000.0,
                "bucket": "next_session",
                "sessionPhase": "before_open",
            },
            now=datetime(2026, 7, 27, 9, 0, tzinfo=IST),
        )
        frozen = predictions.upsert_open_call(
            {
                "symbol": "INFY",
                "targetSession": "2026-07-27",
                "expectedDirection": -1,
                "sentiment": "Negative",
                "conviction": 80,
                "confidence": "high",
                "headline": "Should not stick",
                "reason": "late",
                "newsId": "n2",
                "baselinePrice": 1000.0,
                "bucket": "next_session",
                "sessionPhase": "before_open",
            },
            now=datetime(2026, 7, 27, 9, 15, tzinfo=IST),
        )
        assert frozen["frozen_at"] is not None
        # First touch at open freezes whatever was living; since we revise then
        # freeze in the same post-open call path, the post-open payload is what
        # gets written only when no prior row existed. Here a prior row existed,
        # so fields stay at the pre-open values and only frozen_at is set.
        assert frozen["expected_direction"] == 1
        assert frozen["headline"] == "Pre-open bullish"

        again = predictions.upsert_open_call(
            {
                "symbol": "INFY",
                "targetSession": "2026-07-27",
                "expectedDirection": -1,
                "sentiment": "Negative",
                "conviction": 99,
                "confidence": "high",
                "headline": "Definitely ignore",
                "reason": "x",
                "newsId": "n3",
                "baselinePrice": 1000.0,
                "bucket": "live_session",
                "sessionPhase": "during_market",
            },
            now=datetime(2026, 7, 27, 11, 0, tzinfo=IST),
        )
        assert again["expected_direction"] == 1
        assert again["conviction"] == 60
        assert again["headline"] == "Pre-open bullish"


def test_freeze_due_open_calls_helper():
    from app import config, predictions

    with tempfile.TemporaryDirectory() as td:
        config.settings.predictions_db = Path(td) / "pred.sqlite3"
        predictions.upsert_open_call(
            {
                "symbol": "RELIANCE",
                "targetSession": "2026-07-27",
                "expectedDirection": -1,
                "sentiment": "Negative",
                "conviction": 50,
                "confidence": "medium",
                "headline": "Crude soft",
                "reason": "oil",
                "newsId": "r1",
                "baselinePrice": 1200.0,
                "bucket": "next_session",
                "sessionPhase": "closed_day",
            },
            now=datetime(2026, 7, 26, 20, 0, tzinfo=IST),
        )
        n = predictions.freeze_due_open_calls(now=datetime(2026, 7, 27, 9, 20, tzinfo=IST))
        assert n == 1
        row = predictions.get_open_call("RELIANCE", "2026-07-27")
        assert row is not None and row["frozen_at"] is not None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}  {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}  {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
