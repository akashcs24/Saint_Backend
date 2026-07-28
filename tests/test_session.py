"""Session clock + prediction checkpoint regressions."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IST = ZoneInfo("Asia/Kolkata")


def test_friday_after_close_targets_monday():
    from app.session import classify_published_at, next_session_open, target_session_date

    fri = datetime(2026, 7, 24, 18, 0, tzinfo=IST)
    assert classify_published_at(fri) == "after_close"
    assert target_session_date(fri).isoformat() == "2026-07-27"
    nxt = next_session_open(fri)
    assert nxt.date().isoformat() == "2026-07-27"
    assert nxt.hour == 9 and nxt.minute == 15


def test_during_market_is_live_phase():
    from app.session import classify_published_at, target_session_date

    mid = datetime(2026, 7, 24, 11, 30, tzinfo=IST)
    assert classify_published_at(mid) == "during_market"
    assert target_session_date(mid).isoformat() == "2026-07-24"


def test_before_open_same_day():
    from app.session import classify_published_at, target_session_date

    pre = datetime(2026, 7, 24, 8, 0, tzinfo=IST)
    assert classify_published_at(pre) == "before_open"
    assert target_session_date(pre).isoformat() == "2026-07-24"


def test_weekend_is_closed_day():
    from app.session import classify_published_at, is_trading_day

    sat = datetime(2026, 7, 25, 12, 0, tzinfo=IST)
    assert classify_published_at(sat) == "closed_day"
    assert is_trading_day(sat.date()) is False


def test_republic_day_is_holiday():
    from app.session import is_trading_day
    from datetime import date

    assert is_trading_day(date(2026, 1, 26)) is False


def test_bucket_rules():
    from datetime import date

    from app.service import _assign_bucket

    assert (
        _assign_bucket(
            session_phase="after_close",
            observed_move_pct=0.2,
            expected_direction=-1,
            market_open=False,
        )
        == "next_session"
    )
    assert (
        _assign_bucket(
            session_phase="during_market",
            observed_move_pct=0.2,
            expected_direction=1,
            market_open=True,
        )
        == "live_session"
    )
    assert (
        _assign_bucket(
            session_phase="during_market",
            observed_move_pct=1.8,
            expected_direction=1,
            market_open=True,
        )
        == "already_reacted"
    )
    # Pre-open story for today moves to live once cash session opens.
    today = date(2026, 7, 27)
    assert (
        _assign_bucket(
            session_phase="before_open",
            observed_move_pct=0.2,
            expected_direction=1,
            market_open=True,
            target_session=today,
            today=today,
        )
        == "live_session"
    )
    assert (
        _assign_bucket(
            session_phase="before_open",
            observed_move_pct=0.2,
            expected_direction=1,
            market_open=False,
            target_session=today,
            today=today,
        )
        == "next_session"
    )


def test_prediction_persist_and_judge(tmp_path=None):
    from app import config, predictions

    with tempfile.TemporaryDirectory() as td:
        config.settings.predictions_db = Path(td) / "pred.sqlite3"
        # Clear module-level path cache by writing through settings.
        news = {
            "id": "abc123",
            "headline": "ITC profit rises",
            "publishedAt": datetime(2026, 7, 24, 18, 0, tzinfo=IST).isoformat(),
            "linkType": "direct",
            "relevance": 0.9,
            "credibility": 0.9,
        }
        row = predictions.schedule_for_news(
            news_item=news,
            symbol="ITC",
            expected_direction=1,
            bucket="next_session",
            baseline_price=280.0,
            baseline_label="close@2026-07-24",
            sentiment="Positive",
            conviction=50,
            confidence="medium",
            reason="ITC named in the story",
            scorer="rules",
        )
        assert row["outcome_status"] == "pending"
        assert predictions._judge(1, 1.5, 1.0) == "confirmed"
        assert predictions._judge(1, -1.5, 1.0) == "wrong"
        assert predictions._judge(1, 0.2, 1.0) == "flat"
        again = predictions.schedule_for_news(
            news_item=news,
            symbol="ITC",
            expected_direction=1,
            bucket="next_session",
            baseline_price=999.0,  # must not overwrite
            baseline_label="changed",
            sentiment="Positive",
            conviction=50,
            confidence="medium",
            reason="ITC named in the story",
        )
        assert again["baseline_price"] == 280.0


def test_bitcoin_still_not_on_itc():
    from app.linking import analyze

    a = analyze("Bitcoin trades near $65,000 as Middle East tensions dampen crypto sentiment")
    assert "ITC" not in a.direct
    assert next((l for l in a.links if l.symbol == "ITC"), None) is None


def test_dashboard_and_stock_page_share_conviction():
    """Board filter must not change conviction vs the stock detail page."""
    from app.service import build_dashboard, build_stock_detail

    dash = build_dashboard()
    sample = None
    for bucket in (dash.get("buckets") or {}).values():
        if bucket:
            sample = bucket[0]
            break
    assert sample is not None
    detail = build_stock_detail(sample["symbol"])
    assert detail is not None
    stock = detail["stock"]
    assert stock["confidence"] == sample["confidence"]
    assert stock["conviction"] == sample["conviction"]


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
