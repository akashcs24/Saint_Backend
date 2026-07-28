"""Regression tests for the headline → stock intelligence layer.

Run with: .venv/bin/python -m pytest tests -q
(or .venv/bin/python tests/test_linking.py for a dependency-free run)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.linking import DIRECT_MIN, analyze  # noqa: E402


def _links(analysis, symbol):
    return next((l for l in analysis.links if l.symbol == symbol), None)


def test_substring_of_another_word_is_not_a_match():
    """The original bug: "itc" living inside "bitcoin"."""
    a = analyze("Bitcoin trades near $65,000 as Middle East tensions dampen crypto sentiment")
    assert "ITC" not in a.direct
    assert _links(a, "ITC") is None


def test_crypto_has_no_equity_read_through():
    a = analyze("Ethereum and Solana slump as crypto outflows accelerate")
    assert a.scope == "offshore"
    assert a.equity_nexus == 0.0
    assert a.direct == []


def test_offshore_story_damps_its_passing_mentions():
    """Crypto is the subject; the geopolitics aside must stay marginal."""
    a = analyze("Bitcoin slides as Middle East tensions escalate")
    nifty = _links(a, "NIFTY")
    assert nifty is None or nifty.relevance < 0.25


def test_company_named_is_direct_news():
    a = analyze("ITC Q1 net profit rises 8% on cigarette volume growth")
    assert a.direct == ["ITC"]
    assert _links(a, "ITC").relevance >= DIRECT_MIN
    assert "earnings_beat" in a.events


def test_sebi_penalty_is_never_neutral():
    """Research insight: lexicon-only scoring misses regulatory language."""
    from app.sentiment import polarity

    sent, score = polarity(
        "SEBI imposes Rs 5 crore penalty on listed company for disclosure lapses",
        title="SEBI imposes Rs 5 crore penalty on listed company for disclosure lapses",
    )
    assert sent == "Negative"
    assert score < -0.15


def test_npa_lexicon_is_negative():
    from app.sentiment import polarity

    sent, _ = polarity("Bank's GNPA rises as slippage widens in the quarter")
    assert sent == "Negative"


def test_state_bank_alias_resolves_to_sbin():
    a = analyze("State Bank shares rally after strong credit growth")
    assert "SBIN" in a.direct


def test_hdfc_life_does_not_become_hdfcbank():
    a = analyze("HDFC Life posts higher premium income for the quarter")
    assert "HDFCBANK" not in a.direct
    assert "HDFCLIFE" in a.direct


def test_passing_mention_is_not_a_regulatory_event():
    """"cigarette volumes grew" is an earnings line, not a tax change."""
    a = analyze("ITC Q1 net profit rises 8% as cigarette volumes grow")
    assert "tobacco" not in [t.key for t in a.themes]


def test_regulatory_theme_reaches_the_exposed_name_not_the_whole_sector():
    a = analyze("GST Council may hike cess on cigarettes and tobacco products")
    itc = _links(a, "ITC")
    assert itc is not None
    assert itc.link_type == "sector"  # context, never company news
    assert itc.direction == -1
    # ITC's cigarette exposure must outrank a generic FMCG peer.
    assert itc.relevance > _links(a, "BRITANNIA").relevance


def test_sector_link_is_never_company_news():
    a = analyze("Crude oil surges past $90 after OPEC supply cut")
    ongc = _links(a, "ONGC")
    assert ongc.link_type == "sector"
    assert "ONGC" not in a.direct


def test_signed_sensitivity_splits_winners_from_losers():
    a = analyze("Crude oil surges past $90 after OPEC supply cut")
    assert _links(a, "ONGC").direction == 1  # upstream gains
    assert _links(a, "ASIANPAINT").direction == -1  # input cost pain


def test_inverse_phrasing_resolves_to_the_same_driver():
    weak = analyze("Rupee weakens to record low against the US dollar")
    assert [t.direction for t in weak.themes if t.key == "rupee"] == [-1]
    # A weaker rupee is a tailwind for IT exporters.
    assert _links(weak, "INFY").direction == 1


def test_direction_is_scoped_to_its_own_clause():
    a = analyze("Stock markets tumble in early trade as crude oil prices hit $100 per barrel")
    assert [t.direction for t in a.themes if t.key == "crude"] == [1]


def test_more_specific_name_wins():
    a = analyze("ITC Hotels reports strong quarterly results on higher room rates")
    assert a.direct == ["ITCHOTELS"]
    assert "ITC" not in a.direct


def test_deny_context_protects_the_parent():
    a = analyze("SBI Life posts higher premium income for the quarter")
    assert "SBIN" not in a.direct


def test_competitor_read_across_is_inverted():
    a = analyze("Reliance Jio adds 3 million subscribers in June")
    airtel = _links(a, "BHARTIARTL")
    assert airtel is not None and airtel.link_type == "peer" and airtel.direction == -1


def test_theme_only_in_the_body_cannot_drive_a_sector_link():
    headline = "Rupee hits day's low against US dollar, recovers on RBI support"
    body = "Crude oil prices climbed as traders weighed supply risk."
    a = analyze(headline, body)
    itc = _links(a, "ITC")  # FMCG, only reachable here via crude
    assert itc is None or itc.relevance < 0.2


def test_weak_alias_needs_corroboration():
    bare = analyze("HUL")
    assert _links(bare, "HINDUNILVR").relevance < DIRECT_MIN
    confirmed = analyze("HUL shares rise 3% on NSE after margin beat")
    assert _links(confirmed, "HINDUNILVR").relevance >= DIRECT_MIN


def test_idfc_first_bank_q1_links_directly():
    a = analyze(
        "IDFC First Bank shares in focus after Q1 profit jumps 132% to Rs 1,075 crore"
    )
    assert "IDFCFIRSTB" in a.direct
    assert _links(a, "IDFCFIRSTB").relevance >= DIRECT_MIN


def test_idfc_bank_short_form_links():
    a = analyze("Stocks in news: IDFC Bank, Maruti Suzuki, ONGC, BoB, Adani Energy")
    assert "IDFCFIRSTB" in a.direct
    assert "MARUTI" in a.direct


def test_indigo_links_on_airline_headline():
    a = analyze(
        "IndiGo shares jump as crude oil falls after US pauses strikes on Iran"
    )
    assert "INDIGO" in a.direct
    assert _links(a, "INDIGO").relevance >= DIRECT_MIN


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
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
