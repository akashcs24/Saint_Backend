"""Macro radar from live Yahoo commodities/FX + related headlines."""

from __future__ import annotations

from .news import fetch_news
from .sentiment import impact_score
from .taxonomy import THEMES_BY_KEY

MACRO_INSTRUMENTS = [
    ("CRUDE", "CL=F", "Brent/WTI crude"),
    ("USDINR", "INR=X", "USD / INR"),
    ("GOLD", "GC=F", "Gold"),
]


def _quote_card(key: str, yahoo: str, label: str) -> dict | None:
    # get_quote expects our symbol mapper; pass yahoo directly via temporary path
    from .quotes import _from_yahoo, _from_parquet

    q = _from_yahoo(yahoo)
    if not q:
        return None
    sent = "Positive" if q.change_pct > 0.15 else "Negative" if q.change_pct < -0.15 else "Neutral"
    # For crude/USDINR, "Positive" for India equities may be inverted — keep price polarity literal
    detail = f"{label} last {q.ltp:,.2f} ({q.change_pct:+.2f}%). Source: Yahoo delayed."
    impact = impact_score(
        text=detail,
        source_host="yahoo.com",
        minutes_ago=30,
        ticker_count=1,
    )
    return {
        "id": f"macro-{key.lower()}",
        "title": f"{label}: {q.ltp:,.2f} ({q.change_pct:+.2f}%)",
        "detail": detail,
        "scope": "commodity" if key in {"CRUDE", "GOLD"} else "currency" if key == "USDINR" else "macro",
        "theme": label,
        "sentiment": sent,
        "impact": impact,
        "instruments": [key, "NIFTY"],
        "minutesAgo": 30,
        "source": "Yahoo Finance",
        "ltp": q.ltp,
        "changePct": q.change_pct,
    }


def build_macro_cards(limit: int = 3) -> list[dict]:
    cards: list[dict] = []
    for key, yahoo, label in MACRO_INSTRUMENTS:
        card = _quote_card(key, yahoo, label)
        if card:
            cards.append(card)

    # Supplement with themed headlines the linking layer already classified.
    # Anything naming a company belongs in the stock feed, not here.
    for n in fetch_news():
        if len(cards) >= limit + 2:
            break
        if n.get("tickers"):
            continue
        themes = [THEMES_BY_KEY[k] for k in n.get("themes", []) if k in THEMES_BY_KEY]
        if not themes:
            continue
        lead = themes[0]
        instruments = sorted({l["symbol"] for l in n.get("links", []) if l["type"] == "index"})
        cards.append(
            {
                "id": f"macro-news-{n['id']}",
                "title": n["headline"],
                "detail": n["summary"],
                "scope": lead.scope,
                "theme": lead.label,
                "sentiment": n["sentiment"],
                "impact": n["impact"],
                # A theme with no Indian equity read-through gets no instrument
                # tags, so it reads as global context and nothing more.
                "instruments": instruments or ([] if lead.equity_nexus <= 0 else ["NIFTY"]),
                "minutesAgo": n["minutesAgo"],
                "source": n["source"],
            }
        )

    cards.sort(key=lambda c: (-c["impact"], c["minutesAgo"]))
    return cards[:limit]
