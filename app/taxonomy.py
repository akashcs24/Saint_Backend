"""Sector taxonomy, macro themes and corporate-event vocabulary.

The linking layer reasons about a headline on three levels:

* **Company** — the business is named, so the story is company news.
* **Sector** — a theme the sector is sensitive to moved (crude, rupee, rates,
  GST), so the story is *context*, never company news.
* **Market** — index-level risk appetite only.

Sensitivities are signed. ``+1`` means the sector moves *with* the theme's
detected direction, ``-1`` means it moves against it: crude rising is good for
upstream oil and bad for airlines, so ``OILGAS`` is positive and ``AVIATION``
negative on the same theme.

``equity_nexus`` is the safety valve. A theme scoring 0 (crypto, for instance)
has no read-through to Indian equities and is never propagated to a stock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

SECTORS: dict[str, str] = {
    "IT": "IT & Software",
    "BANK": "Banks",
    "NBFC": "NBFC & Financials",
    "INSURANCE": "Insurance",
    "FMCG": "FMCG",
    "AUTO": "Automobiles",
    "PHARMA": "Pharma & Healthcare",
    "METALS": "Metals & Mining",
    "OILGAS": "Oil & Gas",
    "CEMENT": "Cement",
    "POWER": "Power & Utilities",
    "TELECOM": "Telecom",
    "INFRA": "Infra & Capital Goods",
    "REALTY": "Realty",
    "CHEMICALS": "Chemicals & Paints",
    "CONSUMER": "Consumer Discretionary",
    "RETAIL": "Retail & Internet",
    "HOTELS": "Hotels & Travel",
    "AVIATION": "Aviation",
    "PORTS": "Ports & Logistics",
    "DIVERSIFIED": "Diversified",
    "FINTECH": "Fintech",
}


@dataclass(frozen=True)
class Theme:
    """A macro/thematic driver with signed sector sensitivities."""

    key: str
    label: str
    scope: str
    patterns: tuple[str, ...]
    # Phrases that describe the same driver from the opposite side, e.g. a
    # rising dollar index is a weakening rupee.
    inverse_patterns: tuple[str, ...] = ()
    up: tuple[str, ...] = ()
    down: tuple[str, ...] = ()
    sector_sensitivity: Mapping[str, float] = field(default_factory=dict)
    # Sensitivities for individual names that are far more exposed than their
    # sector — cigarette tax is an ITC story, not an FMCG story.
    symbol_sensitivity: Mapping[str, float] = field(default_factory=dict)
    index_sensitivity: float = 0.0
    equity_nexus: float = 1.0
    ceiling: float = 0.45
    # When true, the theme only counts if one of its own up/down words is
    # present. Stops a passing mention ("cigarette volumes grew") from being
    # read as a regulatory event.
    requires_direction: bool = False


# Direction words, scored inside a window around the theme match so that
# "crude slips" and "rupee slips" resolve independently in one headline.
DIRECTION_UP: tuple[str, ...] = (
    "surge", "surges", "surged", "jump", "jumps", "jumped", "rise", "rises", "rising", "rose",
    "rally", "rallies", "climb", "climbs", "climbed", "gain", "gains", "higher", "high",
    "spike", "spikes", "soar", "soars", "hike", "hikes", "raise", "raises", "raised",
    "above", "tops", "advance", "advances", "strengthen", "strengthens", "firm", "firmer",
    "widen", "widens", "accelerate", "accelerates", "up",
    "hit", "hits", "cross", "crosses", "breach", "breaches", "peak", "peaks", "record",
)
DIRECTION_DOWN: tuple[str, ...] = (
    "fall", "falls", "fell", "falling", "slip", "slips", "slipped", "drop", "drops", "dropped",
    "decline", "declines", "slump", "slumps", "tumble", "tumbles", "plunge", "plunges",
    "crash", "crashes", "lower", "low", "cut", "cuts", "ease", "eases", "eased", "below",
    "weaken", "weakens", "weaker", "depreciate", "depreciates", "soften", "softens",
    "retreat", "retreats", "narrow", "narrows", "cool", "cools", "down",
)


THEMES: tuple[Theme, ...] = (
    Theme(
        key="crypto",
        label="Crypto",
        scope="crypto",
        patterns=(
            "bitcoin", "btc", "ethereum", "ether", "solana", "dogecoin", "altcoin",
            "crypto", "cryptocurrency", "stablecoin", "binance", "coinbase", "memecoin",
        ),
        # Crypto has no read-through to an NSE cash equity. Kept as a tracked
        # theme so it can still surface as global context, never as stock news.
        equity_nexus=0.0,
        ceiling=0.0,
    ),
    Theme(
        key="crude",
        label="Crude oil",
        scope="commodity",
        patterns=("crude", "brent", "wti", "opec", "oil price", "oil prices", "petrol price", "diesel price"),
        # A supply cut lifts the price, so oil needs its own vocabulary: the
        # generic reading of "cut" would point the wrong way.
        up=(
            "surge", "surges", "jump", "jumps", "rise", "rises", "rally", "rallies", "climb", "climbs",
            "gain", "gains", "spike", "spikes", "soar", "soars", "higher", "above", "hit", "hits", "tops",
            "supply cut", "output cut", "production cut", "supply risk", "supply disruption",
        ),
        down=(
            "fall", "falls", "slip", "slips", "drop", "drops", "ease", "eases", "decline", "declines",
            "slump", "slumps", "tumble", "tumbles", "plunge", "plunges", "lower", "below", "retreat",
            "retreats", "cool", "cools", "glut", "oversupply", "output hike", "demand destruction",
        ),
        sector_sensitivity={
            "OILGAS": 0.75,
            "AVIATION": -0.9,
            "CHEMICALS": -0.7,
            "AUTO": -0.45,
            "CEMENT": -0.4,
            "FMCG": -0.3,
            "PORTS": -0.2,
        },
        index_sensitivity=-0.4,
        ceiling=0.5,
    ),
    Theme(
        key="rupee",
        label="Rupee",
        scope="currency",
        patterns=("rupee", "usd/inr", "usdinr", "inr against"),
        inverse_patterns=("dollar index", "greenback", "us dollar strength"),
        sector_sensitivity={
            "IT": -0.8,
            "PHARMA": -0.55,
            "METALS": -0.3,
            "OILGAS": 0.5,
            "AVIATION": 0.5,
            "CHEMICALS": 0.3,
        },
        index_sensitivity=0.2,
        ceiling=0.5,
    ),
    Theme(
        key="rates_in",
        label="RBI policy",
        scope="monetary",
        patterns=(
            "rbi", "repo rate", "reverse repo", "monetary policy committee", "mpc meeting",
            "crr", "slr", "liquidity measures", "rbi governor",
        ),
        up=("hike", "hikes", "raise", "raises", "tighten", "tightens"),
        down=("cut", "cuts", "slash", "slashes", "ease", "eases", "easing"),
        sector_sensitivity={
            "BANK": 0.3,
            "NBFC": -0.65,
            "REALTY": -0.7,
            "AUTO": -0.5,
            "INFRA": -0.4,
            "FINTECH": -0.4,
        },
        index_sensitivity=-0.45,
        ceiling=0.55,
    ),
    Theme(
        key="rates_global",
        label="Fed & global rates",
        scope="monetary",
        patterns=("federal reserve", "fed rate", "fomc", "jerome powell", "ecb", "bank of japan", "treasury yield"),
        up=("hike", "hikes", "raise", "raises", "tighten", "tightens"),
        down=("cut", "cuts", "ease", "eases", "easing", "dovish"),
        sector_sensitivity={"IT": -0.35, "NBFC": -0.35, "METALS": -0.3},
        index_sensitivity=-0.35,
        ceiling=0.4,
    ),
    Theme(
        key="geopolitics",
        label="Geopolitics",
        scope="geopolitics",
        patterns=(
            "geopolit", "middle east", "israel", "iran", "gaza", "red sea", "ukraine",
            "russia sanctions", "war", "military strike", "ceasefire", "border tension",
        ),
        up=("escalate", "escalates", "escalation", "strike", "strikes", "attack", "attacks", "tension", "tensions"),
        down=("ceasefire", "truce", "de-escalate", "peace deal", "talks resume"),
        sector_sensitivity={"OILGAS": 0.4, "AVIATION": -0.4, "HOTELS": -0.3, "IT": -0.2},
        index_sensitivity=-0.55,
        ceiling=0.4,
    ),
    Theme(
        key="trade",
        label="Tariffs & trade",
        scope="policy",
        patterns=("tariff", "tariffs", "trade deal", "trade war", "import duty", "export duty", "anti-dumping", "wto"),
        up=("impose", "imposes", "imposed", "hike", "hikes", "raise", "raises", "levy", "levies"),
        down=("scrap", "scraps", "waive", "waives", "exempt", "exempts", "rollback", "roll back"),
        sector_sensitivity={"METALS": -0.6, "IT": -0.4, "PHARMA": -0.5, "AUTO": -0.35, "CHEMICALS": -0.35},
        index_sensitivity=-0.3,
        ceiling=0.5,
        requires_direction=True,
    ),
    Theme(
        key="tax",
        label="Tax & GST",
        scope="policy",
        patterns=("gst", "gst council", "excise duty", "cess", "income tax", "corporate tax", "capital gains tax", "sin tax"),
        up=("hike", "hikes", "raise", "raises", "impose", "imposes", "increase", "increases"),
        down=("cut", "cuts", "reduce", "reduces", "relief", "exempt", "exempts", "rationalise", "rationalize"),
        sector_sensitivity={"FMCG": -0.5, "AUTO": -0.5, "CONSUMER": -0.5, "RETAIL": -0.4, "HOTELS": -0.4},
        index_sensitivity=-0.2,
        ceiling=0.55,
        requires_direction=True,
    ),
    Theme(
        key="tobacco",
        label="Tobacco regulation",
        scope="policy",
        patterns=("cigarette", "cigarettes", "tobacco", "smoking", "nicotine", "beedi", "pictorial warning"),
        up=("hike", "hikes", "tax", "cess", "ban", "bans", "curb", "curbs", "restrict", "restricts", "warning", "levy"),
        down=("relief", "exempt", "exempts", "rollback", "roll back", "ease", "eases"),
        # Direction "up" means tighter or costlier regulation. Cigarettes are
        # the bulk of ITC's profit pool, so this is a single-name theme.
        symbol_sensitivity={"ITC": -0.9},
        ceiling=0.65,
        requires_direction=True,
    ),
    Theme(
        key="monsoon",
        label="Monsoon & rural demand",
        scope="weather",
        patterns=("monsoon", "rainfall", "imd forecast", "kharif", "rabi", "rural demand", "msp", "sowing"),
        up=("above normal", "surplus", "good", "revive", "revives", "boost", "boosts"),
        down=("deficit", "below normal", "drought", "weak", "delay", "delays", "erratic"),
        sector_sensitivity={"FMCG": 0.6, "AUTO": 0.45, "CONSUMER": 0.4, "NBFC": 0.3, "CHEMICALS": 0.3},
        index_sensitivity=0.2,
        ceiling=0.5,
        requires_direction=True,
    ),
    Theme(
        key="gold",
        label="Gold",
        scope="commodity",
        patterns=("gold price", "gold prices", "bullion", "gold rate", "silver price"),
        symbol_sensitivity={"TITAN": 0.5},
        sector_sensitivity={"NBFC": 0.3},
        ceiling=0.4,
    ),
    Theme(
        key="global_equity",
        label="Global equities",
        scope="global",
        patterns=("dow jones", "nasdaq", "s&p 500", "wall street", "asian markets", "nikkei", "hang seng", "sgx nifty", "gift nifty"),
        sector_sensitivity={"IT": 0.4},
        index_sensitivity=0.5,
        ceiling=0.35,
    ),
    Theme(
        key="inflation",
        label="Inflation",
        scope="macro",
        patterns=("inflation", "cpi ", "wpi ", "retail prices", "food prices", "core inflation"),
        sector_sensitivity={"FMCG": -0.35, "CONSUMER": -0.35, "BANK": -0.25, "NBFC": -0.3},
        index_sensitivity=-0.3,
        ceiling=0.45,
    ),
    Theme(
        key="flows",
        label="FII / DII flows",
        scope="flows",
        patterns=(
            "fii", "fiis", "dii", "diis", "foreign institutional", "domestic institutional",
            "foreign portfolio", "fpi", "fpis",
        ),
        up=("buy", "buys", "bought", "inflow", "inflows", "net buyers", "pour", "pours"),
        down=("sell", "sells", "sold", "outflow", "outflows", "net sellers", "pull out", "dump"),
        index_sensitivity=0.5,
        ceiling=0.35,
    ),
    Theme(
        key="banking_regulation",
        label="Banking regulation",
        scope="policy",
        patterns=("npa", "asset quality", "provisioning norms", "basel", "credit growth", "deposit growth", "bad loans"),
        sector_sensitivity={"BANK": 0.6, "NBFC": 0.5, "FINTECH": 0.35},
        index_sensitivity=0.2,
        ceiling=0.5,
    ),
    Theme(
        key="infra_spend",
        label="Capex & infra spend",
        scope="policy",
        patterns=("capex", "infrastructure spending", "budget allocation", "road projects", "railway order", "pli scheme", "national highway"),
        sector_sensitivity={"INFRA": 0.6, "CEMENT": 0.5, "METALS": 0.45, "POWER": 0.4},
        index_sensitivity=0.25,
        ceiling=0.5,
    ),
)

THEMES_BY_KEY: dict[str, Theme] = {t.key: t for t in THEMES}


# How to describe a theme's own movement, per scope, as (up, down, unclear).
_SCOPE_PHRASES: dict[str, tuple[str, str, str]] = {
    "commodity": ("rising", "falling", "in focus"),
    "currency": ("strengthening", "weakening", "in focus"),
    "monetary": ("tightening", "easing", "in focus"),
    "policy": ("tightening", "loosening", "in focus"),
    "geopolitics": ("escalating", "de-escalating", "in focus"),
    "flows": ("buying", "selling", "in focus"),
    "weather": ("improving", "deteriorating", "in focus"),
    "global": ("rising", "falling", "in focus"),
    "macro": ("rising", "falling", "in focus"),
    "crypto": ("rising", "falling", "in focus"),
}


def theme_phrase(theme: Theme, direction: int) -> str:
    up, down, flat = _SCOPE_PHRASES.get(theme.scope, ("rising", "falling", "in focus"))
    return up if direction > 0 else down if direction < 0 else flat


# Corporate events. These do not link a stock on their own; they raise
# confidence that a matched company name really is the subject of the story.
CORPORATE_EVENTS: dict[str, tuple[str, ...]] = {
    "earnings": (
        "q1 results", "q2 results", "q3 results", "q4 results", "quarterly results",
        "net profit", "revenue rose", "revenue fell", "ebitda", "margin", "earnings",
        "profit rises", "profit falls", "beats estimates", "misses estimates", "guidance",
    ),
    "order_win": ("bags order", "wins order", "order win", "contract win", "awarded a contract", "loi", "order book"),
    "mna": ("acquire", "acquires", "acquisition", "merger", "stake sale", "divest", "demerger", "open offer", "takeover"),
    "rating": ("upgrade", "upgrades", "downgrade", "downgrades", "target price", "initiate coverage", "rating cut", "rerating"),
    "regulatory": ("sebi", "cci ", "probe", "investigation", "show cause", "penalty", "fine", "lawsuit", "nclt", "insolvency"),
    "capital": ("qip", "rights issue", "buyback", "dividend", "bonus issue", "stock split", "fund raise", "preferential issue"),
    "management": ("resigns", "steps down", "appoints", "new ceo", "new md", "board approves"),
    "block_deal": ("block deal", "bulk deal", "promoter stake", "pledge", "offer for sale"),
    "operations": ("plant", "capacity expansion", "new facility", "production halt", "strike at", "recall", "price hike"),
}


# Curated association edges. ``direction`` is +1 when the two names move
# together (parent/subsidiary, supplier) and -1 when one gains at the other's
# expense (direct competitors fighting for the same wallet).
RELATED: tuple[tuple[str, str, float, int, str], ...] = (
    ("ITC", "ITCHOTELS", 0.45, 1, "demerged hotels arm"),
    ("RELIANCE", "BHARTIARTL", 0.35, -1, "telecom market share rivalry"),
    ("BHARTIARTL", "IDEA", 0.3, -1, "telecom market share rivalry"),
    ("HINDUNILVR", "ITC", 0.3, 1, "FMCG demand read-across"),
    ("HINDUNILVR", "BRITANNIA", 0.3, 1, "FMCG demand read-across"),
    ("TATAMOTORS", "M&M", 0.3, 1, "auto demand read-across"),
    ("MARUTI", "TATAMOTORS", 0.3, -1, "passenger vehicle share rivalry"),
    ("TATASTEEL", "JSWSTEEL", 0.4, 1, "steel cycle read-across"),
    ("HINDALCO", "VEDL", 0.35, 1, "base metals cycle read-across"),
    ("HDFCBANK", "ICICIBANK", 0.35, 1, "private bank read-across"),
    ("TCS", "INFY", 0.45, 1, "IT services demand read-across"),
    ("INFY", "WIPRO", 0.4, 1, "IT services demand read-across"),
    ("ONGC", "OIL", 0.35, 1, "upstream crude realisation"),
    ("ZOMATO", "PAYTM", 0.2, 1, "new-age internet risk appetite"),
)
