"""Signed corporate-event classifier for Indian financial headlines.

Inspired by the five-tier event-aware scoring pattern used in
``nse-sentiment-analyzer`` (AshayK003): VADER alone is blind to phrases like
"SEBI imposes ₹5 Cr penalty", so a first-match-wins event map supplies a
signed bias that polarity can blend with.

Events are ordered most-specific first. A beat must beat a generic "profit"
mention; litigation must beat a bare "SEBI" mention.
"""

from __future__ import annotations

import re
from functools import lru_cache

# (event_key, base_sentiment ∈ [-1, 1], patterns)
EVENT_MAP: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    # ── Positive ──────────────────────────────────────────────
    (
        "buyback_dividend",
        0.30,
        (
            r"\bbuy[- ]?back\b",
            r"\bbonus\s+(?:issue|shares|ratio)\b",
            r"\bdividend\s+(?:declared|announced|approved|interim|final|payout)\b",
            r"(?:declares?|announced|approved|interim|final)\s+(?:\S+\s+)?dividend\b",
            r"\b(?:share|stock)\s+split\b",
        ),
    ),
    (
        "order_win",
        0.30,
        (
            r"\bwins?\b\s+.*?\b(?:contract|order|deal|mandate)\b",
            r"\bbags?\b\s+.*?\b(?:order|contract|deal)\b",
            r"\bsecure[sd]?\b\s+.*?\b(?:order|contract|deal)\b",
            r"\b(?:order|contract)\s+worth\b",
        ),
    ),
    (
        "earnings_beat",
        0.35,
        (
            r"\bprofit\s+(?:jumps?|surges?|rises?|grows?|soars?)\b",
            r"\brevenue\s+(?:jumps?|surges?|rises?|grows?)\b",
            r"\bbeat[s]?\s+(?:estimates?|expectations?|consensus)\b",
            r"\bstrong\s+(?:results?|quarter|performance|show)\b",
            r"\brecord\s+(?:profit|revenue|income|quarter(?:ly)?)\b",
            r"\bstellar\b.*?\b(?:results?|quarter|earnings|revenue)\b",
        ),
    ),
    (
        "guidance_positive",
        0.30,
        (
            r"\braises?\b\s+.*?\b(?:guidance|outlook|forecast)\b",
            r"\b(?:upbeat|positive|confident)\s+(?:outlook|guidance|view)\b",
        ),
    ),
    (
        "rating_upgrade",
        0.20,
        (
            r"\bupgrad(?:e|es|ed)\s+(?:credit\s+)?(?:rating|outlook|to\s+buy|to\s+outperform)\b",
            r"\brating\s+upgrad(?:e|ed)\b",
            r"\binitiates?\s+coverage\s+with\s+(?:buy|outperform|overweight)\b",
        ),
    ),
    (
        "regulatory_approval",
        0.15,
        (
            r"\b(?:SEBI|RBI|CCI|NCLT|IRDAI)\s+(?:approves?|clears?|allows?|okays?)\b",
            r"\bgets?\s+(?:SEBI|RBI|regulatory)\s+(?:approval|clearance|nod)\b",
        ),
    ),
    (
        "merger_acquisition",
        0.25,
        (
            r"\bmergers?\b",
            r"\bacqui(?:re|res|red|sition)\b",
            r"\btakeover\b",
            r"\bbuy(?:s|ing)?\s+.*?\bstake\b",
            r"\bstake\s+buy\b",
            r"\bdeal\s+to\s+(?:buy|acquire|merge)\b",
        ),
    ),
    (
        "jv_mou",
        0.15,
        (
            r"\bjoint\s+venture\b|\b\bJV\b",
            r"\bMoU\b|\bmemorandum\s+of\s+understanding\b",
            r"\bstrategic\s+(?:alliance|partnership|collaboration|tie[- ]up)\b",
            r"\bsigns?\s+(?:MoU|MOU|agreement|pact|partnership)\b",
        ),
    ),
    (
        "fundraise",
        0.15,
        (
            r"\bQIP\b|\bFPO\b|\brights\s+issue\b",
            r"\bpreferential\s+(?:issue|allotment)\b",
            r"\bfund\s+raise\b|\braises?\s+.*?\b(?:via|through)\s+(?:QIP|FPO|rights)\b",
        ),
    ),
    (
        "expansion",
        0.20,
        (
            r"\bexpansion\s+(?:plan|drive|into|mode|strategy)\b",
            r"\bnew\s+(?:plant|factory|facility|unit|venture)\b",
            r"\binvests?\b\s+.*?\d+\s*(?:cr|crore|mn|bn)\b",
            r"\bforay\s+into\b",
        ),
    ),
    (
        "product_launch",
        0.20,
        (
            r"\blaunch(?:es|ed)?\s+(?:new|first|india)\b",
            r"\bunveils?\s+.*?\b(?:product|service|platform|app|vehicle|range|plan)\b",
        ),
    ),
    # ── Negative ─────────────────────────────────────────────
    (
        "debt_stress",
        -0.40,
        (
            r"\bdowngrad(?:e|es|ed)\s+(?:credit\s+)?(?:rating|outlook)\b",
            r"\bcredit\s+watch\b",
            r"\bdefault(?:s|ed)?\s+on\s+(?:debt|payment|obligation)\b",
            r"\bNPAs?\b|\bGNPA\b|\bNNPA\b|\bnon[- ]performing\b",
            r"\binsolvency\b|\bbankruptcy\b",
            r"\bdebt\s+(?:trap|burden|crisis|restructuring)\b",
            r"\bliquidity\s+(?:crisis|crunch|squeeze|stress)\b",
        ),
    ),
    (
        "litigation",
        -0.35,
        (
            r"\b(?:imposes?|levies?|slaps?)\b.*?\b(?:penalty|fine)\b",
            r"\bpenalty\s+(?:of|imposed|levied)\b",
            r"\bSEBI\s+(?:probe|investigat|notice|order|directive|slaps?|summon)\b",
            r"\b(?:CBI|ED|SFIO|DRI|NIA)\s+(?:probe|investigat|raids?|attachment|files?\s+case)\b",
            r"\bincome\s+tax\s+(?:raids?|survey|notice|probe)\b",
            r"\bshow[- ]cause\s+notice\b",
            r"\bfraud\b|\bscam\b|\bembezzlement\b",
            r"\blawsuit\b|\blitigation\b",
        ),
    ),
    (
        "earnings_miss",
        -0.35,
        (
            r"\bprofit\s+(?:falls?|declines?|drops?|plunges?|slips?|tumbles|shrinks)\b",
            r"\brevenue\s+(?:falls?|declines?|drops?|plunges?|slips?|tumbles)\b",
            r"\bmiss(?:es)?\s+(?:estimates?|expectations?|target|consensus)\b",
            r"\bweak\s+(?:results?|quarter|performance|show|demand)\b",
            r"\bbelow\s+(?:estimates?|expectations?|consensus|street)\b",
            r"\bloss\s+(?:widens?|deepens?|mounts?|swells)\b",
            r"\bdisappointing\s+(?:results?|quarter|performance)\b",
        ),
    ),
    (
        "guidance_negative",
        -0.30,
        (
            r"\b(?:lowers?|cuts?)\b\s+.*?\b(?:guidance|outlook|forecast)\b",
            r"\bcautious\s+(?:outlook|guidance|view)\b",
        ),
    ),
    (
        "rating_downgrade",
        -0.25,
        (
            r"\bdowngrad(?:e|es|ed)\s+(?:to\s+)?(?:sell|underperform|underweight|reduce)\b",
            r"\brating\s+cut\b|\bcuts?\s+target\s+price\b",
        ),
    ),
    (
        "regulatory",
        -0.30,
        (
            r"\bregulatory\s+(?:crackdown|hurdle|barrier|issue|action|probe)\b",
            r"\bRBI\s+(?:restricts?|curbs?|directive|action|slaps?|penalty)\b",
            r"\b(?:CCI|competition\s+commission)\s+(?:probe|notice)\b",
            r"\b(?:GST|tax)\s+notice\b",
            r"\bNCLT\b|\bNCLAT\b",
        ),
    ),
    (
        "contract_loss",
        -0.30,
        (
            r"\bloses?\b\s+.*?\b(?:contract|order|deal|mandate|client)\b",
            r"\b(?:contract|order|deal)\s+(?:lost|terminated|cancelled)\b",
            r"\bcancell?(?:ed|ation)?\s+.*?\b(?:order|contract|deal)\b",
        ),
    ),
    (
        "mgmt_change_negative",
        -0.20,
        (
            r"\bresign(?:s|ed|ation)\b",
            r"\bsteps?\s+down\b",
            r"\boust(?:er|ed|s)\b",
            r"\bquit(?:s|ted)?\b\s+(?:as|from)\b",
            r"\bsack(?:ed|s)?\b",
        ),
    ),
    (
        "divestment",
        -0.15,
        (
            r"\bdivest(?:s|ed|iture)?\b",
            r"\bpromoter\s+sells?\b",
            r"\bstake\s+sale\b",
            r"\bsells?\s+(?:stake|subsidiary|unit)\b",
        ),
    ),
)


@lru_cache(maxsize=1)
def _compiled() -> tuple[tuple[str, float, tuple[re.Pattern[str], ...]], ...]:
    return tuple(
        (key, bias, tuple(re.compile(p, re.IGNORECASE) for p in patterns))
        for key, bias, patterns in EVENT_MAP
    )


def classify_event(title: str, body: str = "") -> tuple[str | None, float]:
    """Return ``(event_key, signed_bias)``. First match wins."""
    text = f"{title} {body or ''}".strip()
    if not text:
        return None, 0.0
    for key, bias, patterns in _compiled():
        for pattern in patterns:
            if pattern.search(text):
                return key, bias
    return None, 0.0


# Evidence tiers from the 60-day Nifty backtest: stress/M&A moved price;
# order wins, beats, buybacks and MoUs were mostly flat or wrong-way.
EVENT_STRESS: frozenset[str] = frozenset(
    {
        "earnings_miss",
        "debt_stress",
        "litigation",
        "merger_acquisition",
        "guidance_negative",
        "rating_downgrade",
        "contract_loss",
        "mgmt_change_negative",
        "regulatory",
    }
)
EVENT_HYPE: frozenset[str] = frozenset(
    {
        "order_win",
        "buyback_dividend",
        "earnings_beat",
        "jv_mou",
        "fundraise",
        "product_launch",
        "expansion",
        "guidance_positive",
        "rating_upgrade",
        "regulatory_approval",
    }
)


def event_evidence_multiplier(event_key: str | None) -> float:
    """Scale story weight in conviction aggregation by event type."""
    if not event_key:
        return 1.0
    if event_key in EVENT_HYPE:
        return 0.5
    if event_key in EVENT_STRESS:
        return 1.4
    return 1.0


def is_hype_event(event_key: str | None) -> bool:
    return bool(event_key) and event_key in EVENT_HYPE


def is_stress_event(event_key: str | None) -> bool:
    return bool(event_key) and event_key in EVENT_STRESS


def blend_with_event(polarity_score: float, event_bias: float) -> float:
    """Blend lexical polarity with the event's signed bias.

    When the lexicon is confident (|score| > 0.3), trust it and nudge.
    When it is uncertain — typical of regulatory headlines with no soft
    sentiment words — let the event dominate.
    """
    if event_bias == 0.0:
        return polarity_score
    if abs(polarity_score) > 0.3:
        blended = 0.8 * polarity_score + 0.2 * event_bias
    else:
        blended = 0.3 * polarity_score + 0.7 * event_bias
    return max(-1.0, min(1.0, blended))
