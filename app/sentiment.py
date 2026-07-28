"""Headline polarity + impact scoring for Indian market news.

Lexicon scoring is intentionally lightweight (no GPU). India-specific terms —
NPA, GNPA, tezi/mandi, upper circuit, PAT, ROCE — come from the open-source
``nse-sentiment-analyzer`` financial boosters research. Signed corporate-event
bias (``events.py``) corrects cases where soft English words are absent, e.g.
"SEBI imposes ₹5 Cr penalty".
"""

from __future__ import annotations

from .events import blend_with_event, classify_event

# Token / short-phrase polarity. Matched as substrings inside a padded haystack.
POSITIVE = (
    # Generic market English
    "surge", "soar", "rally", "gain", "gains", "rise", "rises", "jump", "jumps",
    "beat", "beats", "record", "win", "wins", "bag", "bags",
    "upgrade", "upgrades", "profit", "profits", "growth", "strong", "boost",
    "approval", "approves", "deal", "expansion", "higher", "up ",
    "bullish", "outperform", "overweight", "upside", "accumulate", "breakout",
    "robust", "resilient", "stellar", "tailwind", "tailwinds", "oversubscribed",
    "inflow", "inflows", "buying", "doubled", "tripled", "multibagger",
    "infusion", "recapitalization", "appreciation", "buyout",
    # India metrics / shorthand (positive when present without a sink verb)
    " pat ", " ebitda ", " roe ", " roce ", " aum ", " nim ",
    # Hinglish
    " tezi ", " tej ", " chada ", " chade ", " chadi ",
)

NEGATIVE = (
    "fall", "falls", "slip", "slips", "drop", "drops", "crash", "plunge",
    "cut", "cuts", "miss", "misses", "loss", "losses", "weak", "downgrade",
    "probe", "fraud", "scam", "allegation", "allegations", "ban", "fine",
    "lawsuit", "default", "warning", "slump", "tumble", "tumbles", "lower",
    "down ", "selloff", "sell-off", "concern", "concerns",
    "bearish", "underperform", "underweight", "downside", "breakdown",
    "headwind", "headwinds", "undersubscribed", "outflow", "outflows",
    "selling", "mismanagement", "scrutiny", "depreciation", "deficit",
    "moratorium", "pledged", "slippage", "provisioning",
    # Banking stress — India-specific
    " npa ", " npas ", " gnpa ", " nnpa ", "non-performing", "non performing",
    # Circuit / governance
    "lower circuit", "insider trading", "front running", "margin call",
    # Hinglish
    " mandi ", " mand ", " gira ", " gire ", " giri ",
)

# Source priors aligned with the research doc's Bayesian defaults, keyed by
# hostname fragment. Tweets stay lower; wire services stay highest.
SOURCE_CREDIBILITY = {
    "reuters.com": 0.95,
    "bloomberg.com": 0.95,
    "economictimes.indiatimes.com": 0.90,
    "moneycontrol.com": 0.90,
    "livemint.com": 0.80,
    "thehindu.com": 0.80,
    "ndtv.com": 0.70,
    "business-standard.com": 0.80,
    "cnbctv18.com": 0.75,
    "news.google.com": 0.55,
    "x.com": 0.50,
    "twitter.com": 0.50,
}

SOURCE_NAME_CREDIBILITY = {
    "Economic Times Markets": 0.90,
    "Economic Times": 0.90,
    "Moneycontrol": 0.90,
    "LiveMint Markets": 0.80,
    "The Hindu Markets": 0.80,
    "NDTV Profit": 0.70,
    "FII/DII Flows": 0.75,
}


def _lexicon_score(text: str) -> float:
    t = f" {text.lower()} "
    pos = sum(1 for w in POSITIVE if w in t)
    neg = sum(1 for w in NEGATIVE if w in t)
    if pos == neg == 0:
        return 0.0
    return (pos - neg) / max(pos + neg, 1)


def polarity(text: str, *, title: str | None = None, body: str | None = None) -> tuple[str, float]:
    """Return ``(Positive|Negative|Neutral, score ∈ [-1, 1])``.

    When ``title`` is supplied, the signed event classifier can rescue
    financially loaded headlines that the lexicon alone would call Neutral.
    """
    lex = _lexicon_score(text)
    event_key, event_bias = classify_event(title or text, body or "")
    score = blend_with_event(lex, event_bias)
    if score > 0.15:
        return "Positive", score
    if score < -0.15:
        return "Negative", score
    # A detected event with a clear signed bias wins over a flat lexicon.
    if event_key and abs(event_bias) >= 0.2:
        return ("Positive" if event_bias > 0 else "Negative"), event_bias
    return "Neutral", score


def source_credibility(source_host: str = "", source_name: str = "") -> float:
    for host, c in SOURCE_CREDIBILITY.items():
        if host and host in (source_host or "").lower():
            return c
    if source_name and source_name in SOURCE_NAME_CREDIBILITY:
        return SOURCE_NAME_CREDIBILITY[source_name]
    return 0.6


def impact_score(
    *,
    text: str,
    source_host: str,
    minutes_ago: int,
    ticker_count: int,
    title: str | None = None,
    body: str | None = None,
    source_name: str = "",
) -> int:
    """Story impact 1–10 from credibility, urgency, breadth, polarity, and event."""
    _, pol = polarity(text, title=title, body=body)
    _, event_bias = classify_event(title or text, body or "")
    cred = source_credibility(source_host, source_name)
    freshness = 1.0 if minutes_ago <= 60 else 0.85 if minutes_ago <= 360 else 0.65 if minutes_ago <= 1440 else 0.4
    breadth = min(1.0, 0.55 + 0.15 * ticker_count)
    strength = min(1.0, max(abs(pol), abs(event_bias)) * 1.4 + 0.35)
    raw = 10 * (0.35 * cred + 0.25 * freshness + 0.2 * breadth + 0.2 * strength)
    return int(max(1, min(10, round(raw))))


def conviction_score(
    *,
    evidence_weight: float,
    agreement: float,
    source_count: int,
    direct_share: float,
    volume_ratio: float,
    price_agrees: bool | None,
    session_factor: float = 1.0,
    event_factor: float = 1.0,
    sr_factor: float = 1.0,
    tape_factor: float = 1.0,
) -> tuple[int, str, list[str]]:
    """How much to trust the read, 0–100, with the drivers spelled out.

    Sentiment says *which way*; conviction says *how much evidence*. A single
    low-relevance sector story and five corroborating company headlines can
    produce the same label, and this is what separates them.

    ``session_factor`` / ``event_factor`` are Phase B multipliers from the
    backtest: overnight timing and stress/M&A events lift trust; live-session
    timing and hype events (orders, beats, buybacks) cut it.
    """
    drivers: list[str] = []

    evidence = min(1.0, evidence_weight / 4.0)
    if evidence >= 0.6:
        drivers.append("multiple corroborating stories")
    elif evidence <= 0.25:
        drivers.append("thin evidence base")

    diversity = min(1.0, (source_count - 1) / 3.0) if source_count > 1 else 0.0
    if source_count >= 3:
        drivers.append(f"{source_count} independent sources")
    elif source_count == 1:
        drivers.append("single source")

    if agreement >= 0.7:
        drivers.append("sources agree")
    elif agreement <= 0.3:
        drivers.append("sources conflict")

    if direct_share >= 0.7:
        drivers.append("company-specific news")
    elif direct_share <= 0.3:
        drivers.append("mostly sector/macro read-through")

    confirmation = 0.0
    if volume_ratio >= 1.5:
        confirmation += 0.5
        drivers.append("volume confirming")
    elif volume_ratio and volume_ratio < 0.7:
        drivers.append("volume not confirming")
    if price_agrees is True:
        confirmation += 0.5
        drivers.append("price moving with the news")
    elif price_agrees is False:
        drivers.append("price moving against the news")

    raw = 100 * (0.32 * evidence + 0.16 * diversity + 0.22 * agreement + 0.16 * direct_share + 0.14 * confirmation)

    # Phase B calibration — applied after the structural score so drivers stay
    # readable and the adjustment is explicit.
    adj = 0
    if session_factor >= 1.15:
        adj += 8
        drivers.append("overnight / next-open timing")
    elif session_factor <= 0.85:
        adj -= 10
        drivers.append("live-session timing (weaker historically)")

    if event_factor >= 1.2:
        adj += 10
        drivers.append("stress/M&A event type")
    elif event_factor <= 0.65:
        adj -= 12
        drivers.append("hype event type demoted")

    # Phase C — support/resistance structure. A bullish read only holds up when
    # price is coiled against a barrier (breakout); a mid-range long is the
    # weakest surface in the backtest and is cut hard.
    if sr_factor >= 1.2:
        adj += 10
        drivers.append("breakout setup (pressed against resistance)")
    elif sr_factor <= 0.65:
        adj -= 15
        drivers.append("bullish but stuck mid-range (no breakout)")
    elif sr_factor >= 1.05:
        adj += 4
        drivers.append("breakdown / at-support continuation")

    # Phase D — overnight volume/VWAP tape (closed-session only). Live news
    # leaves tape_factor at 1.0; extended-above-VWAP longs are demoted hard.
    if tape_factor <= 0.55:
        adj -= 18
        drivers.append("extended ≥1% above VWAP (fade zone)")
    elif tape_factor >= 1.25:
        adj += 10
        drivers.append("vol surge + below VWAP (overnight tape)")
    elif tape_factor >= 1.15:
        adj += 8
        drivers.append("prior-day volume surge")
    elif tape_factor >= 1.05:
        adj += 5
        drivers.append("prior close still below VWAP")

    score = int(max(0, min(100, round(raw + adj))))
    label = "high" if score >= 60 else "medium" if score >= 40 else "low"
    return score, label, drivers[:5]


def bias_and_action(
    sentiment: str,
    impact: int,
    *,
    change_pct: float = 0.0,
    move_since_news_pct: float | None = None,
    breakout_long: bool = False,
    structure: dict | None = None,
    tape_blocks_buy: bool = False,
    tape_supports_buy: bool = False,
) -> tuple[str, str, str | None]:
    """Returns (bias, action, actionNote).

    Actions are framed as executable intents — ``buy long`` / ``buy short`` —
    not as positive/negative mood. Structure and tape still annotate *why*,
    and the publish gate (conviction tier) decides whether the board lights green.
    """
    move = move_since_news_pct if move_since_news_pct is not None else change_pct
    note: str | None = None
    if move_since_news_pct is not None:
        note = f"{move_since_news_pct:+.1f}% since headline"
    elif change_pct:
        note = f"{change_pct:+.1f}% today"

    if sentiment == "Positive":
        bias = "bullish"
        if move >= 1.5 or (impact >= 8 and change_pct >= 1.0):
            action = "already priced"
        elif tape_blocks_buy:
            action = "watch"
            tag = "extended above VWAP — fade risk (overnight)"
            note = f"{note} · {tag}" if note else tag
        else:
            action = "buy long"
            if breakout_long:
                level = (structure or {}).get("nearestResistance")
                tag = f"breakout setup near {level}" if level else "breakout setup"
                if tape_supports_buy:
                    tag = f"{tag} + overnight tape"
            else:
                tag = "no breakout yet (mid-range)"
            note = f"{note} · {tag}" if note else tag
    elif sentiment == "Negative":
        bias = "bearish"
        if move <= -1.5:
            action = "already fallen"
        else:
            action = "buy short"
            if impact >= 7 and move > -1.0:
                tag = "stress still open"
            elif impact >= 5:
                tag = "bearish lean"
            else:
                tag = "soft bearish lean"
            note = f"{note} · {tag}" if note else tag
    else:
        bias = "mixed"
        action = "watch"
    return bias, action, note


# Publish / visual tiers — green only for strong conviction, never red for shorts.
SIGNAL_STRONG_CONV = 60
SIGNAL_MEDIUM_CONV = 40


def theme_conflict(related_news: list[dict], *, story_direction) -> bool:
    """True when bullish and bearish evidence are both material and close in weight."""
    up = 0.0
    down = 0.0
    for n in related_news:
        d = int(story_direction(n) or 0)
        if d == 0:
            continue
        impact = float(n.get("impact") or 1) / 10.0
        relevance = float(n.get("relevance", 1.0))
        mins = n.get("minutesAgo") or 9999
        freshness = 1.0 if mins <= 60 else 0.85 if mins <= 360 else 0.65 if mins <= 1440 else 0.4
        w = impact * relevance * freshness
        if d > 0:
            up += w
        else:
            down += w
    if up < 0.12 or down < 0.12:
        return False
    return (min(up, down) / max(up, down)) >= 0.55


def publish_signal(
    bias: str,
    action: str,
    conviction: int,
    *,
    conflict: bool = False,
) -> str:
    """Gate the executable label. Colouring is conviction-tier on the client.

    Buy long / buy short only when conviction ≥ 60 and evidence is clean.
    Theme conflict, tape-blocked watch, or thinner evidence → watch.
    """
    if action in {"already priced", "already fallen"}:
        return action
    if conflict:
        return "watch"
    if conviction < SIGNAL_STRONG_CONV or bias == "mixed":
        return "watch"
    # Keep prior gates (e.g. tape-blocked watch) — do not revive into a buy.
    if action in {"buy long", "buy"}:
        return "buy long"
    if action in {"buy short", "short", "avoid"}:
        return "buy short"
    return "watch"
