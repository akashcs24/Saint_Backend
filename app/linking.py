"""The intelligence layer: headline → which stocks it actually applies to, and why.

Pipeline per headline:

1. **Normalise** — lowercase, collapse whitespace, unify punctuation.
2. **Resolve entities** — token-boundary alias matching with deny-context
   guards and a prefer-the-specific-name rule.
3. **Detect themes** — macro drivers plus the direction they moved, scored in a
   window around each match so two drivers in one headline resolve separately.
4. **Detect corporate events** — earnings, orders, M&A, regulatory. These
   corroborate a weak name match.
5. **Propagate** — themes reach stocks through signed sector sensitivities,
   named companies reach peers through curated association edges, and index
   read-through is kept at index level.
6. **Score relevance** — every link carries a 0–1 relevance, a type, an
   expected direction, and a sentence explaining itself.

A link is only company news when a name was actually matched. Everything else
is explicitly labelled context, and themes with no Indian equity read-through
(crypto) propagate nowhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .events import classify_event
from .taxonomy import (
    DIRECTION_DOWN,
    DIRECTION_UP,
    RELATED,
    SECTORS,
    THEMES,
    THEMES_BY_KEY,
    Theme,
    theme_phrase,
)
from .universe import UNIVERSE, symbols_in_sector

# Relevance a link must reach to be shown as company news.
DIRECT_MIN = 0.60
# Relevance a link must reach to be shown at all (as sector/market context).
# Raised from 0.18 — weak theme→sector spray was flooding conviction with junk.
CONTEXT_MIN = 0.28
# Peer / sector links below this are ignored for stock scoring (kill junk).
ACTIONABLE_CONTEXT_MIN = 0.50

# An uncorroborated weak alias must land below DIRECT_MIN, so it shows as a
# possible link rather than as confirmed company news.
_TIER_BASE = {"strong": 0.95, "medium": 0.85, "weak": 0.50}
# A weak alias ("ril", "hul") only becomes company news with corroboration.
_WEAK_CORROBORATED = 0.80

# Words that mark a story as being about the Indian market, used to corroborate
# weak alias hits.
_INDIA_CONTEXT = (
    "nse", "bse", "sensex", "nifty", "dalal street", "sebi", "rbi", "crore", "lakh",
    "rupee", "india", "indian", "mumbai", "shares", "stock", "equity", "listed",
)

_DIRECTION_WINDOW = 70

# Clause separators. Direction words are only counted within the clause that
# mentions the theme.
_CLAUSE_SPLIT = re.compile(
    r"[;:,.]|\s(?:as|while|but|after|amid|amidst|though|although|despite|ahead\sof|even\sas|and)\s"
)

# A theme with no equity read-through (crypto) tells us the story is offshore,
# so any secondary theme in the same headline is a passing mention at best.
_OFFSHORE_DAMPING = 0.5

# A theme mentioned only in the body of a story is supporting detail, not what
# the story is about.
_BODY_ONLY_DISCOUNT = 0.5

# An indirect link whose direction could not be resolved is weak evidence: we
# know the stock is exposed, not which way.
_UNCLEAR_DISCOUNT = 0.7


@dataclass(frozen=True)
class EntityHit:
    symbol: str
    alias: str
    tier: str
    start: int
    end: int


@dataclass(frozen=True)
class ThemeHit:
    key: str
    label: str
    scope: str
    direction: int
    confidence: float


@dataclass(frozen=True)
class Link:
    """One resolved headline → symbol relationship."""

    symbol: str
    relevance: float
    link_type: str  # direct | peer | sector | index
    reason: str
    direction: int  # +1 tailwind, -1 headwind, 0 unclear

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "relevance": round(self.relevance, 3),
            "type": self.link_type,
            "reason": self.reason,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class Analysis:
    direct: list[str]
    links: list[Link]
    themes: list[ThemeHit]
    events: list[str]
    scope: str  # company | sector | market | offshore | unclassified
    equity_nexus: float
    entities: list[EntityHit] = field(default_factory=list)

    def relevance_for(self, symbol: str) -> Link | None:
        sym = symbol.upper()
        return next((l for l in self.links if l.symbol == sym), None)

    def as_payload(self) -> dict:
        return {
            "tickers": list(self.direct),
            "links": [l.as_dict() for l in self.links],
            "themes": [t.key for t in self.themes],
            "themeLabels": [t.label for t in self.themes],
            "events": list(self.events),
            "scope": self.scope,
            "equityNexus": round(self.equity_nexus, 2),
        }


@lru_cache(maxsize=4096)
def _boundary(alias: str) -> re.Pattern[str]:
    """Match an alias only on token boundaries.

    ``(?<!\\w)``/``(?!\\w)`` is used instead of ``\\b`` so aliases containing
    ``&`` or ``.`` behave predictably. This is what keeps ``itc`` from matching
    inside ``bitcoin``.
    """
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")


@lru_cache(maxsize=1)
def _alias_patterns() -> tuple[tuple[str, str, str, re.Pattern[str]], ...]:
    out: list[tuple[str, str, str, re.Pattern[str]]] = []
    for symbol, meta in UNIVERSE.items():
        for tier in ("strong", "medium", "weak"):
            for alias in meta.get(tier) or []:
                out.append((symbol, alias.lower().strip(), tier, _boundary(alias.lower().strip())))
    # Longest alias first so "adani ports" is seen before "adani".
    out.sort(key=lambda row: len(row[1]), reverse=True)
    return tuple(out)


@lru_cache(maxsize=1)
def _deny_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    out: list[tuple[str, re.Pattern[str]]] = []
    for symbol, meta in UNIVERSE.items():
        for phrase in meta.get("deny") or []:
            out.append((symbol, _boundary(phrase.lower().strip())))
    return tuple(out)


@lru_cache(maxsize=1)
def _theme_patterns() -> tuple[tuple[Theme, tuple[tuple[re.Pattern[str], int], ...]], ...]:
    out: list[tuple[Theme, tuple[tuple[re.Pattern[str], int], ...]]] = []
    for theme in THEMES:
        phrases = [(_boundary(p.strip()), 1) for p in theme.patterns]
        phrases += [(_boundary(p.strip()), -1) for p in theme.inverse_patterns]
        out.append((theme, tuple(phrases)))
    return tuple(out)


@lru_cache(maxsize=1)
def _related_index() -> dict[str, list[tuple[str, float, int, str]]]:
    """Undirected association edges, keyed by symbol."""
    idx: dict[str, list[tuple[str, float, int, str]]] = {}
    for a, b, weight, direction, note in RELATED:
        idx.setdefault(a, []).append((b, weight, direction, note))
        idx.setdefault(b, []).append((a, weight, direction, note))
    return idx


def normalise(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("’", "'").replace("`", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _sign(up: int, down: int) -> int:
    if up > down:
        return 1
    if down > up:
        return -1
    return 0


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """The clause containing a match.

    "Stock markets tumble as crude oil hits $100" has two subjects; without
    clause scoping the equity verb would set the crude direction.
    """
    lo = max((m.end() for m in _CLAUSE_SPLIT.finditer(text, 0, start)), default=0)
    next_break = _CLAUSE_SPLIT.search(text, end)
    hi = next_break.start() if next_break else len(text)
    # Fall back to a plain window only if clause detection collapsed onto the
    # match itself and so carries no direction information at all.
    if lo > start or hi < end:
        return max(0, start - _DIRECTION_WINDOW), min(len(text), end + _DIRECTION_WINDOW)
    return lo, hi


def _direction_near(text: str, start: int, end: int, theme: Theme) -> tuple[int, int]:
    """Direction of a theme, from words in the same clause as its match.

    Returns ``(own, generic)`` — the direction implied by the theme's own
    vocabulary, and the one implied by generic up/down words. Themes flagged
    ``requires_direction`` are only trusted when ``own`` is non-zero.
    """
    lo, hi = _clause_bounds(text, start, end)
    window = text[lo:hi]

    own = 0
    if theme.up or theme.down:
        own = _sign(
            sum(1 for w in theme.up if _boundary(w).search(window)),
            sum(1 for w in theme.down if _boundary(w).search(window)),
        )
    generic = _sign(
        sum(1 for w in DIRECTION_UP if _boundary(w).search(window)),
        sum(1 for w in DIRECTION_DOWN if _boundary(w).search(window)),
    )
    return own, generic


def resolve_entities(text: str) -> list[EntityHit]:
    """Companies named in the text, keeping the most specific name per span."""
    norm = normalise(text)
    denied: set[str] = set()
    for symbol, pattern in _deny_patterns():
        if pattern.search(norm):
            denied.add(symbol)

    hits: list[EntityHit] = []
    claimed: list[tuple[int, int]] = []

    def uncovered(match: re.Match[str]) -> bool:
        return not any(match.start() >= lo and match.end() <= hi for lo, hi in claimed)

    # Longest alias first, so "adani ports" claims its span before "adani" and
    # "itc hotels" before "itc".
    for symbol, alias, tier, pattern in _alias_patterns():
        if symbol in denied or any(h.symbol == symbol for h in hits):
            continue
        match = next((m for m in pattern.finditer(norm) if uncovered(m)), None)
        if not match:
            continue
        hits.append(EntityHit(symbol=symbol, alias=alias, tier=tier, start=match.start(), end=match.end()))
        claimed.append((match.start(), match.end()))
    return hits


def _score_theme(
    theme: Theme,
    phrases: tuple[tuple[re.Pattern[str], int], ...],
    text: str,
) -> tuple[int, int, int] | None:
    """Direction of a theme across every mention. Returns (own, generic, hits).

    Every occurrence votes, so "Gold edges up as Brent eases" resolves crude as
    falling even if a later clause uses an upward word.
    """
    if not text:
        return None
    own_up = own_down = gen_up = gen_down = hits = 0
    for pattern, flip in phrases:
        for match in pattern.finditer(text):
            hits += 1
            own, generic = _direction_near(text, match.start(), match.end(), theme)
            own *= flip
            generic *= flip
            own_up += own > 0
            own_down += own < 0
            gen_up += generic > 0
            gen_down += generic < 0
    if not hits:
        return None
    return _sign(own_up, own_down), _sign(gen_up, gen_down), hits


def detect_themes(text: str, summary: str = "") -> list[ThemeHit]:
    """Themes in a story.

    The title carries the subject of the story; a theme that only appears in the
    body is a supporting detail and is discounted so it cannot drive a sector
    link on its own.
    """
    norm_title = normalise(text)
    norm_body = normalise(summary)

    out: list[ThemeHit] = []
    for theme, phrases in _theme_patterns():
        in_title = _score_theme(theme, phrases, norm_title)
        in_body = _score_theme(theme, phrases, norm_body)
        scored = in_title or in_body
        if not scored:
            continue
        own, generic, hits = scored
        # Policy-style themes need their own vocabulary present: "cigarette
        # volumes grew" is an earnings line, not a tax change.
        if theme.requires_direction and not own:
            continue
        direction = own or generic
        confidence = min(1.0, 0.55 + 0.15 * (hits - 1) + (0.15 if direction else 0.0))
        if in_title is None:
            confidence *= _BODY_ONLY_DISCOUNT
        out.append(
            ThemeHit(key=theme.key, label=theme.label, scope=theme.scope, direction=direction, confidence=confidence)
        )
    out.sort(key=lambda t: -t.confidence)
    return out


def detect_events(title: str, body: str = "") -> list[str]:
    """Corporate events with a signed bias, most-specific first.

    Uses the India-tuned event map (earnings beat/miss, SEBI action, NPA,
    order wins, etc.) so a story like "SEBI imposes penalty" is never Neutral.
    """
    key, _bias = classify_event(title, body)
    return [key] if key else []


def _has_india_context(text: str) -> bool:
    return any(_boundary(w).search(text) for w in _INDIA_CONTEXT)


def _effect_label(direction: int) -> str:
    return "tailwind" if direction > 0 else "headwind" if direction < 0 else "direction unclear"


def _exposure_phrase(what: str, direction: int) -> str:
    if direction > 0:
        return f"{what} benefits (tailwind)"
    if direction < 0:
        return f"{what} comes under pressure (headwind)"
    return f"{what} is exposed (direction unclear)"


def analyze(headline: str, summary: str = "") -> Analysis:
    """Full analysis of one story.

    Pass the headline and body separately: entities count from either, but the
    story's *subject* comes from the headline, which is what decides how far a
    theme is allowed to propagate.
    """
    norm = normalise(f"{headline}. {summary}" if summary else headline)
    entities = resolve_entities(norm)
    themes = detect_themes(headline, summary)
    events = detect_events(headline, summary)

    corroborated = bool(events) or _has_india_context(norm)

    links: dict[str, Link] = {}

    def offer(link: Link) -> None:
        current = links.get(link.symbol)
        if current is None or link.relevance > current.relevance:
            links[link.symbol] = link

    # 1. Direct mentions. Only these ever count as company news.
    direct: list[str] = []
    for hit in entities:
        base = _TIER_BASE[hit.tier]
        if hit.tier == "weak" and corroborated:
            base = _WEAK_CORROBORATED
        if events:
            base = min(0.99, base + 0.04)

        sector = UNIVERSE.get(hit.symbol, {}).get("sector") or ""
        direction, theme_note = 0, ""
        for theme_hit in themes:
            theme = THEMES_BY_KEY[theme_hit.key]
            if theme.equity_nexus <= 0 or not theme_hit.direction:
                continue
            sensitivity = theme.symbol_sensitivity.get(hit.symbol) or theme.sector_sensitivity.get(sector, 0.0)
            if not sensitivity:
                continue
            direction = 1 if sensitivity * theme_hit.direction > 0 else -1
            theme_note = (
                f" · {theme.label.lower()} {theme_phrase(theme, theme_hit.direction)}"
                f" ({_effect_label(direction)})"
            )
            break

        event_note = f" ({', '.join(events[:2]).replace('_', ' ')})" if events else ""
        name = UNIVERSE.get(hit.symbol, {}).get("name", hit.symbol)
        reason = f"{name} named in the story{event_note}{theme_note}"
        if hit.tier == "weak" and not corroborated:
            reason += " · short-form name only, unconfirmed"
        offer(Link(symbol=hit.symbol, relevance=base, link_type="direct", reason=reason, direction=direction))
        if base >= DIRECT_MIN:
            direct.append(hit.symbol)

    # 2. Read-across to associated names.
    for symbol in list(direct):
        for peer, weight, edge_direction, note in _related_index().get(symbol, []):
            if peer in direct or peer not in UNIVERSE:
                continue
            offer(
                Link(
                    symbol=peer,
                    relevance=min(0.55, 0.35 * weight + 0.2),
                    link_type="peer",
                    reason=f"Read-across from {UNIVERSE[symbol]['name']} — {note}",
                    direction=edge_direction,
                )
            )

    # 3. Theme propagation. A theme with no equity read-through marks the story
    # as offshore, which damps whatever else it happens to mention.
    offshore = bool(themes) and not entities and any(THEMES_BY_KEY[t.key].equity_nexus <= 0 for t in themes)
    damping = _OFFSHORE_DAMPING if offshore else 1.0
    equity_nexus = min([THEMES_BY_KEY[t.key].equity_nexus for t in themes], default=1.0) if not entities else 1.0

    for theme_hit in themes:
        theme = THEMES_BY_KEY[theme_hit.key]
        if theme.equity_nexus <= 0:
            continue
        phrase = f"{theme.label} {theme_phrase(theme, theme_hit.direction)}"
        clarity = 1.0 if theme_hit.direction else _UNCLEAR_DISCOUNT

        # (symbol, sensitivity, what is exposed) — single names first so their
        # sharper sensitivity wins over the sector average.
        targets: list[tuple[str, float, str]] = [
            (symbol, sensitivity, UNIVERSE[symbol]["name"])
            for symbol, sensitivity in theme.symbol_sensitivity.items()
            if symbol in UNIVERSE
        ]
        for sector, sensitivity in theme.sector_sensitivity.items():
            for symbol in symbols_in_sector(sector):
                targets.append((symbol, sensitivity, SECTORS.get(sector, sector)))

        for symbol, sensitivity, exposure_label in targets:
            if symbol in direct:
                continue
            relevance = (
                min(theme.ceiling, abs(sensitivity) * theme_hit.confidence * theme.equity_nexus) * damping * clarity
            )
            if relevance < CONTEXT_MIN:
                continue
            direction = (1 if sensitivity * theme_hit.direction > 0 else -1) if theme_hit.direction else 0
            reason = f"{phrase} — {_exposure_phrase(exposure_label, direction)}"
            offer(Link(symbol=symbol, relevance=relevance, link_type="sector", reason=reason, direction=direction))

        if theme.index_sensitivity:
            relevance = min(theme.ceiling, abs(theme.index_sensitivity) * theme_hit.confidence) * damping * clarity
            if relevance >= CONTEXT_MIN:
                direction = (
                    (1 if theme.index_sensitivity * theme_hit.direction > 0 else -1) if theme_hit.direction else 0
                )
                offer(
                    Link(
                        symbol="NIFTY",
                        relevance=relevance,
                        link_type="index",
                        reason=f"{phrase} — {_exposure_phrase('broad market', direction)}",
                        direction=direction,
                    )
                )

    ordered = sorted(links.values(), key=lambda l: -l.relevance)

    if direct:
        scope = "company"
    elif offshore:
        scope = "offshore"
    elif any(l.link_type == "sector" or l.link_type == "peer" for l in ordered):
        scope = "sector"
    elif any(l.link_type == "index" for l in ordered):
        scope = "market"
    else:
        scope = "unclassified"

    return Analysis(
        direct=direct,
        links=ordered,
        themes=themes,
        events=events,
        scope=scope,
        equity_nexus=equity_nexus,
        entities=entities,
    )
