from __future__ import annotations

import time
from collections import defaultdict
from datetime import date, datetime, timezone

import pandas as pd

from .config import settings
from .events import event_evidence_multiplier, is_hype_event
from .fundamentals import get_fundamentals
from .levels import is_breakout_long, sr_conviction_factor, sr_position
from .linking import ACTIONABLE_CONTEXT_MIN, DIRECT_MIN
from .tape import (
    news_is_closed_session,
    prior_tape,
    tape_blocks_buy as tape_blocks_buy_fn,
    tape_conviction_factor,
    tape_supports_buy as tape_supports_buy_fn,
)
from .macro import build_macro_cards
from .model_sentiment import blend_company_sentiment, finbert_enabled, score_headlines
from .nifty_breadth import build_nifty_breadth
from .news import fetch_news, news_for_symbol
from .predictions import (
    accuracy_summary,
    public_outcome,
    resolve_due_predictions,
    schedule_for_news,
    upsert_open_call,
)
from .prices import (
    fetch_intraday_15m,
    load_ohlcv,
    move_since_news_pct,
    nearest_bar_time,
    observed_move_from_news,
    price_series,
)
from .quotes import get_index_quotes, get_quote
from .sentiment import bias_and_action, conviction_score, polarity, publish_signal, theme_conflict
from .session import (
    classify_published_at,
    is_cash_session_open,
    is_open_window,
    now_ist,
    session_snapshot,
    target_session_date,
)
from .taxonomy import SECTORS
from .thesis import apply_thesis_exit, thesis_health_for_stock
from .universe import UNIVERSE

INDEX_KEYS = {"NIFTY", "BANKNIFTY", "SENSEX", "VIX", "FINNIFTY", "MIDCAP"}

# Sector/peer stories need a higher bar than company news to appear on the board.
# Sector/peer must clear this to appear on the dashboard (stricter than CONTEXT_MIN).
SECTOR_BOARD_MIN = 0.52


def _volume_stats(symbol: str, last_volume: float | None) -> tuple[float, float]:
    """Return (volume_lakhs, avg_volume_lakhs) from parquet + last quote volume."""
    df = load_ohlcv(symbol)
    avg = 0.0
    if not df.empty and "Volume" in df.columns:
        avg = float(df["Volume"].tail(20).mean() or 0.0) / 100_000.0
    vol = (last_volume / 100_000.0) if last_volume else 0.0
    if vol <= 0 and not df.empty and "Volume" in df.columns:
        vol = float(df["Volume"].iloc[-1] or 0.0) / 100_000.0
    return round(vol, 1), round(avg, 1)


def _day_year_ranges(symbol: str, ltp: float) -> tuple[list[float], list[float]] | tuple[None, None]:
    df = load_ohlcv(symbol)
    if df.empty or "Low" not in df.columns or "High" not in df.columns:
        return None, None
    day = df.tail(1)
    year = df.tail(252)
    day_range = [round(float(day["Low"].iloc[0]), 2), round(float(day["High"].iloc[0]), 2)]
    year_range = [round(float(year["Low"].min()), 2), round(float(year["High"].max()), 2)]
    return day_range, year_range


def _event_key(item: dict) -> str | None:
    key = item.get("event")
    if key:
        return str(key)
    events = item.get("events") or []
    return str(events[0]) if events else None


def _session_evidence_multiplier(item: dict) -> float:
    """Overnight / next-open news outweighed live prints in the backtest."""
    published = item.get("publishedAt")
    if not published:
        return 1.0
    phase = classify_published_at(published)
    if phase == "during_market":
        return 0.7
    if phase in {"after_close", "closed_day", "before_open"}:
        return 1.25
    return 1.0


def _evidence_weight(item: dict) -> float:
    impact = float(item.get("impact") or 1) / 10.0
    relevance = float(item.get("relevance", 1.0))
    credibility = float(item.get("credibility") or 0.6)
    mins = item.get("minutesAgo") or 9999
    freshness = 1.0 if mins <= 60 else 0.85 if mins <= 360 else 0.65 if mins <= 1440 else 0.4
    base = impact * relevance * credibility * freshness
    link_type = item.get("linkType") or "direct"
    # Indirect links count less even when they clear the junk floor.
    if link_type in {"peer", "sector"}:
        base *= 0.55
    elif link_type == "index":
        base *= 0.35
    return base * event_evidence_multiplier(_event_key(item)) * _session_evidence_multiplier(item)


def _usable_evidence(related_news: list[dict]) -> list[dict]:
    """Drop junk peer/sector/index spray before conviction and action scoring."""
    usable: list[dict] = []
    for n in related_news:
        link_type = n.get("linkType") or "direct"
        relevance = float(n.get("relevance") or 0)
        if link_type == "direct":
            usable.append(n)
        elif link_type in {"peer", "sector"} and relevance >= ACTIONABLE_CONTEXT_MIN:
            usable.append(n)
        # index / weak context: ignored for stock calls
    return usable


def _anchor_news(related_news: list[dict], sentiment: str) -> dict | None:
    """Pick the story that best explains the stock call.

    Important: headline *tone* ("oil prices fall" = Negative text) is not the
    same as stock *direction* (oil fall = Positive for autos). Prefer stories
    whose linked direction agrees with the aggregate call.
    """
    if not related_news:
        return None
    want = 1 if sentiment == "Positive" else -1 if sentiment == "Negative" else 0

    def rank(n: dict) -> tuple:
        return (_evidence_weight(n), -(n.get("minutesAgo") or 0))

    # Prefer non-hype when anything else is available (Phase B).
    preferred = [n for n in related_news if not is_hype_event(_event_key(n))]
    pool = preferred or list(related_news)

    if want != 0:
        aligned = [n for n in pool if _story_direction(n) == want]
        if aligned:
            return max(aligned, key=rank)
        # Neutral-direction stories next (explain context without flipping the call).
        neutral = [n for n in pool if _story_direction(n) == 0]
        if neutral:
            return max(neutral, key=rank)

    return max(pool, key=rank)


def _weighted_factor(related_news: list[dict], per_item) -> float:
    """Evidence-weighted average of a per-story multiplier (before that multiplier)."""
    num = 0.0
    den = 0.0
    for n in related_news:
        # Reconstruct pre-multiplier weight so the average isn't circular.
        impact = float(n.get("impact") or 1) / 10.0
        relevance = float(n.get("relevance", 1.0))
        credibility = float(n.get("credibility") or 0.6)
        mins = n.get("minutesAgo") or 9999
        freshness = 1.0 if mins <= 60 else 0.85 if mins <= 360 else 0.65 if mins <= 1440 else 0.4
        w = impact * relevance * credibility * freshness
        if w <= 0:
            continue
        factor = float(per_item(n))
        num += factor * w
        den += w
    return (num / den) if den else 1.0


def _story_direction(item: dict) -> int:
    expected = int(item.get("expectedDirection") or 0)
    polarity_dir = 1 if item.get("sentiment") == "Positive" else -1 if item.get("sentiment") == "Negative" else 0
    if item.get("linkType") in {"peer", "sector", "index"} and expected:
        return expected if polarity_dir == 0 else expected * abs(polarity_dir)
    return polarity_dir


def _plain_direction(sentiment: str, expected: int) -> str:
    if expected > 0 or (expected == 0 and sentiment == "Positive"):
        return "up"
    if expected < 0 or (expected == 0 and sentiment == "Negative"):
        return "down"
    return "unclear"


def _plain_sentiment(sentiment: str) -> str:
    if sentiment == "Positive":
        return "Positive"
    if sentiment == "Negative":
        return "Negative"
    return "Unclear"


def aggregate_stock_sentiment(
    symbol: str,
    related_news: list[dict],
    *,
    change_pct: float,
    ltp: float,
    volume_ratio: float = 0.0,
) -> dict:
    related_news = _usable_evidence(related_news)
    if not related_news:
        return {
            "sentiment": "Neutral",
            "impact": 1,
            "bias": "mixed",
            "action": "watch",
            "actionNote": None,
            "move": None,
            "conviction": 0,
            "confidence": "low",
            "convictionDrivers": ["no linked stories"],
            "expectedDirection": 0,
            "scorer": "rules",
        }

    score = 0.0
    weight = 0.0
    impact_weighted = 0.0
    direct_weight = 0.0
    for n in related_news:
        w = _evidence_weight(n)
        score += _story_direction(n) * w
        weight += w
        impact_weighted += float(n.get("impact") or 1) * w
        if n.get("linkType", "direct") == "direct":
            direct_weight += w

    net = (score / weight) if weight else 0.0
    sent = "Positive" if net > 0.2 else "Negative" if net < -0.2 else "Neutral"
    avg_impact = int(round(impact_weighted / weight)) if weight else 1
    avg_impact = max(1, min(10, avg_impact))

    anchor = _anchor_news(related_news, sent)
    scorer = "rules"
    if anchor and anchor.get("linkType") == "direct" and finbert_enabled():
        _, rules_score = polarity(
            f"{anchor.get('headline', '')}. {anchor.get('summary', '')}",
            title=anchor.get("headline"),
            body=anchor.get("summary"),
        )
        sent, _blended, scorer = blend_company_sentiment(sent, rules_score, anchor.get("headline") or "")

    move = move_since_news_pct(symbol, anchor, ltp) if anchor else None

    # Phase C — support/resistance structure at the current price.
    sr_pos = sr_position(symbol, ltp)
    breakout_long = is_breakout_long(sent, sr_pos)
    sr_factor = sr_conviction_factor(sent, sr_pos)

    # Phase D — prior-session volume/VWAP tape. Applied only when news evidence
    # is overnight-dominated; live/open-session leaves tape gates neutral.
    closed_news = news_is_closed_session(related_news)
    tape = prior_tape(symbol) if closed_news else {}
    tape_factor = tape_conviction_factor(sent, tape, closed_session=closed_news)
    blocks_buy = tape_blocks_buy_fn(sent, tape, closed_session=closed_news)
    supports_buy = tape_supports_buy_fn(sent, tape, closed_session=closed_news)

    bias, action, note = bias_and_action(
        sent,
        avg_impact,
        change_pct=change_pct,
        move_since_news_pct=move,
        breakout_long=breakout_long,
        structure=sr_pos,
        tape_blocks_buy=blocks_buy,
        tape_supports_buy=supports_buy,
    )

    price_agrees: bool | None = None
    if sent != "Neutral" and abs(change_pct) >= 0.2:
        price_agrees = (change_pct > 0) == (sent == "Positive")

    session_factor = _weighted_factor(related_news, _session_evidence_multiplier)
    event_factor = _weighted_factor(
        related_news, lambda n: event_evidence_multiplier(_event_key(n))
    )
    conviction, confidence, drivers = conviction_score(
        evidence_weight=weight,
        agreement=abs(net),
        source_count=len({n.get("source") for n in related_news}),
        direct_share=(direct_weight / weight) if weight else 0.0,
        volume_ratio=volume_ratio,
        price_agrees=price_agrees,
        session_factor=session_factor,
        event_factor=event_factor,
        sr_factor=sr_factor,
        tape_factor=tape_factor,
    )

    conflict = theme_conflict(related_news, story_direction=_story_direction)
    if conflict:
        drivers = [*(drivers or []), "opposing themes — wait for clarity"][:5]
        if note:
            note = f"{note} · opposing themes"
        else:
            note = "opposing themes — wait for clarity"
    action = publish_signal(bias, action, conviction, conflict=conflict)

    expected = 1 if sent == "Positive" else -1 if sent == "Negative" else 0
    if anchor:
        story_dir = _story_direction(anchor)
        # Anchor may refine an unclear call, but must not flip a bullish/bearish
        # aggregate (that created Bias/Buy) into the opposite overnight thesis.
        if story_dir and (expected == 0 or story_dir == expected):
            expected = story_dir

    return {
        "sentiment": sent,
        "impact": avg_impact,
        "bias": bias,
        "action": action,
        "actionNote": note,
        "move": move,
        "conviction": conviction,
        "confidence": confidence,
        "convictionDrivers": drivers,
        "expectedDirection": expected,
        "scorer": scorer,
        "anchor": anchor,
        "structure": sr_pos or None,
        "breakoutLong": breakout_long,
        "tape": tape or None,
        "closedSessionNews": closed_news,
        "tapeFactor": tape_factor,
        "themeConflict": conflict,
        "signalTier": (
            "strong" if conviction >= 60 else "medium" if conviction >= 40 else "weak"
        ),
    }


def build_stock_row(symbol: str, related_news: list[dict] | None = None) -> dict | None:
    sym = symbol.upper()
    q = get_quote(sym)
    if not q:
        return None

    fund = get_fundamentals(sym)
    meta = UNIVERSE.get(sym, {})
    related = related_news if related_news is not None else news_for_symbol(sym)
    vol, avg = _volume_stats(sym, q.volume)
    read = aggregate_stock_sentiment(
        sym,
        related,
        change_pct=q.change_pct,
        ltp=q.ltp,
        volume_ratio=(vol / avg) if avg else 0.0,
    )
    day_range, year_range = _day_year_ranges(sym, q.ltp)
    company_news = [n for n in related if n.get("linkType", "direct") == "direct"]
    latest_mins = min((n.get("minutesAgo", 9999) for n in company_news or related), default=9999)
    anchor = read.get("anchor")

    return {
        "symbol": sym,
        "name": fund.get("name") or meta.get("name") or sym,
        "index": meta.get("index") or "NIFTY",
        "ltp": q.ltp,
        "changePct": q.change_pct,
        "volume": vol,
        "avgVolume": avg,
        "newsCount": len(company_news),
        "contextCount": len(related) - len(company_news),
        "latestNewsMins": latest_mins,
        "sentiment": read["sentiment"],
        "plainSentiment": _plain_sentiment(read["sentiment"]),
        "impact": read["impact"],
        "bias": read["bias"],
        "action": read["action"],
        "actionNote": read["actionNote"],
        "moveSinceNewsPct": read["move"],
        "conviction": read["conviction"],
        "confidence": read["confidence"],
        "convictionDrivers": read["convictionDrivers"],
        "expectedDirection": read["expectedDirection"],
        "direction": _plain_direction(read["sentiment"], read["expectedDirection"]),
        "scorer": read.get("scorer") or "rules",
        "anchorHeadline": (anchor or {}).get("headline"),
        "anchorReason": (anchor or {}).get("linkReason"),
        "anchorId": (anchor or {}).get("id"),
        "anchorPublishedAt": (anchor or {}).get("publishedAt"),
        "anchorMinutesAgo": (anchor or {}).get("minutesAgo"),
        "anchorLinkType": (anchor or {}).get("linkType"),
        "anchorCredibility": (anchor or {}).get("credibility"),
        "anchorRelevance": (anchor or {}).get("relevance"),
        "themeConflict": bool(read.get("themeConflict")),
        "signalTier": read.get("signalTier")
        or (
            "strong"
            if int(read.get("conviction") or 0) >= 60
            else "medium"
            if int(read.get("conviction") or 0) >= 40
            else "weak"
        ),
        "sectorGroup": SECTORS.get(meta.get("sector") or "", meta.get("sector") or "—"),
        "sector": fund.get("sector") or "—",
        "marketCap": fund.get("marketCap") or "—",
        "dayRange": day_range or [q.ltp, q.ltp],
        "yearRange": year_range or [q.ltp, q.ltp],
        "peRatio": fund.get("peRatio"),
        "about": fund.get("about") or "",
        "quoteSource": q.source,
        "structure": read.get("structure"),
        "breakoutLong": read.get("breakoutLong"),
        "tape": read.get("tape"),
        "nearestResistance": (read.get("structure") or {}).get("nearestResistance"),
        "nearestSupport": (read.get("structure") or {}).get("nearestSupport"),
        "distResistPct": (read.get("structure") or {}).get("distResistPct"),
        "distSupportPct": (read.get("structure") or {}).get("distSupportPct"),
        "sessionVwap": (read.get("tape") or {}).get("sessionVwap"),
        "distVwapPct": (read.get("tape") or {}).get("distVwapPct"),
        "_anchor": anchor,
        "_related": related,
    }


def _dedupe_context(items: list[dict], per_theme: int = 2) -> list[dict]:
    items = sorted(items, key=lambda n: (-n.get("relevance", 0.0), n.get("minutesAgo", 9999)))
    seen: dict[tuple, int] = {}
    out: list[dict] = []
    for n in items:
        theme = (n.get("themes") or ["_"])[0]
        key = (n.get("linkType"), theme)
        if seen.get(key, 0) >= per_theme:
            continue
        seen[key] = seen.get(key, 0) + 1
        out.append(n)
    return out


def build_morning_brief(indices: list[dict], news: list[dict], macro: list[dict]) -> dict:
    nifty = next((i for i in indices if i["key"] == "NIFTY"), None)
    bank = next((i for i in indices if i["key"] == "BANKNIFTY"), None)
    parts = []
    if nifty:
        parts.append(f"Nifty {nifty['ltp']:,.2f} ({nifty['changePct']:+.2f}%)")
    if bank:
        parts.append(f"Bank Nifty {bank['ltp']:,.2f} ({bank['changePct']:+.2f}%)")
    headline = "; ".join(parts) if parts else "Indian market snapshot"
    if news:
        headline += f" · Top story: {news[0]['headline'][:110]}"

    bullets: list[str] = []
    for i in indices[:4]:
        bullets.append(f"{i['name']}: {i['ltp']:,.2f} ({i['changePct']:+.2f}%) · {i.get('source', 'delayed')}")
    for m in macro[:2]:
        bullets.append(m["title"])
    for n in news[:3]:
        tick = ",".join(n["tickers"][:3]) if n["tickers"] else "market"
        bullets.append(f"[{n['sentiment']} · impact {n['impact']}] {n['headline']} ({tick})")

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "headline": headline,
        "bullets": bullets[:8],
    }


def _is_board_worthy(link: dict, news_item: dict) -> bool:
    """Credibility gate — no arbitrary stock-count limit."""
    if link["symbol"] in INDEX_KEYS:
        return False
    relevance = float(link.get("relevance") or 0)
    credibility = float(news_item.get("credibility") or 0.6)
    min_rel = float(settings.board_min_relevance)
    min_cred = float(settings.board_min_credibility)
    if credibility < min_cred:
        return False
    link_type = link.get("type") or "direct"
    if link_type == "direct":
        return relevance >= min(DIRECT_MIN, min_rel)
    if link_type in {"sector", "peer"}:
        return relevance >= max(SECTOR_BOARD_MIN, min_rel)
    return False


def _assign_bucket(
    *,
    session_phase: str,
    observed_move_pct: float | None,
    expected_direction: int,
    market_open: bool,
    target_session: date | None = None,
    today: date | None = None,
) -> str:
    threshold = float(settings.reaction_threshold_pct)
    moved = observed_move_pct is not None and abs(observed_move_pct) >= threshold
    agrees = False
    if moved and expected_direction != 0 and observed_move_pct is not None:
        agrees = (observed_move_pct > 0) == (expected_direction > 0)

    if moved and (agrees or expected_direction == 0):
        return "already_reacted"
    if session_phase == "during_market" and market_open:
        return "live_session"
    if session_phase in {"after_close", "closed_day", "before_open"}:
        # Overnight / pre-open calls for today's session move to live once cash opens.
        if market_open and target_session is not None and today is not None and target_session == today:
            return "live_session"
        return "next_session"
    # During market but session already closed in snapshot edge cases
    if session_phase == "during_market":
        return "live_session" if market_open else "already_reacted"
    return "next_session"


def _enrich_session_row(row: dict) -> dict | None:
    """Attach session phase, observed move, bucket, and prediction outcome."""
    anchor = row.pop("_anchor", None)
    related = row.pop("_related", None) or []
    if not anchor:
        # Fall back to strongest related story so sector-only promotions work.
        if not related:
            return None
        anchor = max(related, key=lambda n: (_evidence_weight(n), -(n.get("minutesAgo") or 0)))

    published = anchor.get("publishedAt")
    phase = classify_published_at(published)
    move_info = observed_move_from_news(
        row["symbol"],
        anchor,
        row["ltp"],
        session_phase=phase,
    )
    observed = move_info["observedMovePct"]
    # Prefer session-aware move on the card.
    if observed is not None:
        row["moveSinceNewsPct"] = observed

    market_open = is_cash_session_open()
    expected_direction = int(row.get("expectedDirection") or 0)
    target = target_session_date(published)
    bucket = _assign_bucket(
        session_phase=phase,
        observed_move_pct=observed,
        expected_direction=expected_direction,
        market_open=market_open,
        target_session=target,
        today=now_ist().date(),
    )

    reason = anchor.get("linkReason") or ""
    if not reason and anchor.get("headline"):
        reason = anchor["headline"]

    # Per-headline audit trail (append-only).
    pred = schedule_for_news(
        news_item=anchor,
        symbol=row["symbol"],
        expected_direction=expected_direction,
        bucket=bucket,
        baseline_price=move_info["baselinePrice"],
        baseline_label=move_info["baselineLabel"],
        sentiment=row["sentiment"],
        conviction=int(row.get("conviction") or 0),
        confidence=row.get("confidence") or "low",
        reason=reason,
        scorer=row.get("scorer") or "rules",
    )

    # Living overnight call for the target session: revise until 09:15, then freeze.
    open_call = None
    if phase in {"after_close", "closed_day", "before_open"}:
        open_call = upsert_open_call(
            {
                "symbol": row["symbol"],
                "targetSession": target.isoformat(),
                "expectedDirection": expected_direction,
                "sentiment": row.get("sentiment"),
                "conviction": int(row.get("conviction") or 0),
                "confidence": row.get("confidence") or "low",
                "headline": anchor.get("headline"),
                "reason": reason,
                "newsId": anchor.get("id"),
                "linkType": anchor.get("linkType"),
                "baselinePrice": move_info.get("baselinePrice"),
                "baselineLabel": move_info.get("baselineLabel"),
                "bucket": bucket,
                "sessionPhase": phase,
                "scorer": row.get("scorer") or "rules",
            }
        )
        # After the open bell, overnight cards must score the frozen call — not
        # a later intraday rewrite of bias from fresh cash-session headlines.
        if open_call and open_call.get("frozen_at") and open_call.get("expected_direction") is not None:
            expected_direction = int(open_call["expected_direction"] or 0)
            if open_call.get("conviction") is not None:
                row["conviction"] = int(open_call["conviction"])
            if open_call.get("confidence"):
                row["confidence"] = open_call["confidence"]
            if open_call.get("sentiment"):
                row["sentiment"] = open_call["sentiment"]
            if open_call.get("headline"):
                # Keep the locked overnight anchor visible on the card.
                anchor = {
                    **anchor,
                    "headline": open_call.get("headline") or anchor.get("headline"),
                    "id": open_call.get("news_id") or anchor.get("id"),
                    "linkType": open_call.get("link_type") or anchor.get("linkType"),
                }
                reason = open_call.get("reason") or reason
            if open_call.get("baseline_price"):
                move_info = {
                    **move_info,
                    "baselinePrice": open_call.get("baseline_price"),
                    "baselineLabel": open_call.get("baseline_label") or move_info.get("baselineLabel"),
                }

    row.update(
        {
            "bucket": bucket,
            # Board direction follows the living call before open, frozen call after.
            "expectedDirection": expected_direction,
            "direction": "up" if expected_direction > 0 else "down" if expected_direction < 0 else "unclear",
            "plainSentiment": "Positive" if expected_direction > 0 else "Negative" if expected_direction < 0 else "Unclear",
            "sessionPhase": phase,
            "targetSession": target.isoformat(),
            "baselinePrice": move_info["baselinePrice"],
            "baselineLabel": move_info["baselineLabel"],
            "observedMovePct": observed,
            "anchorHeadline": anchor.get("headline"),
            "anchorReason": reason,
            "anchorId": anchor.get("id"),
            "anchorPublishedAt": published,
            "anchorMinutesAgo": anchor.get("minutesAgo"),
            "anchorLinkType": anchor.get("linkType"),
            "outcome": public_outcome(open_call or pred),
            "openCallLocked": bool(open_call and open_call.get("frozen_at")),
            "openCallRevisedAt": (open_call or {}).get("revised_at"),
            "openCallFrozenAt": (open_call or {}).get("frozen_at"),
        }
    )

    # Overnight thesis live health during cash hours (gap / hold / fade).
    # Open-window uses a short 15m TTL; stock detail may force a re-pull.
    if phase in {"after_close", "closed_day", "before_open"} and expected_direction != 0:
        health = thesis_health_for_stock(
            row["symbol"],
            expected_direction=expected_direction,
            baseline_price=move_info.get("baselinePrice"),
            session_phase=phase,
            current_ltp=row.get("ltp"),
        )
        if health:
            row["thesisHealth"] = health.get("thesisHealth")
            row["thesisLabel"] = health.get("label")
            row["thesisGapState"] = health.get("gapState")
            row["thesisHoldState"] = health.get("holdState")
            row["thesisOpenMovePct"] = health.get("openMovePct")
            row["thesisPlus15MovePct"] = health.get("plus15MovePct")
            row["thesisPlus30MovePct"] = health.get("plus30MovePct")
            row["thesisLastMovePct"] = health.get("lastMovePct")
            row["thesisPeakFavPct"] = health.get("peakFavPct")
            row["thesisGivebackFrac"] = health.get("givebackFrac")
            row["thesisTrailDropPct"] = health.get("trailDropPct")
            row["thesisExitTrigger"] = health.get("exitTrigger")
            row["thesisSessionHighPct"] = health.get("sessionHighPct")
            row["thesisSessionLowPct"] = health.get("sessionLowPct")
            # Hold/exit: don't leave Buy long/short lit once path kills the thesis.
            new_action, new_note = apply_thesis_exit(
                row.get("action") or "watch",
                health.get("thesisHealth"),
                action_note=row.get("actionNote"),
            )
            if new_action != row.get("action"):
                row["action"] = new_action
                row["actionNote"] = new_note

    return row


def build_dashboard() -> dict:
    news = fetch_news()
    resolve_due_predictions()
    if finbert_enabled():
        score_headlines(
            [
                n.get("headline") or ""
                for n in news
                if any(link.get("type") == "direct" for link in (n.get("links") or []))
            ]
        )

    by_sym: dict[str, list[dict]] = defaultdict(list)
    promoted: set[str] = set()
    for n in news:
        if n["minutesAgo"] > 60 * 36:
            continue
        for link in n.get("links") or []:
            if not _is_board_worthy(link, n):
                continue
            item = dict(n)
            item["relevance"] = link["relevance"]
            item["linkType"] = link["type"]
            item["linkReason"] = link["reason"]
            item["expectedDirection"] = link["direction"]
            by_sym[link["symbol"]].append(item)
            # Direct always promotes; strong sector/peer also promote.
            if link["type"] == "direct" or float(link["relevance"]) >= SECTOR_BOARD_MIN:
                promoted.add(link["symbol"])

    stocks: list[dict] = []
    for sym in promoted:
        # Score conviction on the same evidence set as the stock page.
        # Board-worthiness only decides who appears and which story is shown.
        row = build_stock_row(sym, related_news=news_for_symbol(sym, news))
        if not row:
            continue
        board_items = by_sym.get(sym) or []
        if board_items:
            preferred = [n for n in board_items if not is_hype_event(_event_key(n))]
            pool = preferred or board_items
            want = int(row.get("expectedDirection") or 0)
            if want != 0:
                aligned = [n for n in pool if int(n.get("expectedDirection") or 0) == want]
                if aligned:
                    pool = aligned
            row["_anchor"] = max(
                pool,
                key=lambda n: (_evidence_weight(n), -(n.get("minutesAgo") or 0)),
            )
        enriched = _enrich_session_row(row)
        if enriched:
            stocks.append(enriched)

    buckets = {
        "next_session": [],
        "live_session": [],
        "already_reacted": [],
    }
    for s in stocks:
        buckets.setdefault(s["bucket"], []).append(s)

    def _sort_key(s: dict, *, bucket: str):
        # Live board: strongest conviction first while calls validate.
        if bucket == "live_session":
            return (
                -int(s.get("conviction") or 0),
                -int(s.get("impact") or 0),
                s.get("latestNewsMins", 9999),
            )
        # Next-session overnight board: strongest conviction first (high → low).
        if bucket == "next_session":
            return (
                -int(s.get("conviction") or 0),
                -int(s.get("impact") or 0),
                s.get("latestNewsMins", 9999),
            )
        return (s.get("latestNewsMins", 9999), -s.get("conviction", 0), -s.get("impact", 0))

    for key in buckets:
        buckets[key].sort(key=lambda s, k=key: _sort_key(s, bucket=k))

    # Flat list kept for back-compat (uncapped).
    all_stocks = [*buckets["next_session"], *buckets["live_session"], *buckets["already_reacted"]]

    # Indices / breadth / macro are independent — fetch concurrently.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_idx = pool.submit(get_index_quotes)
        fut_breadth = pool.submit(build_nifty_breadth)
        fut_macro = pool.submit(build_macro_cards, 3)
        indices = fut_idx.result()
        nifty_breadth = fut_breadth.result()
        macro = fut_macro.result()

    def market_relevant(n: dict) -> bool:
        return n.get("scope") != "unclassified"

    def take(pred, limit: int, *, relevant_only: bool = True) -> list[dict]:
        pool = [n for n in news if pred(n) and (market_relevant(n) or not relevant_only)]
        pool.sort(key=lambda n: (n["minutesAgo"], -n["impact"]))
        return pool[:limit]

    mixed: list[dict] = []
    seen_ids: set[str] = set()

    def add_all(chunk: list[dict]) -> None:
        for n in chunk:
            if n["id"] in seen_ids:
                continue
            seen_ids.add(n["id"])
            mixed.append(n)

    add_all(take(lambda n: n.get("kind") != "tweet" and ("FII" in n.get("tags", []) or "DII" in n.get("tags", [])), 3))
    add_all(take(lambda n: n.get("kind") != "tweet" and n.get("scope") == "company", 5))
    add_all(take(lambda n: n.get("kind") != "tweet", 6))
    add_all(take(lambda n: n.get("kind") == "tweet", 4))
    add_all(take(lambda _n: True, 12, relevant_only=False))
    feed = mixed[:12]
    feed.sort(key=lambda n: (n["minutesAgo"], -n["impact"]))
    brief = build_morning_brief(indices, feed, macro)
    accuracy = accuracy_summary()

    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "session": session_snapshot(),
        "indices": indices,
        "niftyBreadth": nifty_breadth,
        "buckets": buckets,
        "topStocks": all_stocks,
        "morningBrief": brief,
        "news": feed[:12],
        "macro": macro,
        "accuracy": accuracy,
    }


_DASH_CACHE: dict = {"ts": 0.0, "payload": None, "building": False}
_DASH_LOCK = __import__("threading").Lock()


def _store_dashboard(payload: dict) -> dict:
    payload = dict(payload)
    payload["cached"] = False
    payload["stale"] = False
    _DASH_CACHE["ts"] = time.time()
    _DASH_CACHE["payload"] = payload
    return payload


def _rebuild_dashboard_bg(*, force: bool = False) -> None:
    """Background refresh — never blocks the request that served stale data."""
    with _DASH_LOCK:
        if _DASH_CACHE["building"]:
            return
        _DASH_CACHE["building"] = True
    try:
        payload = build_dashboard()
        _store_dashboard(payload)
    except Exception:  # noqa: BLE001
        # Keep last good cache; next request can retry.
        pass
    finally:
        with _DASH_LOCK:
            _DASH_CACHE["building"] = False


def get_dashboard(*, force: bool = False) -> dict:
    """Serve last board instantly when possible; refresh Yahoo/news in background.

    Free hosting + Yahoo make a full rebuild 30–120s. Stale-while-revalidate
    keeps the phone UI fast after the first successful build.
    """
    from .session import is_open_window

    now = time.time()
    ttl = float(settings.dashboard_ttl_open_s if is_open_window() else settings.dashboard_ttl_s)
    cached = _DASH_CACHE["payload"]
    age = now - float(_DASH_CACHE["ts"]) if cached is not None else None

    if cached is not None:
        out = dict(cached)
        out["cached"] = True
        out["cacheAgeS"] = round(float(age or 0), 1)
        stale = bool(age is not None and age >= ttl)
        out["stale"] = stale or force
        # Soft path: always return cache; refresh when stale or force.
        if force or stale:
            t = __import__("threading").Thread(
                target=_rebuild_dashboard_bg,
                kwargs={"force": force},
                daemon=True,
                name="saint-dash-rebuild",
            )
            t.start()
        return out

    # Cold start — nothing to serve yet; must build synchronously once.
    with _DASH_LOCK:
        # Another request may have started building.
        if _DASH_CACHE["building"] and _DASH_CACHE["payload"] is None:
            pass
        _DASH_CACHE["building"] = True
    try:
        # Re-check: background may have finished.
        if _DASH_CACHE["payload"] is not None and not force:
            out = dict(_DASH_CACHE["payload"])
            out["cached"] = True
            out["stale"] = False
            return out
        payload = build_dashboard()
        return _store_dashboard(payload)
    finally:
        with _DASH_LOCK:
            _DASH_CACHE["building"] = False


def build_stock_detail(symbol: str) -> dict | None:
    sym = symbol.upper()
    related = news_for_symbol(sym)
    row = build_stock_row(sym, related_news=related)
    if not row:
        return None

    fetch_intraday_15m(sym, force=is_open_window())
    enriched_row = _enrich_session_row(dict(row))
    if enriched_row:
        row = enriched_row
    else:
        row.pop("_anchor", None)
        row.pop("_related", None)

    enriched: list[dict] = []
    for n in related[:40]:
        item = dict(n)
        when = None
        if n.get("publishedAt"):
            try:
                when = pd.Timestamp(n["publishedAt"])
            except Exception:
                when = None
        if when is None and n.get("minutesAgo") is not None and n["minutesAgo"] < 9000:
            when = pd.Timestamp.utcnow() - pd.Timedelta(minutes=int(n["minutesAgo"]))
        item["chartDate"] = nearest_bar_time(sym, when) if when is not None else None
        enriched.append(item)

    company = [n for n in enriched if n.get("linkType") == "direct"]
    company.sort(key=lambda n: (n["minutesAgo"], -n["impact"]))
    context = _dedupe_context([n for n in enriched if n.get("linkType") != "direct"])

    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "session": session_snapshot(),
        "stock": row,
        "news": company,
        "context": context[:12],
    }


def build_prices(symbol: str, range_key: str = "1M") -> dict:
    points, interval = price_series(symbol, range_key=range_key)
    return {
        "symbol": symbol.upper(),
        "range": range_key.upper(),
        "interval": interval,
        "source": "yahoo+parquet" if interval.startswith("15m") else ("parquet" if points else "none"),
        "points": points,
    }
