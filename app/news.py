"""India market RSS ingest (Economic Times, Moneycontrol, Mint, FII/DII, etc.)."""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Lock
from urllib.parse import urlparse

import feedparser
import httpx

from .linking import CONTEXT_MIN, analyze
from .sentiment import impact_score, polarity
from .tweets import fetch_tweets

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

FEEDS = [
    ("Economic Times Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", ("news",)),
    ("Economic Times", "https://economictimes.indiatimes.com/rssfeedsdefault.cms", ("news",)),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/MCtopnews.xml", ("news",)),
    ("LiveMint Markets", "https://www.livemint.com/rss/markets", ("news",)),
    ("The Hindu Markets", "https://www.thehindu.com/business/markets/feeder/default.rss", ("news",)),
    ("NDTV Profit", "https://feeds.feedburner.com/ndtvprofit-latest", ("news",)),
    (
        "FII/DII Flows",
        "https://news.google.com/rss/search?q=FII+OR+DII+OR+%22foreign+institutional%22+OR+%22domestic+institutional%22+India+stocks+when:2d&hl=en-IN&gl=IN&ceid=IN:en",
        ("news", "flows", "FII", "DII"),
    ),
]

_cache: tuple[float, list[dict]] | None = None
_lock = Lock()
# Backend re-fetches live feeds at most every 5 minutes
TTL_S = 300


def _minutes_ago(entry: dict) -> int:
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            try:
                dt = datetime(*st[:6], tzinfo=timezone.utc)
                return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 60))
            except Exception:
                pass
    for key in ("published", "updated"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 60))
        except Exception:
            continue
    return 9999


def _published_iso(entry: dict) -> str | None:
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            try:
                return datetime(*st[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    for key in ("published", "updated"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            continue
    return None


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _clean_html(summary: str) -> str:
    summary = re.sub(r"<[^>]+>", " ", summary or "")
    return re.sub(r"\s+", " ", summary).strip()


def _flow_tags(text: str, base: tuple[str, ...]) -> list[str]:
    tags = list(base)
    low = text.lower()
    if "fii" in low or "foreign institutional" in low or "foreign portfolio" in low or "fpi" in low:
        if "FII" not in tags:
            tags.append("FII")
        if "flows" not in tags:
            tags.append("flows")
    if "dii" in low or "domestic institutional" in low:
        if "DII" not in tags:
            tags.append("DII")
        if "flows" not in tags:
            tags.append("flows")
    return tags


def _fetch_feed(name: str, url: str, base_tags: tuple[str, ...]) -> list[dict]:
    out: list[dict] = []
    try:
        with httpx.Client(
            headers={"User-Agent": UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"},
            follow_redirects=True,
            timeout=25.0,
        ) as client:
            r = client.get(url)
            r.raise_for_status()
            parsed = feedparser.parse(r.text)
    except Exception:
        return out

    for e in parsed.entries:
        title = (e.get("title") or "").strip()
        if not title:
            continue
        summary = _clean_html(e.get("summary") or e.get("description") or "")
        link = e.get("link") or url
        mins = _minutes_ago(e)
        if mins > 60 * 36:
            continue
        text = f"{title}. {summary}"
        analysis = analyze(title, summary)
        tickers = analysis.direct
        tags = _flow_tags(text, base_tags)
        sent, _ = polarity(text, title=title, body=summary)
        host = _host(link) or _host(url)
        impact = impact_score(
            text=text,
            source_host=host,
            minutes_ago=mins,
            ticker_count=len(tickers) or 1,
            title=title,
            body=summary,
            source_name=name,
        )
        if "FII" in tags or "DII" in tags:
            impact = max(impact, min(10, impact + 1))
        nid = hashlib.sha1(f"{title}|{link}".encode()).hexdigest()[:12]
        out.append(
            {
                "id": nid,
                "headline": title,
                "summary": summary[:400] if summary else title,
                "source": name,
                "url": link,
                "minutesAgo": mins,
                "publishedAt": _published_iso(e),
                "sentiment": sent,
                "impact": impact,
                "credibility": next(
                    (
                        c
                        for h, c in [
                            ("economictimes", 0.9),
                            ("moneycontrol", 0.9),
                            ("livemint", 0.8),
                            ("thehindu", 0.8),
                            ("ndtv", 0.7),
                            ("google", 0.55),
                        ]
                        if h in host
                    ),
                    0.6,
                ),
                "kind": "news",
                "tags": tags,
                "event": analysis.events[0] if analysis.events else None,
                **analysis.as_payload(),
            }
        )
    return out


def fetch_news(*, force: bool = False, include_tweets: bool = True) -> list[dict]:
    global _cache
    now = time.time()
    with _lock:
        if not force and _cache and now - _cache[0] < TTL_S:
            return list(_cache[1])

    items: list[dict] = []
    seen: set[str] = set()
    for name, url, tags in FEEDS:
        for it in _fetch_feed(name, url, tags):
            key = it["headline"].lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(it)

    if include_tweets:
        for it in fetch_tweets(force=force):
            key = it["headline"].lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(it)

    items.sort(key=lambda x: (x["minutesAgo"], -x["impact"]))
    with _lock:
        _cache = (now, items)
    return list(items)


def news_for_symbol(
    symbol: str,
    items: list[dict] | None = None,
    *,
    min_relevance: float = CONTEXT_MIN,
) -> list[dict]:
    """Stories linked to a symbol, annotated with why they were linked.

    Includes indirect (sector/macro) links above ``min_relevance``; callers
    split them out using ``linkType``.
    """
    items = items if items is not None else fetch_news()
    sym = symbol.upper()
    matched: list[dict] = []
    for n in items:
        link = next((l for l in n.get("links", []) if l["symbol"] == sym), None)
        if not link or link["relevance"] < min_relevance:
            continue
        item = dict(n)
        item["relevance"] = link["relevance"]
        item["linkType"] = link["type"]
        item["linkReason"] = link["reason"]
        item["expectedDirection"] = link["direction"]
        matched.append(item)
    matched.sort(key=lambda n: (-n["relevance"], n["minutesAgo"], -n["impact"]))
    return matched
