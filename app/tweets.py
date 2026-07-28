"""X/Twitter ingest via Nitter RSS mirrors (+ optional official API later)."""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Lock

import feedparser
import httpx

from .linking import analyze
from .sentiment import impact_score, polarity

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# India market / finance handles worth polling
HANDLES = [
    "CNBCTV18Live",
    "ZeeBusiness",
    "moneycontrolcom",
    "EconomicTimes",
    "NSEIndia",
    "BSEIndia",
    "LiveMint",
]

NITTER_HOSTS = [
    "https://nitter.net",
    "https://xcancel.com",
]

_cache: tuple[float, list[dict]] | None = None
_lock = Lock()
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
    raw = entry.get("published") or entry.get("updated")
    if raw:
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 60))
        except Exception:
            pass
    return 9999


def _published_iso(entry: dict) -> str | None:
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            try:
                return datetime(*st[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    return None


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _fetch_handle(handle: str) -> list[dict]:
    out: list[dict] = []
    for host in NITTER_HOSTS:
        url = f"{host}/{handle}/rss"
        try:
            with httpx.Client(
                headers={"User-Agent": UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"},
                follow_redirects=True,
                timeout=20.0,
            ) as client:
                r = client.get(url)
                if r.status_code != 200:
                    continue
                parsed = feedparser.parse(r.text)
                if not parsed.entries:
                    continue
                for e in parsed.entries:
                    title = _clean(e.get("title") or "")
                    if not title or title.lower().startswith("rss reader"):
                        continue
                    summary = _clean(e.get("summary") or e.get("description") or "")
                    link = e.get("link") or f"https://x.com/{handle}"
                    # Prefer canonical x.com links when nitter rewrites
                    link = link.replace("nitter.net", "x.com").replace("xcancel.com", "x.com")
                    mins = _minutes_ago(e)
                    if mins > 60 * 36:
                        continue
                    text = f"{title}. {summary}"
                    analysis = analyze(title, summary)
                    sent, _ = polarity(text, title=title, body=summary)
                    impact = impact_score(
                        text=text,
                        source_host="x.com",
                        minutes_ago=mins,
                        ticker_count=len(analysis.direct),
                        title=title,
                        body=summary,
                        source_name=f"@{handle}",
                    )
                    # Tweets move fast but are noisier — slight impact dampen
                    impact = max(1, min(10, impact - 1))
                    nid = hashlib.sha1(f"tw|{handle}|{title}|{link}".encode()).hexdigest()[:12]
                    tags = ["tweet"]
                    low = text.lower()
                    if "fii" in low or "foreign institutional" in low:
                        tags.append("FII")
                    if "dii" in low or "domestic institutional" in low:
                        tags.append("DII")
                    out.append(
                        {
                            "id": nid,
                            "headline": title[:220],
                            "summary": (summary or title)[:400],
                            "source": f"@{handle}",
                            "url": link,
                            "minutesAgo": mins,
                            "publishedAt": _published_iso(e),
                            "sentiment": sent,
                            "impact": impact,
                            "credibility": 0.55,
                            "kind": "tweet",
                            "tags": tags,
                            **analysis.as_payload(),
                        }
                    )
                if out:
                    return out
        except Exception:
            continue
    return out


def fetch_tweets(*, force: bool = False) -> list[dict]:
    global _cache
    now = time.time()
    with _lock:
        if not force and _cache and now - _cache[0] < TTL_S:
            return list(_cache[1])

    items: list[dict] = []
    seen: set[str] = set()
    for handle in HANDLES:
        for it in _fetch_handle(handle):
            key = it["headline"].lower()[:160]
            if key in seen:
                continue
            seen.add(key)
            items.append(it)

    items.sort(key=lambda x: (x["minutesAgo"], -x["impact"]))
    with _lock:
        _cache = (now, items)
    return list(items)
