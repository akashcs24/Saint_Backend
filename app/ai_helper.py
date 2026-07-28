"""On-demand AI helper for a single stock (Gemini primary, OpenAI optional).

Grounded packet = Saint news signal + dedicated F&T sources (Yahoo fundamentals
+ computed technicals from daily OHLCV). Model must not invent numbers.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .config import settings
from .fundamentals import get_fundamentals_deep
from .levels import sr_position
from .technicals import get_technicals

_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL_S = 8 * 60  # shorter — timing advice goes stale fast
_VERDICTS = {
    "buy_now",
    "wait_pullback",
    "avoid_chase",
    "short_now",
    "stay_out",
    "watch",
}

_SYSTEM = """You are Saint's timing co-pilot for Indian NSE stocks (intraday / swing entry).
You receive a grounded JSON packet with:
1) saintSignal / thesis / news — catalyst from Saint (NOT a permission to chase)
2) fundamentals — Yahoo (valuation, growth, leverage) — context only
3) technicals — FRESH computed indicators: RSI, EMA9/21/50, MACD, Bollinger, volume ratio, session VWAP, range
4) levels — support / resistance / VWAP distance

Your job is ENTRY TIMING, not restating the dashboard.
Saint may show Buy long with medium conviction while price is still early — that is often the USEFUL window.
Do NOT wait for high conviction / already-priced moves; those are usually late.

Rules:
- Use ONLY packet facts. Never invent RSI/EMA/MACD/VWAP/delivery/prices.
- If deliveryPct is null, say delivery unknown — do not invent.
- Prefer actionable timing: buy_now / wait_pullback / avoid_chase / short_now / stay_out / watch.
- Headline must answer: "Is this a good time to enter NOW?"
- Bullets must cite concrete indicator values from technicals (RSI, EMA stack, MACD state, VWAP, BB, volume).
- At least 2 bullets must be technical timing; 1 may be fundamental context; 1 news catalyst.
- If news is bullish but tape is extended above VWAP + RSI overbought → avoid_chase or wait_pullback (not buy_now).
- If news is bullish, RSI mid, near EMA21/support or reclaiming VWAP with MACD improving → buy_now even if Saint conviction is only medium.
- If already moved hard (actionNote like +X% since headline, high RSI, near 20d high) → avoid_chase.
- Be crisp. No essays. No "as an AI".

Output MUST be one JSON object:
{
  "verdict": "buy_now" | "wait_pullback" | "avoid_chase" | "short_now" | "stay_out" | "watch",
  "timing": "early" | "ok" | "late",
  "headline": "one sentence answering buy/short NOW or wait",
  "setup": "one line trigger (e.g. reclaim VWAP / hold EMA21 / MACD hist flip)",
  "bullets": ["max 4 short bullets with numbers"],
  "conflicts": ["0-2 conflicts"],
  "confidence": "low" | "medium" | "high"
}
"""


def gemini_configured() -> bool:
    return bool((settings.gemini_api_key or "").strip())


def openai_configured() -> bool:
    return bool((settings.openai_api_key or "").strip())


def ai_configured() -> bool:
    provider = (settings.ai_provider or "gemini").strip().lower()
    if provider == "openai":
        return openai_configured() or gemini_configured()
    return gemini_configured() or openai_configured()


def _cache_get(symbol: str) -> dict | None:
    hit = _CACHE.get(symbol.upper())
    if not hit:
        return None
    ts, payload = hit
    if time.time() - ts > _CACHE_TTL_S:
        return None
    return dict(payload)


def _cache_set(symbol: str, payload: dict) -> None:
    _CACHE[symbol.upper()] = (time.time(), payload)


def build_helper_packet(
    stock: dict,
    news: list[dict],
    context: list[dict],
    *,
    force_fresh: bool = True,
) -> dict[str, Any]:
    """Grounded packet: Saint signal + fresh F&T sources for timing."""
    sym = str(stock.get("symbol") or "").upper()
    ltp = stock.get("ltp")
    sr = {}
    try:
        if isinstance(ltp, (int, float)) and ltp > 0:
            sr = sr_position(sym, float(ltp)) or {}
    except Exception:  # noqa: BLE001
        sr = {}

    fundamentals = get_fundamentals_deep(sym)
    technicals = get_technicals(
        sym,
        ltp=float(ltp) if isinstance(ltp, (int, float)) else None,
        force=force_fresh,
    )

    def _news_bits(items: list[dict], limit: int) -> list[dict]:
        out = []
        for n in items[:limit]:
            out.append(
                {
                    "headline": n.get("headline"),
                    "sentiment": n.get("sentiment"),
                    "impact": n.get("impact"),
                    "minutesAgo": n.get("minutesAgo"),
                    "linkType": n.get("linkType"),
                    "linkReason": n.get("linkReason"),
                }
            )
        return out

    fund_for_model = dict(fundamentals)
    about = fund_for_model.get("about")
    if isinstance(about, str) and len(about) > 220:
        fund_for_model["about"] = about[:220] + "…"

    return {
        "symbol": sym,
        "name": stock.get("name") or fundamentals.get("name"),
        "sector": stock.get("sector") or fundamentals.get("sector"),
        "index": stock.get("index"),
        "question": "Is this a good time to enter NOW (not after the move is done)?",
        "dataSources": {
            "news": "saint",
            "fundamentals": fundamentals.get("source") or "yahoo_finance",
            "technicals": technicals.get("source") or "yahoo_daily+intraday",
            "levels": "saint_swing_pivots",
        },
        "price": {
            "ltp": stock.get("ltp"),
            "changePct": stock.get("changePct"),
            "dayRange": stock.get("dayRange"),
            "yearRange": stock.get("yearRange"),
            "volume": stock.get("volume"),
            "avgVolume": stock.get("avgVolume"),
        },
        "fundamentals": fund_for_model,
        "technicals": technicals,
        "saintSignal": {
            "bias": stock.get("bias"),
            "action": stock.get("action"),
            "actionNote": stock.get("actionNote"),
            "conviction": stock.get("conviction"),
            "confidence": stock.get("confidence"),
            "convictionDrivers": stock.get("convictionDrivers"),
            "impact": stock.get("impact"),
            "themeConflict": stock.get("themeConflict"),
            "note": "Medium conviction can still be an early entry — do not wait for peak conviction.",
        },
        "thesis": {
            "health": stock.get("thesisHealth"),
            "label": stock.get("thesisLabel"),
            "openMovePct": stock.get("thesisOpenMovePct"),
            "lastMovePct": stock.get("thesisLastMovePct"),
        },
        "levels": {
            "nearestResistance": stock.get("nearestResistance") or sr.get("nearestResistance"),
            "nearestSupport": stock.get("nearestSupport") or sr.get("nearestSupport"),
            "distResistPct": stock.get("distResistPct") or sr.get("distResistPct"),
            "distSupportPct": stock.get("distSupportPct") or sr.get("distSupportPct"),
            "sessionVwap": stock.get("sessionVwap") or technicals.get("sessionVwap"),
            "atResistance": sr.get("atResistance"),
            "atSupport": sr.get("atSupport"),
        },
        "tape": {
            "gapPct": stock.get("gapPct"),
            "openMovePct": stock.get("openMovePct"),
            "fromVwapPct": stock.get("fromVwapPct") or technicals.get("distVwapPct"),
        },
        "companyNews": _news_bits(news, 5),
        "contextNews": _news_bits(context, 4),
    }


def _normalize_result(raw: dict, *, model: str, cached: bool, source: str) -> dict:
    verdict = str(raw.get("verdict") or "watch").strip().lower()
    # Back-compat mapping from older prompts
    legacy = {
        "support_long": "buy_now",
        "support_short": "short_now",
        "neutral": "watch",
        "caution": "avoid_chase",
    }
    verdict = legacy.get(verdict, verdict)
    if verdict not in _VERDICTS:
        verdict = "watch"
    bullets = raw.get("bullets") or []
    if not isinstance(bullets, list):
        bullets = [str(bullets)]
    conflicts = raw.get("conflicts") or []
    if not isinstance(conflicts, list):
        conflicts = [str(conflicts)]
    conf = str(raw.get("confidence") or "low").strip().lower()
    if conf not in {"low", "medium", "high"}:
        conf = "low"
    timing = str(raw.get("timing") or "").strip().lower()
    if timing not in {"early", "ok", "late"}:
        timing = "ok" if verdict in {"buy_now", "short_now"} else "late" if verdict == "avoid_chase" else "ok"
    return {
        "ready": True,
        "verdict": verdict,
        "timing": timing,
        "headline": str(raw.get("headline") or "No clear timing edge")[:220],
        "setup": str(raw.get("setup") or "")[:220],
        "bullets": [str(b)[:240] for b in bullets[:4]],
        "conflicts": [str(c)[:220] for c in conflicts[:2]],
        "confidence": conf,
        "model": model,
        "cached": cached,
        "source": source,
        "dataSources": {
            "fundamentals": "yahoo_finance",
            "technicals": "yahoo_daily+intraday",
            "news": "saint",
        },
    }


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences if the model wraps them.
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        raw = json.loads(text)
        if isinstance(raw, dict):
            return raw
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        raw = json.loads(text[start : end + 1])
        if isinstance(raw, dict):
            return raw
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _call_gemini(packet: dict) -> dict:
    key = settings.gemini_api_key.strip()
    model = (settings.gemini_model or "gemini-flash-latest").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = (
        _SYSTEM
        + "\n\nAnalyse this Saint packet and return ONLY the JSON verdict object.\n\n"
        + json.dumps(packet, ensure_ascii=False, default=str)
    )
    with httpx.Client(timeout=45.0) as client:
        r = client.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": key,
            },
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "responseMimeType": "application/json",
                },
            },
        )
    if r.status_code >= 400:
        return {
            "ready": False,
            "error": "gemini_http",
            "message": f"Gemini error {r.status_code}: {r.text[:300]}",
        }
    body = r.json()
    parts = (((body.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    text = ""
    for p in parts:
        if isinstance(p, dict) and p.get("text"):
            text += str(p["text"])
    raw = _extract_json_object(text or "{}")
    return _normalize_result(raw, model=model, cached=False, source="gemini")


def _call_openai(packet: dict) -> dict:
    key = settings.openai_api_key.strip()
    model = (settings.openai_model or "gpt-4o-mini").strip()
    with httpx.Client(timeout=45.0) as client:
        r = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            "Analyse this Saint packet and return the JSON verdict object.\n\n"
                            + json.dumps(packet, ensure_ascii=False, default=str)
                        ),
                    },
                ],
            },
        )
    if r.status_code >= 400:
        return {
            "ready": False,
            "error": "openai_http",
            "message": f"OpenAI error {r.status_code}: {r.text[:300]}",
        }
    body = r.json()
    content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}"
    raw = _extract_json_object(content)
    return _normalize_result(raw, model=model, cached=False, source="openai")


def run_ai_helper(
    stock: dict,
    news: list[dict],
    context: list[dict],
    *,
    force: bool = False,
) -> dict:
    """Return helper verdict dict, or an error-shaped payload."""
    sym = str(stock.get("symbol") or "").upper()
    if not sym:
        return {"ready": False, "error": "missing_symbol", "message": "No symbol."}

    if not force:
        cached = _cache_get(sym)
        if cached:
            out = dict(cached)
            out["cached"] = True
            return out

    if not ai_configured():
        return {
            "ready": False,
            "error": "not_configured",
            "message": "Set SAINT_GEMINI_API_KEY (or SAINT_OPENAI_API_KEY) on the backend.",
        }

    packet = build_helper_packet(stock, news, context, force_fresh=True)
    provider = (settings.ai_provider or "gemini").strip().lower()

    try:
        if provider == "openai":
            if openai_configured():
                result = _call_openai(packet)
            elif gemini_configured():
                result = _call_gemini(packet)
            else:
                return {
                    "ready": False,
                    "error": "not_configured",
                    "message": "No AI API key configured.",
                }
        else:
            if gemini_configured():
                result = _call_gemini(packet)
            elif openai_configured():
                result = _call_openai(packet)
            else:
                return {
                    "ready": False,
                    "error": "not_configured",
                    "message": "No AI API key configured.",
                }

        if result.get("ready"):
            _cache_set(sym, result)
        return result
    except json.JSONDecodeError:
        return {"ready": False, "error": "bad_json", "message": "Could not parse model JSON."}
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "error": "request_failed", "message": str(exc)[:240]}
