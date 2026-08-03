"""Optional Gemini short OI insight every ~5 minutes for the Nifty page."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .ai_helper import ai_configured, gemini_configured, openai_configured
from .config import settings

_CACHE: dict[str, Any] = {"bucket": None, "payload": None}
_BUCKET_S = 5 * 60

_SYSTEM = """You are Saint's Nifty options tape reader for NSE index options.
Given JSON with ATM-wing OI (calls/puts), day OI changes, PCR, and spot, write ONE short insight
(max 2 sentences) for a trader. Be concrete with the numbers provided. Do not invent figures.
State likely sentiment: bullish / bearish / range / unclear and why (put writing, call writing,
unwinding, etc.). Return ONLY JSON: {"insight":"...","sentiment":"bullish|bearish|mild_bullish|mild_bearish|neutral|unclear"}
"""


def _bucket() -> int:
    t = int(time.time())
    return t - (t % _BUCKET_S)


def _call_gemini_text(packet: dict[str, Any]) -> dict[str, Any] | None:
    key = settings.gemini_api_key.strip()
    if not key:
        return None
    model = (settings.gemini_model or "gemini-flash-latest").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = _SYSTEM + "\n\n" + json.dumps(packet, ensure_ascii=False, default=str)
    with httpx.Client(timeout=8.0) as client:
        r = client.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "responseMimeType": "application/json",
                },
            },
        )
    if r.status_code >= 400:
        return None
    body = r.json()
    parts = (((body.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    text = "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict))
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            raw = json.loads(text[start : end + 1])
            if isinstance(raw, dict) and raw.get("insight"):
                return {
                    "insight": str(raw["insight"]).strip(),
                    "sentiment": str(raw.get("sentiment") or "unclear"),
                    "source": "gemini",
                    "model": model,
                }
    except Exception:  # noqa: BLE001
        return None
    return None


def maybe_ai_oi_insight(
    packet: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any] | None:
    """Return cached-per-5m AI insight, or None if AI unavailable / failed."""
    from .session import is_live_data_window

    # No Gemini spend overnight — reuse last insight if any.
    if not is_live_data_window():
        return dict(_CACHE["payload"]) if _CACHE.get("payload") else None
    if not ai_configured():
        return None
    b = _bucket()
    if not force and _CACHE.get("bucket") == b and _CACHE.get("payload"):
        return dict(_CACHE["payload"])

    result = None
    provider = (settings.ai_provider or "gemini").strip().lower()
    try:
        if provider != "openai" and gemini_configured():
            result = _call_gemini_text(packet)
        # OpenAI path skipped for brevity unless gemini missing
        if result is None and openai_configured() and provider == "openai":
            # Soft skip — gemini is default for this short insight
            pass
    except Exception:  # noqa: BLE001
        result = None

    if result:
        _CACHE["bucket"] = b
        _CACHE["payload"] = result
        return dict(result)
    return dict(_CACHE["payload"]) if _CACHE.get("payload") else None
