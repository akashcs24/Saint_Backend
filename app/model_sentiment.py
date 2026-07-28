"""Optional India FinBERT scoring for direct company headlines.

Off by default (``SAINT_USE_FINBERT=false``). When enabled and the model loads,
batch-scores headlines and blends with the rule/event polarity. Macro themes
and sector sensitivities stay rule-driven — the model only helps company news.
"""

from __future__ import annotations

import logging
from threading import Lock

from .config import settings

logger = logging.getLogger(__name__)

_lock = Lock()
_pipeline = None
_load_attempted = False
_cache: dict[str, tuple[str, float]] = {}


def finbert_enabled() -> bool:
    return bool(getattr(settings, "use_finbert", False))


def _get_pipeline():
    global _pipeline, _load_attempted
    if not finbert_enabled():
        return None
    with _lock:
        if _load_attempted:
            return _pipeline
        _load_attempted = True
        try:
            from transformers import pipeline

            _pipeline = pipeline(
                "text-classification",
                model="Vansh180/FinBERT-India-v1",
                truncation=True,
                max_length=128,
            )
            logger.info("FinBERT-India loaded")
        except Exception as exc:  # noqa: BLE001 — optional dependency
            logger.warning("FinBERT unavailable, using rules only: %s", exc)
            _pipeline = None
        return _pipeline


def score_headline(text: str) -> tuple[str, float] | None:
    """Return (Positive|Negative|Neutral, confidence) or None if unavailable."""
    key = (text or "").strip()
    if not key:
        return None
    if key in _cache:
        return _cache[key]
    pipe = _get_pipeline()
    if pipe is None:
        return None
    try:
        out = pipe(key[:512])
        # pipeline may return list[dict] or list[list[dict]] depending on version
        rows = out[0] if out and isinstance(out[0], list) else out
        if isinstance(rows, dict):
            rows = [rows]
        best = max(rows, key=lambda r: float(r.get("score") or 0))
        label = str(best.get("label") or "").lower()
        score = float(best.get("score") or 0)
        if "pos" in label:
            sent = "Positive"
        elif "neg" in label:
            sent = "Negative"
        else:
            sent = "Neutral"
        _cache[key] = (sent, score)
        return sent, score
    except Exception as exc:  # noqa: BLE001
        logger.debug("FinBERT score failed: %s", exc)
        return None


def score_headlines(texts: list[str], *, batch_size: int = 16) -> dict[str, tuple[str, float]]:
    """Batch-score uncached headlines and return every available result."""
    keys = list(dict.fromkeys((text or "").strip() for text in texts if (text or "").strip()))
    missing = [key for key in keys if key not in _cache]
    pipe = _get_pipeline()
    if missing and pipe is not None:
        try:
            outputs = pipe([key[:512] for key in missing], batch_size=batch_size)
            for key, output in zip(missing, outputs, strict=False):
                rows = output if isinstance(output, list) else [output]
                best = max(rows, key=lambda row: float(row.get("score") or 0))
                label = str(best.get("label") or "").lower()
                score = float(best.get("score") or 0)
                sent = "Positive" if "pos" in label else "Negative" if "neg" in label else "Neutral"
                _cache[key] = (sent, score)
        except Exception as exc:  # noqa: BLE001
            logger.debug("FinBERT batch score failed: %s", exc)
    return {key: _cache[key] for key in keys if key in _cache}


def blend_company_sentiment(
    rules_sentiment: str,
    rules_score: float,
    headline: str,
) -> tuple[str, float, str]:
    """Blend rules with FinBERT for a company headline.

    Returns (sentiment, score, scorer_tag).
    """
    model = score_headline(headline)
    if model is None:
        return rules_sentiment, rules_score, "rules"

    m_sent, m_conf = model
    # High-confidence model can override a weak/neutral rules read.
    if m_conf >= 0.7 and rules_sentiment == "Neutral":
        signed = m_conf if m_sent == "Positive" else -m_conf if m_sent == "Negative" else 0.0
        return m_sent, signed, "finbert+rules"
    if m_conf >= 0.75 and m_sent == rules_sentiment:
        boosted = max(abs(rules_score), m_conf)
        signed = boosted if m_sent == "Positive" else -boosted if m_sent == "Negative" else 0.0
        return m_sent, signed, "finbert+rules"
    if m_conf >= 0.8 and rules_sentiment != "Neutral" and m_sent != rules_sentiment:
        # Disagreement: keep rules (event map is more reliable for SEBI/NPA),
        # but damp conviction via a lower absolute score.
        return rules_sentiment, rules_score * 0.7, "rules"
    return rules_sentiment, rules_score, "rules"
