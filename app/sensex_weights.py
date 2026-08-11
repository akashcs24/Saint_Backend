"""Sensex 30 constituent weights — scrape with baked-in fallback."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from .universe import UNIVERSE

_URL = "https://www.smart-investing.in/indices-bse-nse.php?index=SENSEX"
_TTL_S = 12 * 60 * 60
_MIN_NAMES = 25
_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "sensex_weights_cache.json"

FALLBACK_WEIGHTS: dict[str, float] = {
    "RELIANCE": 12.0,
    "HDFCBANK": 9.5,
    "ICICIBANK": 8.2,
    "INFY": 6.8,
    "TCS": 6.5,
    "BHARTIARTL": 5.9,
    "LT": 5.2,
    "ITC": 4.8,
    "SBIN": 4.5,
    "HINDUNILVR": 4.2,
    "KOTAKBANK": 3.8,
    "AXISBANK": 3.5,
    "MARUTI": 3.2,
    "SUNPHARMA": 3.0,
    "TITAN": 2.8,
    "ULTRACEMCO": 2.5,
    "M&M": 2.3,
    "NTPC": 2.2,
    "BAJFINANCE": 2.0,
    "HCLTECH": 1.9,
    "ASIANPAINT": 1.8,
    "TATASTEEL": 1.7,
    "POWERGRID": 1.6,
    "WIPRO": 1.5,
    "JSWSTEEL": 1.4,
    "TECHM": 1.3,
    "INDUSINDBK": 1.2,
    "NESTLEIND": 1.1,
    "HINDALCO": 1.0,
    "TRENT": 0.9,
}

_NAME_TO_SYMBOL: dict[str, str] = {
    "Mahindra & Mahindra": "M&M",
    "Mahindra and Mahindra": "M&M",
    "Larsen & Toubro": "LT",
    "Larsen and Toubro": "LT",
    "Bajaj Auto": "BAJAJ-AUTO",
    "Dr. Reddy's Laboratories": "DRREDDY",
    "Dr Reddys Laboratories": "DRREDDY",
    "Nestle India": "NESTLEIND",
    "IndusInd Bank": "INDUSINDBK",
}

_MEM: dict[str, Any] = {"ts": 0.0, "pack": None}


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


def _symbol_for_name(name: str) -> str | None:
    clean = _normalize_name(name)
    if clean in _NAME_TO_SYMBOL:
        return _NAME_TO_SYMBOL[clean]
    for sym, meta in UNIVERSE.items():
        display = str(meta.get("name") or "")
        if display and _normalize_name(display).lower() == clean.lower():
            return sym
        if sym == clean.upper().replace(" ", ""):
            return sym
    token = clean.upper().replace(" ", "").replace("&", "").replace(".", "")
    for sym in UNIVERSE:
        if sym.replace("-", "") == token:
            return sym
    return None


def _load_disk() -> dict[str, Any] | None:
    try:
        if not _CACHE_PATH.exists():
            return None
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("weights"):
            return data
    except Exception:  # noqa: BLE001
        pass
    return None


def _save_disk(pack: dict[str, Any]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _scrape() -> dict[str, float]:
    resp = requests.get(_URL, timeout=20, headers={"User-Agent": "SaintMarket/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError("Sensex weight table not found")
    weights: dict[str, float] = {}
    for row in table.find_all("tr"):
        cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        name, pct_raw = cells[0], cells[-1]
        m = re.search(r"([\d.]+)", pct_raw.replace(",", ""))
        if not m:
            continue
        sym = _symbol_for_name(name)
        if not sym or sym not in UNIVERSE:
            continue
        weights[sym] = float(m.group(1))
    if len(weights) < _MIN_NAMES:
        raise ValueError(f"Too few Sensex names mapped ({len(weights)})")
    return weights


def get_sensex_weights(*, force: bool = False) -> dict[str, Any]:
    now = time.time()
    if not force and _MEM.get("pack") and now - float(_MEM.get("ts") or 0) < _TTL_S:
        return dict(_MEM["pack"])

    disk = None if force else _load_disk()
    if disk and now - float(disk.get("fetchedAt") or 0) < _TTL_S:
        _MEM["ts"] = now
        _MEM["pack"] = disk
        return dict(disk)

    source = "fallback"
    unmapped: list[str] = []
    try:
        weights = _scrape()
        source = "smart-investing.in"
    except Exception:  # noqa: BLE001
        weights = {k: v for k, v in FALLBACK_WEIGHTS.items() if k in UNIVERSE}

    pack = {
        "weights": weights,
        "source": source,
        "count": len(weights),
        "fetchedAt": now,
        "unmapped": unmapped,
    }
    if source != "fallback":
        _save_disk(pack)
    _MEM["ts"] = now
    _MEM["pack"] = pack
    return dict(pack)
