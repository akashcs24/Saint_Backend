"""Auto-refresh Nifty 50 constituent weights.

Primary source: smart-investing.in index weightage table (market-cap based %).
Falls back to a baked-in snapshot if the scrape fails or maps too few names.
Cached ~12h so dashboard refreshes don't hammer the site.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from .universe import UNIVERSE

_URL = "https://www.smart-investing.in/indices-bse-nse.php?index=NIFTY"
_TTL_S = 12 * 60 * 60
_MIN_NAMES = 40
_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "nifty_weights_cache.json"

# Baked snapshot (Jul 2026) — used offline / on scrape failure.
FALLBACK_WEIGHTS: dict[str, float] = {
    "RELIANCE": 9.00,
    "BHARTIARTL": 6.18,
    "HDFCBANK": 5.92,
    "ICICIBANK": 5.39,
    "SBIN": 4.90,
    "TCS": 4.32,
    "BAJFINANCE": 3.39,
    "LT": 2.72,
    "HINDUNILVR": 2.66,
    "SUNPHARMA": 2.46,
    "INFY": 2.28,
    "MARUTI": 2.24,
    "TITAN": 2.19,
    "ADANIENT": 2.14,
    "ADANIPORTS": 2.12,
    "M&M": 2.09,
    "KOTAKBANK": 1.99,
    "AXISBANK": 1.98,
    "ITC": 1.86,
    "HCLTECH": 1.83,
    "ULTRACEMCO": 1.82,
    "NTPC": 1.77,
    "BAJAJ-AUTO": 1.60,
    "BAJAJFINSV": 1.59,
    "JSWSTEEL": 1.58,
    "ONGC": 1.56,
    "BEL": 1.55,
    "ETERNAL": 1.48,
    "POWERGRID": 1.40,
    "COALINDIA": 1.37,
    "ASIANPAINT": 1.35,
    "SHRIRAMFIN": 1.27,
    "TATASTEEL": 1.20,
    "EICHERMOT": 1.11,
    "HINDALCO": 1.10,
    "GRASIM": 1.10,
    "INDIGO": 1.05,
    "SBILIFE": 0.97,
    "WIPRO": 0.92,
    "TRENT": 0.81,
    "JIOFIN": 0.81,
    "TECHM": 0.80,
    "APOLLOHOSP": 0.66,
    "TMPV": 0.63,
    "HDFCLIFE": 0.62,
    "CIPLA": 0.59,
    "TATACONSUM": 0.57,
    "MAXHEALTH": 0.56,
    "DRREDDY": 0.50,
}

# Company-name overrides when UNIVERSE display names don't match the table.
_NAME_TO_SYMBOL: dict[str, str] = {
    "RELIANCE INDUSTRIES": "RELIANCE",
    "BHARTI AIRTEL": "BHARTIARTL",
    "HDFC BANK": "HDFCBANK",
    "ICICI BANK": "ICICIBANK",
    "STATE BANK OF INDIA": "SBIN",
    "TATA CONSULTANCY SERVICES": "TCS",
    "BAJAJ FINANCE": "BAJFINANCE",
    "LARSEN & TOUBRO": "LT",
    "HINDUSTAN UNILEVER": "HINDUNILVR",
    "SUN PHARMACEUTICAL INDUSTRIES": "SUNPHARMA",
    "INFOSYS": "INFY",
    "MARUTI SUZUKI INDIA": "MARUTI",
    "TITAN": "TITAN",
    "ADANI ENTERPRISES": "ADANIENT",
    "ADANI PORTS AND SPECIAL ECONOMIC ZONE": "ADANIPORTS",
    "MAHINDRA & MAHINDRA": "M&M",
    "KOTAK MAHINDRA BANK": "KOTAKBANK",
    "AXIS BANK": "AXISBANK",
    "ITC": "ITC",
    "HCL TECHNOLOGIES": "HCLTECH",
    "ULTRATECH CEMENT": "ULTRACEMCO",
    "NTPC": "NTPC",
    "BAJAJ AUTO": "BAJAJ-AUTO",
    "BAJAJ FINSERV": "BAJAJFINSV",
    "JSW STEEL": "JSWSTEEL",
    "OIL & NATURAL GAS CORPORATION": "ONGC",
    "BHARAT ELECTRONICS": "BEL",
    "ETERNAL": "ETERNAL",
    "POWER GRID CORPORATION OF INDIA": "POWERGRID",
    "COAL INDIA": "COALINDIA",
    "ASIAN PAINTS": "ASIANPAINT",
    "SHRIRAM FINANCE": "SHRIRAMFIN",
    "TATA STEEL": "TATASTEEL",
    "EICHER MOTORS": "EICHERMOT",
    "HINDALCO INDUSTRIES": "HINDALCO",
    "GRASIM INDUSTRIES": "GRASIM",
    "INTERGLOBE AVIATION": "INDIGO",
    "SBI LIFE INSURANCE": "SBILIFE",
    "WIPRO": "WIPRO",
    "TRENT": "TRENT",
    "JIO FINANCIAL SERVICES": "JIOFIN",
    "TECH MAHINDRA": "TECHM",
    "APOLLO HOSPITALS ENTERPRISE": "APOLLOHOSP",
    "TATA MOTORS PASSENGER VEHICLES": "TMPV",
    "HDFC LIFE INSURANCE": "HDFCLIFE",
    "CIPLA": "CIPLA",
    "TATA CONSUMER PRODUCTS": "TATACONSUM",
    "MAX HEALTHCARE INSTITUTE": "MAXHEALTH",
    "DR REDDYS LABORATORIES": "DRREDDY",
    "DR REDDY S LABORATORIES": "DRREDDY",
}

_mem: dict[str, Any] = {"ts": 0.0, "payload": None}


def _norm(name: str) -> str:
    s = name.upper()
    s = re.sub(r"\b(LIMITED|LTD\.?|CO\.?|COMPANY|THE)\b", "", s)
    s = re.sub(r"[^A-Z0-9&]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _resolve_symbol(company: str) -> str | None:
    n = _norm(company)
    if n in _NAME_TO_SYMBOL:
        return _NAME_TO_SYMBOL[n]
    # Match against Saint universe display names.
    for sym, meta in UNIVERSE.items():
        un = _norm(str(meta.get("name") or ""))
        if not un:
            continue
        if un == n or n.startswith(un) or un.startswith(n):
            return sym
    return None


def _scrape() -> tuple[dict[str, float], list[str]] | None:
    r = requests.get(
        _URL,
        timeout=20,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Saint/1.0)",
            "Accept": "text/html",
        },
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = None
    for t in soup.find_all("table"):
        first = t.find("tr")
        if not first:
            continue
        headers = [c.get_text(" ", strip=True).lower() for c in first.find_all(["th", "td"])]
        if any("weight" in h for h in headers) and any("company" in h for h in headers):
            table = t
            break
    if table is None:
        return None

    weights: dict[str, float] = {}
    unmapped: list[str] = []
    for tr in table.find_all("tr")[1:]:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        company, raw_w = cells[0], cells[1]
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", raw_w)
        if not m:
            continue
        wt = float(m.group(1))
        sym = _resolve_symbol(company)
        if sym:
            weights[sym] = wt
        else:
            unmapped.append(company)
    if len(weights) < _MIN_NAMES:
        return None
    return weights, unmapped


def _load_disk() -> dict[str, Any] | None:
    try:
        if not _CACHE_PATH.exists():
            return None
        data = json.loads(_CACHE_PATH.read_text())
        if not isinstance(data, dict) or "weights" not in data:
            return None
        return data
    except Exception:  # noqa: BLE001
        return None


def _save_disk(payload: dict[str, Any]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:  # noqa: BLE001
        pass


def _payload(weights: dict[str, float], *, source: str, unmapped: list[str] | None = None) -> dict[str, Any]:
    return {
        "weights": weights,
        "source": source,
        "count": len(weights),
        "sum": round(sum(weights.values()), 2),
        "unmapped": unmapped or [],
        "fetchedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def get_nifty_weights(*, force: bool = False) -> dict[str, Any]:
    """Return `{weights, source, ...}` — live scrape when fresh, else fallback."""
    now = time.time()
    if not force and _mem["payload"] is not None and now - float(_mem["ts"]) < _TTL_S:
        return _mem["payload"]

    disk = _load_disk()
    if (
        not force
        and disk
        and isinstance(disk.get("weights"), dict)
        and len(disk["weights"]) >= _MIN_NAMES
        and now - float(disk.get("ts") or 0) < _TTL_S
    ):
        _mem["ts"] = now
        _mem["payload"] = disk
        return disk

    try:
        scraped = _scrape()
    except Exception:  # noqa: BLE001
        scraped = None

    if scraped:
        weights, unmapped = scraped
        payload = _payload(weights, source="smart-investing", unmapped=unmapped)
        payload["ts"] = now
        _mem["ts"] = now
        _mem["payload"] = payload
        _save_disk(payload)
        return payload

    if disk and isinstance(disk.get("weights"), dict) and disk["weights"]:
        disk = {**disk, "source": f"{disk.get('source', 'cache')} (stale)"}
        _mem["ts"] = now
        _mem["payload"] = disk
        return disk

    payload = _payload(dict(FALLBACK_WEIGHTS), source="fallback")
    payload["ts"] = now
    _mem["ts"] = now
    _mem["payload"] = payload
    return payload
