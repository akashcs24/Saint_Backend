from __future__ import annotations

YAHOO_OVERRIDES: dict[str, str] = {
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "J&KBANK": "J&KBANK.NS",
    "M&M": "M&M.NS",
    "M&MFIN": "M&MFIN.NS",
    "L&TFH": "L&TFH.NS",
    "NAM-INDIA": "NAM-INDIA.NS",
    "GMRINFRA": "GMRINFRA.NS",
    "ZOMATO": "ETERNAL.NS",
    "TMPV": "TMPV.NS",
}

# Yahoo index symbols used for the dashboard strip
INDEX_YAHOO: dict[str, str] = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "VIX": "^INDIAVIX",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCAP": "^NSEMDCP50",
}

INDEX_NAMES: dict[str, str] = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "BANK NIFTY",
    "SENSEX": "SENSEX",
    "VIX": "INDIA VIX",
    "FINNIFTY": "FIN NIFTY",
    "MIDCAP": "NIFTY MIDCAP",
}


def to_yahoo_ticker(symbol: str) -> str:
    sym = symbol.strip().upper()
    if sym in INDEX_YAHOO:
        return INDEX_YAHOO[sym]
    if sym in YAHOO_OVERRIDES:
        return YAHOO_OVERRIDES[sym]
    if sym.endswith(".NS") or sym.endswith(".BO") or sym.startswith("^"):
        return sym
    return f"{sym}.NS"


def parquet_name(yahoo_ticker: str) -> str:
    return yahoo_ticker.replace("^", "_").replace("/", "_") + ".parquet"
