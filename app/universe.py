"""NSE research universe with tiered name aliases for headline entity linking.

Membership lists are real index constituents (not invented prices/sentiment).

Aliases are grouped by how much they prove the company is the subject:

* ``strong`` — legal/full names that cannot mean anything else
  ("hindustan unilever", "itc limited").
* ``medium`` — the everyday name of the business ("infosys", "itc").
* ``weak`` — initialisms and ticker tokens that are easy to confuse
  ("ril", "hul", "sbi"). These need a second signal — a corporate event or
  India-market context — before they are treated as company news.

All aliases are matched on token boundaries, so ``itc`` can never be found
inside ``bitcoin``.
"""

from __future__ import annotations

UNIVERSE: dict[str, dict] = {
    "RELIANCE": {
        "name": "Reliance Industries",
        "sector": "OILGAS",
        "index": "NIFTY",
        "strong": ["reliance industries", "reliance ltd", "reliance limited"],
        "medium": ["reliance", "jio", "jio platforms", "reliance retail"],
        "weak": ["ril"],
    },
    "TCS": {
        "name": "Tata Consultancy Services",
        "sector": "IT",
        "index": "NIFTY",
        "strong": ["tata consultancy services", "tata consultancy"],
        "medium": ["tcs"],
        "weak": [],
    },
    "HDFCBANK": {
        "name": "HDFC Bank",
        "sector": "BANK",
        "index": "BANKNIFTY",
        "strong": ["hdfc bank", "hdfc ltd", "hdfc limited"],
        "medium": ["hdfcbank"],
        # Bare "hdfc" is ambiguous with HDFC Life — keep weak + deny life/insurance.
        "weak": ["hdfc"],
        "deny": ["hdfc life", "hdfc life insurance", "hdfc amc", "hdfc mutual fund"],
    },
    "ICICIBANK": {
        "name": "ICICI Bank",
        "sector": "BANK",
        "index": "BANKNIFTY",
        "strong": ["icici bank"],
        "medium": ["icicibank", "icici"],
        "weak": [],
    },
    "INFY": {
        "name": "Infosys",
        "sector": "IT",
        "index": "NIFTY",
        "strong": ["infosys ltd", "infosys limited"],
        "medium": ["infosys"],
        "weak": ["infy"],
    },
    "HINDUNILVR": {
        "name": "Hindustan Unilever",
        "sector": "FMCG",
        "index": "NIFTY",
        "strong": ["hindustan unilever", "hind unilever"],
        "medium": ["unilever india"],
        "weak": ["hul"],
    },
    "ITC": {
        "name": "ITC",
        "sector": "FMCG",
        "index": "NIFTY",
        "strong": ["itc limited", "itc ltd"],
        "medium": ["itc"],
        "weak": [],
        "deny": ["itc hotels", "itc infotech"],
    },
    "ITCHOTELS": {
        "name": "ITC Hotels",
        "sector": "HOTELS",
        "index": "NIFTY",
        "strong": ["itc hotels"],
        "medium": [],
        "weak": [],
    },
    "SBIN": {
        "name": "State Bank of India",
        "sector": "BANK",
        "index": "BANKNIFTY",
        "strong": ["state bank of india", "state bank"],
        "medium": ["sbin"],
        "weak": ["sbi"],
        "deny": ["sbi life", "sbi card", "sbi mutual fund", "sbi funds", "sbi life insurance"],
    },
    "BHARTIARTL": {
        "name": "Bharti Airtel",
        "sector": "TELECOM",
        "index": "NIFTY",
        "strong": ["bharti airtel"],
        "medium": ["airtel", "bharti"],
        "weak": [],
    },
    "IDEA": {
        "name": "Vodafone Idea",
        "sector": "TELECOM",
        "index": "NIFTY",
        "strong": ["vodafone idea"],
        "medium": ["vi ltd", "vodafone india"],
        "weak": [],
    },
    "LT": {
        "name": "Larsen & Toubro",
        "sector": "INFRA",
        "index": "NIFTY",
        "strong": ["larsen & toubro", "larsen and toubro", "l and t"],
        "medium": ["l&t", "l & t", "larsen"],
        "weak": [],
        "deny": ["l&t finance", "l&t technology", "ltimindtree"],
    },
    "AXISBANK": {
        "name": "Axis Bank",
        "sector": "BANK",
        "index": "BANKNIFTY",
        "strong": ["axis bank"],
        "medium": ["axisbank"],
        "weak": ["axis"],
    },
    "BAJFINANCE": {
        "name": "Bajaj Finance",
        "sector": "NBFC",
        "index": "NIFTY",
        "strong": ["bajaj finance"],
        "medium": [],
        "weak": [],
    },
    "KOTAKBANK": {
        "name": "Kotak Mahindra Bank",
        "sector": "BANK",
        "index": "BANKNIFTY",
        "strong": ["kotak mahindra bank", "kotak bank"],
        "medium": ["kotak mahindra"],
        "weak": ["kotak"],
    },
    "ASIANPAINT": {
        "name": "Asian Paints",
        "sector": "CHEMICALS",
        "index": "NIFTY",
        "strong": ["asian paints"],
        "medium": [],
        "weak": [],
    },
    "MARUTI": {
        "name": "Maruti Suzuki",
        "sector": "AUTO",
        "index": "NIFTY",
        "strong": ["maruti suzuki"],
        "medium": ["maruti"],
        "weak": [],
    },
    "SUNPHARMA": {
        "name": "Sun Pharma",
        "sector": "PHARMA",
        "index": "NIFTY",
        "strong": ["sun pharmaceutical", "sun pharma"],
        "medium": [],
        "weak": [],
    },
    "TITAN": {
        "name": "Titan Company",
        "sector": "CONSUMER",
        "index": "NIFTY",
        "strong": ["titan company"],
        "medium": ["titan", "tanishq"],
        "weak": [],
    },
    "ULTRACEMCO": {
        "name": "UltraTech Cement",
        "sector": "CEMENT",
        "index": "NIFTY",
        "strong": ["ultratech cement", "ultra tech"],
        "medium": ["ultratech"],
        "weak": [],
    },
    "NTPC": {
        "name": "NTPC",
        "sector": "POWER",
        "index": "NIFTY",
        "strong": ["ntpc ltd", "ntpc limited"],
        "medium": ["ntpc"],
        "weak": [],
    },
    "POWERGRID": {
        "name": "Power Grid",
        "sector": "POWER",
        "index": "NIFTY",
        "strong": ["power grid corporation", "power grid"],
        "medium": ["powergrid"],
        "weak": [],
    },
    "TATAMOTORS": {
        "name": "Tata Motors",
        "sector": "AUTO",
        "index": "NIFTY",
        "strong": ["tata motors", "tata motor"],
        "medium": ["tatamotors", "jaguar land rover", "jlr"],
        "weak": [],
    },
    "TATASTEEL": {
        "name": "Tata Steel",
        "sector": "METALS",
        "index": "NIFTY",
        "strong": ["tata steel"],
        "medium": [],
        "weak": [],
    },
    "ADANIENT": {
        "name": "Adani Enterprises",
        "sector": "DIVERSIFIED",
        "index": "NIFTY",
        "strong": ["adani enterprises"],
        "medium": ["adani group", "adani"],
        "weak": [],
    },
    "ADANIPORTS": {
        "name": "Adani Ports",
        "sector": "PORTS",
        "index": "NIFTY",
        "strong": ["adani ports", "mundra port"],
        "medium": [],
        "weak": [],
    },
    "ONGC": {
        "name": "ONGC",
        "sector": "OILGAS",
        "index": "NIFTY",
        "strong": ["oil and natural gas corporation", "oil and natural gas"],
        "medium": ["ongc"],
        "weak": [],
    },
    "COALINDIA": {
        "name": "Coal India",
        "sector": "METALS",
        "index": "NIFTY",
        "strong": ["coal india"],
        "medium": [],
        "weak": ["cil"],
    },
    "BAJAJFINSV": {
        "name": "Bajaj Finserv",
        "sector": "NBFC",
        "index": "NIFTY",
        "strong": ["bajaj finserv"],
        "medium": [],
        "weak": [],
    },
    "WIPRO": {
        "name": "Wipro",
        "sector": "IT",
        "index": "NIFTY",
        "strong": ["wipro ltd", "wipro limited"],
        "medium": ["wipro"],
        "weak": [],
    },
    "HCLTECH": {
        "name": "HCL Tech",
        "sector": "IT",
        "index": "NIFTY",
        "strong": ["hcl technologies", "hcl tech"],
        "medium": ["hcltech"],
        "weak": ["hcl"],
    },
    "TECHM": {
        "name": "Tech Mahindra",
        "sector": "IT",
        "index": "NIFTY",
        "strong": ["tech mahindra"],
        "medium": ["techm"],
        "weak": [],
    },
    "NESTLEIND": {
        "name": "Nestle India",
        "sector": "FMCG",
        "index": "NIFTY",
        "strong": ["nestle india"],
        "medium": ["nestle", "maggi"],
        "weak": [],
    },
    "INDUSINDBK": {
        "name": "IndusInd Bank",
        "sector": "BANK",
        "index": "BANKNIFTY",
        "strong": ["indusind bank"],
        "medium": ["indusind"],
        "weak": [],
    },
    "JSWSTEEL": {
        "name": "JSW Steel",
        "sector": "METALS",
        "index": "NIFTY",
        "strong": ["jsw steel"],
        "medium": [],
        "weak": [],
    },
    "M&M": {
        "name": "Mahindra & Mahindra",
        "sector": "AUTO",
        "index": "NIFTY",
        "strong": ["mahindra & mahindra", "mahindra and mahindra", "m and m"],
        "medium": ["m&m", "mahindra auto"],
        "weak": ["mahindra"],
        "deny": ["m&m financial", "mahindra finance", "tech mahindra", "mahindra lifespace"],
    },
    "CIPLA": {
        "name": "Cipla",
        "sector": "PHARMA",
        "index": "NIFTY",
        "strong": ["cipla ltd", "cipla limited"],
        "medium": ["cipla"],
        "weak": [],
    },
    "DRREDDY": {
        "name": "Dr Reddy's",
        "sector": "PHARMA",
        "index": "NIFTY",
        "strong": ["dr reddy", "dr. reddy", "dr reddy's laboratories", "dr reddys"],
        "medium": [],
        "weak": [],
    },
    "APOLLOHOSP": {
        "name": "Apollo Hospitals",
        "sector": "PHARMA",
        "index": "NIFTY",
        "strong": ["apollo hospitals", "apollo hospital"],
        "medium": ["apollo"],
        "weak": [],
        "deny": ["apollo tyres", "apollo micro systems"],
    },
    "BPCL": {
        "name": "BPCL",
        "sector": "OILGAS",
        "index": "NIFTY",
        "strong": ["bharat petroleum"],
        "medium": ["bpcl"],
        "weak": [],
    },
    "EICHERMOT": {
        "name": "Eicher Motors",
        "sector": "AUTO",
        "index": "NIFTY",
        "strong": ["eicher motors", "royal enfield"],
        "medium": ["eicher"],
        "weak": [],
    },
    "GRASIM": {
        "name": "Grasim",
        "sector": "CHEMICALS",
        "index": "NIFTY",
        "strong": ["grasim industries"],
        "medium": ["grasim", "birla opus"],
        "weak": [],
    },
    "HDFCLIFE": {
        "name": "HDFC Life",
        "sector": "INSURANCE",
        "index": "NIFTY",
        "strong": ["hdfc life", "hdfc life insurance"],
        "medium": [],
        "weak": [],
    },
    "SBILIFE": {
        "name": "SBI Life",
        "sector": "INSURANCE",
        "index": "NIFTY",
        "strong": ["sbi life", "sbi life insurance"],
        "medium": [],
        "weak": [],
    },
    "DIVISLAB": {
        "name": "Divi's Labs",
        "sector": "PHARMA",
        "index": "NIFTY",
        "strong": ["divi's laboratories", "divis laboratories", "divi's labs", "divis lab", "divi lab"],
        "medium": ["divis", "divi"],
        "weak": [],
    },
    "BAJAJ-AUTO": {
        "name": "Bajaj Auto",
        "sector": "AUTO",
        "index": "NIFTY",
        "strong": ["bajaj auto"],
        "medium": [],
        "weak": [],
    },
    "HEROMOTOCO": {
        "name": "Hero MotoCorp",
        "sector": "AUTO",
        "index": "NIFTY",
        "strong": ["hero motocorp", "hero honda"],
        "medium": ["hero moto"],
        "weak": ["hero"],
        "deny": ["hero electric", "super hero"],
    },
    "BRITANNIA": {
        "name": "Britannia",
        "sector": "FMCG",
        "index": "NIFTY",
        "strong": ["britannia industries"],
        "medium": ["britannia"],
        "weak": [],
    },
    "TATACONSUM": {
        "name": "Tata Consumer",
        "sector": "FMCG",
        "index": "NIFTY",
        "strong": ["tata consumer products", "tata consumer", "tata tea", "tata coffee"],
        "medium": [],
        "weak": [],
    },
    "HINDALCO": {
        "name": "Hindalco",
        "sector": "METALS",
        "index": "NIFTY",
        "strong": ["hindalco industries"],
        "medium": ["hindalco", "novelis"],
        "weak": [],
    },
    "VEDL": {
        "name": "Vedanta",
        "sector": "METALS",
        "index": "NIFTY",
        "strong": ["vedanta ltd", "vedanta limited", "vedanta resources"],
        "medium": ["vedanta"],
        "weak": [],
    },
    "ZOMATO": {
        "name": "Zomato / Eternal",
        "sector": "RETAIL",
        "index": "NIFTY",
        "strong": ["zomato", "zomato limited"],
        "medium": ["blinkit"],
        "weak": [],
    },
    "ETERNAL": {
        "name": "Eternal",
        "sector": "RETAIL",
        "index": "NIFTY",
        "strong": ["eternal ltd", "eternal limited"],
        "medium": [],
        "weak": [],
    },
    "PAYTM": {
        "name": "Paytm",
        "sector": "FINTECH",
        "index": "NIFTY",
        "strong": ["paytm", "one97 communications"],
        "medium": ["one97"],
        "weak": [],
    },
    # Actively traded names outside the core Nifty-50 set — still need board coverage.
    "IDFCFIRSTB": {
        "name": "IDFC First Bank",
        "sector": "BANK",
        "index": "NIFTY",
        "strong": [
            "idfc first bank",
            "idfc first",
            "idfcfirstbank",
            "idfc first bank ltd",
            "idfc first bank limited",
        ],
        # Headlines often shorten to "IDFC Bank" after the merger branding.
        "medium": ["idfc bank", "idfcfirstb"],
        "weak": ["idfc"],
        "deny": ["idfc mutual fund", "idfc amc", "idfc nifty"],
    },
    "INDIGO": {
        "name": "InterGlobe Aviation",
        "sector": "AVIATION",
        "index": "NIFTY",
        "strong": [
            "interglobe aviation",
            "interglobe",
            "indigo airlines",
            "indi go",
            "goindigo",
        ],
        "medium": ["indigo"],
        "weak": [],
    },
    # Newer / re-weighted Nifty 50 names for breadth + linking.
    "BEL": {
        "name": "Bharat Electronics",
        "sector": "DEFENCE",
        "index": "NIFTY",
        "strong": ["bharat electronics", "bharat electronics ltd", "bharat electronics limited"],
        "medium": ["bel ltd", "bel india"],
        "weak": ["bel"],
        "deny": ["belgium", "bell"],
    },
    "SHRIRAMFIN": {
        "name": "Shriram Finance",
        "sector": "NBFC",
        "index": "NIFTY",
        "strong": ["shriram finance", "shriram finance ltd", "shriram finance limited"],
        "medium": ["shriram fin", "shriram transport"],
        "weak": ["shriram"],
    },
    "TRENT": {
        "name": "Trent",
        "sector": "RETAIL",
        "index": "NIFTY",
        "strong": ["trent ltd", "trent limited", "westside", "zudio"],
        "medium": ["trent"],
        "weak": [],
    },
    "JIOFIN": {
        "name": "Jio Financial Services",
        "sector": "FINANCIALS",
        "index": "NIFTY",
        "strong": [
            "jio financial",
            "jio financial services",
            "jiofin",
            "jio finance",
        ],
        "medium": ["jio fin"],
        "weak": [],
        "deny": ["jio cinema", "jio platforms", "reliance jio"],
    },
    "MAXHEALTH": {
        "name": "Max Healthcare",
        "sector": "HEALTHCARE",
        "index": "NIFTY",
        "strong": ["max healthcare", "max healthcare institute", "max hospital"],
        "medium": ["max health"],
        "weak": [],
    },
    "TMPV": {
        "name": "Tata Motors Passenger Vehicles",
        "sector": "AUTO",
        "index": "NIFTY",
        "strong": [
            "tata motors passenger",
            "tata motors passenger vehicles",
            "tmpv",
        ],
        "medium": [],
        "weak": [],
        "deny": ["tata motors finance", "tmf", "tata motors commercial", "tmcv"],
    },
}

# Merge Nifty 100 extras (news/alerts only — breadth stays Nifty 50 weights).
from .universe_nifty100 import NIFTY100_EXTRA  # noqa: E402

for _sym, _meta in NIFTY100_EXTRA.items():
    if _sym not in UNIVERSE:
        UNIVERSE[_sym] = _meta

INDEX_SYMBOLS: frozenset[str] = frozenset({"NIFTY", "BANKNIFTY", "SENSEX", "VIX", "FINNIFTY", "MIDCAP"})


def sector_of(symbol: str) -> str | None:
    meta = UNIVERSE.get(symbol.upper())
    return meta.get("sector") if meta else None


def symbols_in_sector(sector: str) -> list[str]:
    return [sym for sym, meta in UNIVERSE.items() if meta.get("sector") == sector]


def link_tickers(text: str) -> list[str]:
    """Symbols the text names directly. Thin wrapper over the linking layer."""
    from .linking import analyze

    return analyze(text).direct
