from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _BACKEND_DIR / ".env"


class Settings(BaseSettings):
    # Absolute .env path so keys load even if uvicorn isn't started from backend/
    model_config = SettingsConfigDict(
        env_prefix="SAINT_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Absolute or relative path to the NSE parquet cache (Yahoo tickers like RELIANCE.NS.parquet)
    price_cache: Path = Path(
        "/Users/akashcs/Chartink/Chartink conditions/New Folder With Items/data/price_cache"
    )
    # How long (seconds) to cache Yahoo quote snapshots in memory
    quote_ttl_s: int = 120
    # Near cash open (09:15–09:45) we poll quotes much harder for gap validation
    quote_ttl_open_s: int = 30
    # How long 15m parquet stays "fresh" before a stock-page open re-pulls Yahoo
    intraday_ttl_s: int = 900
    # Near cash open, force a Yahoo 15m re-pull more often
    intraday_ttl_open_s: int = 60
    # Dashboard response cache — stale-while-revalidate serves older boards
    # instantly and refreshes Yahoo/news in the background.
    dashboard_ttl_s: int = 120
    dashboard_ttl_open_s: int = 45
    # Move size (%%) that counts as "already reacted"
    reaction_threshold_pct: float = 1.0
    # Floor for a story to promote a stock onto the session board
    board_min_relevance: float = 0.35
    board_min_credibility: float = 0.55
    # SQLite file for prediction snapshots / verification history
    predictions_db: Path = Path(__file__).resolve().parent.parent / "data" / "predictions.sqlite3"
    # Optional India FinBERT for direct company headlines (off by default)
    use_finbert: bool = False
    # Allow browser calls from the Vite app
    cors_origins: str = "http://127.0.0.1:3000,http://localhost:3000"
    # AI helper provider: gemini (default) | openai
    ai_provider: str = "gemini"
    # Google AI Studio / Gemini API (preferred for freemium)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    # OpenAI — optional fallback
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Telegram alerts (high-bar board → phone push)
    alerts_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    # Optional shared secret for /api/alerts/* (UptimeRobot query ?key=...)
    alerts_secret: str = ""
    alerts_min_conviction: int = 60
    alerts_max_news_age_mins: int = 360
    # Hard technical gate is off by default — Saint board decides; tech/AI are commentary.
    alerts_require_technicals: bool = False
    # Best-effort Gemini/OpenAI blurb on ENTRY; never blocks the Telegram send.
    alerts_ai_comment: bool = True
    alerts_db: Path = Path(__file__).resolve().parent.parent / "data" / "alerts.sqlite3"


settings = Settings()
