"""Central configuration for the X-sentiment forecasting pipeline.

All secrets and environment-specific settings are read from a local
``.env`` file (see ``.env.example``). Nothing sensitive is hard-coded.

The pipeline stages import :data:`config` values from here so there is a
single place to change tickers, lookback windows, paths and API settings.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (DATA_DIR, OUTPUTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Secrets (loaded from .env)
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

APIFY_TOKEN: str | None = os.getenv("APIFY_TOKEN") or None
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY") or None

# Apify actor used for X (Twitter) search scraping.
# NOTE: placeholder default - set APIFY_TWITTER_ACTOR_ID in .env to the
# real actor you want to run (Apify Store actors use "username~actor-name").
APIFY_TWITTER_ACTOR_ID: str = os.getenv(
    "APIFY_TWITTER_ACTOR_ID", "placeholder~twitter-scraper"
)

# ---------------------------------------------------------------------------
# Asset / market settings
# ---------------------------------------------------------------------------
TICKER = "BTC-USD"          # yfinance symbol for Bitcoin
ASSET_NAME = "Bitcoin"
LOOKBACK_DAYS = 180          # how much history to ingest (spec: 180 days)

# ---------------------------------------------------------------------------
# X (Twitter) scraping settings
# ---------------------------------------------------------------------------
# Search terms sent to the Apify actor. Cashtag + plain keyword cover both
# trader chatter ($BTC) and general discussion (Bitcoin).
TWITTER_SEARCH_TERMS = ["$BTC", "Bitcoin"]
# Hard cap on posts per Apify run (cost control). Configurable via env.
TWITTER_MAX_ITEMS = int(os.getenv("TWITTER_MAX_ITEMS", "3000"))

# ---------------------------------------------------------------------------
# News settings (Tavily)
# ---------------------------------------------------------------------------
NEWS_QUERY = "Bitcoin BTC price market news"
NEWS_MAX_RESULTS = 25

# ---------------------------------------------------------------------------
# Modelling settings
# ---------------------------------------------------------------------------
HORIZONS = [0, 1, 2, 3, 4, 5]        # forecast price at n+0 ... n+5 days
WF_INITIAL_TRAIN_DAYS = 90           # walk-forward: first training window
WF_TEST_STEP_DAYS = 7                # walk-forward: test block / step size
RANDOM_STATE = 42

# Files
PRICES_CSV = DATA_DIR / "prices.csv"
TWEETS_RAW_JSON = DATA_DIR / "tweets_raw.json"
TWEETS_CSV = DATA_DIR / "tweets.csv"
NEWS_JSON = DATA_DIR / "news.json"
DB_PATH = DATA_DIR / "agent.db"
DAILY_FEATURES_CSV = DATA_DIR / "daily_features.csv"
