# X-Sentiment Price Forecasting Agent

A Python machine-learning pipeline that forecasts an asset's price at
horizons **n+0, n+1, n+2, n+3, n+4 and n+5 days** using **only**
X (Twitter) sentiment and price history.

Asset: **Bitcoin (BTC-USD)**, daily bars, 180 days of history.

The system scrapes X posts via Apify, pulls recent news via Tavily,
stores prices + posts in a SQLite database, engineers sentiment and
price features, and trains per-horizon regressors under an
expanding-window **walk-forward** validation scheme. A simple long/cash
strategy built on the forecasts is backtested out-of-sample and scored
with Sharpe, Sortino and Maximum Drawdown against buy-and-hold.

## Pipeline overview

```
Apify (X posts) ──┐
                  ├──> SQLite (agent.db) ──> VADER sentiment ──> daily aggregates
yfinance (OHLCV) ─┘                                                    │
Tavily (news) ────> news.json (context)                                 v
                                          features (sentiment + price only)
                                                       │
                                          walk-forward Ridge / RandomForest
                                                       │
                        predictions.csv · metrics.json · charts · reports
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your tokens
```

### Tokens (`.env`)

| Variable | Where to get it | Notes |
|---|---|---|
| `APIFY_TOKEN` | https://console.apify.com/account/integrations | Required for X scraping |
| `APIFY_TWITTER_ACTOR_ID` | Apify Store | **Placeholder in `.env.example`** - set it to the Twitter/X search scraper actor you want to run (`user~actor` or hex ID format both work) |
| `TAVILY_API_KEY` | https://app.tavily.com | Required for news ingestion |
| `TWITTER_MAX_ITEMS` | - | Optional cap on posts per scrape (default 3000, cost control) |

`.env` is git-ignored. Never commit real tokens.

## Running the pipeline

Run stages individually:

```bash
python -m src.ingest_prices     # yfinance -> data/prices.csv + agent.db
python -m src.ingest_tweets     # Apify    -> data/tweets.csv  + agent.db
python -m src.ingest_news       # Tavily   -> data/news.json
```

Or everything end-to-end, then build the reports:

```bash
python -m src.run_pipeline      # -> outputs/predictions.csv, metrics.json, charts
python -m src.make_report       # -> reports/evaluation_report.md, sentiment_outlook.md
```

## Outputs

| File | Contents |
|---|---|
| `data/agent.db` | SQLite join of `prices` and `tweets` tables |
| `outputs/predictions.csv` | One row per out-of-sample day: predicted vs actual price per horizon, chosen model, strategy columns |
| `outputs/metrics.json` | Per-horizon accuracy + strategy/benchmark Sharpe, Sortino, Max Drawdown |
| `outputs/price_sentiment.png` | Price vs daily sentiment (dual axis) |
| `outputs/visual_matrix.png` | Visual matrix: OOS RMSE heatmap, walk-forward folds x horizons |
| `outputs/equity_drawdown.png` | Strategy vs buy-and-hold equity curve + drawdown |
| `reports/evaluation_report.md` | Full evaluation report |
| `reports/sentiment_outlook.md` | Data pack for the 1-page sentiment outlook |

## Methodology notes

- **Sentiment:** VADER compound per post, aggregated daily - mean,
  engagement-weighted (likes + reposts + views), follower-weighted,
  post volume, bullish share (compound > 0.05), 3d/7d momentum.
- **Features:** sentiment aggregates + price-only features (lagged log
  returns 1/2/3/5/7d, rolling volatility, momentum, distance from
  14-day MA) + day of week. Nothing else - per the assignment spec.
- **Targets:** forward price levels at n+0..n+5; models are trained on
  forward log returns and prices are recovered as `close * exp(ret)`.
  n+0 is the current price (persistence) by construction.
- **Validation:** expanding-window walk-forward (90-day initial train,
  7-day test blocks). Ridge vs RandomForest selection happens *inside*
  each training window on a time-ordered 80/20 split - test folds are
  never used to select models.
- **Strategy:** long BTC when predicted next-day return > 0, else cash.
  Metrics annualised with 365 days (crypto trades daily).

## Honest limitations

- VADER is a general lexicon model; crypto slang/sarcasm are only
  partially captured. Sentiment is a proxy, not ground truth.
- Scraped X posts are a sample, not the full firehose.
- Daily bars ignore intraday timing of posts.
- The backtest ignores fees, slippage and funding - live results would
  be worse.
- 180 days is a short sample across limited market regimes; treat the
  metrics as indicative, not as proof of a tradable edge.

## Project layout

```
config.py               # paths, tokens (from .env), all tunables
src/
  ingest_prices.py      # yfinance OHLCV -> CSV + SQLite
  ingest_tweets.py      # Apify run/poll/fetch -> normalised tweets
  ingest_news.py        # Tavily search -> news.json
  sentiment.py          # VADER scoring + daily aggregation
  features.py           # feature/target engineering (sentiment+price only)
  model.py              # walk-forward validation + strategy backtest
  charts.py             # price/sentiment, visual matrix, equity/drawdown
  run_pipeline.py       # end-to-end orchestrator
  make_report.py        # evaluation_report.md + sentiment_outlook.md
```
