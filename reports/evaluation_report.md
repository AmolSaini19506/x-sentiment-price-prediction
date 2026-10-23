# Evaluation Report - X Sentiment Price Forecasting (Bitcoin)

## 1. Objective
Predict Bitcoin's (BTC-USD) price at horizons n+0, n+1, n+2, n+3, n+4 and n+5 days using **only** X (Twitter) sentiment features and price history, and evaluate the forecasts with rigorous out-of-sample walk-forward validation.

## 2. Data
- **Prices:** daily OHLCV for BTC-USD, last 180 days, fetched via the CoinGecko API. Yahoo Finance (via yfinance) is the primary source in code but rate-limits this machine, so this run used the CoinGecko fallback; the `source` column in `data/prices.csv` records CoinGecko for every row, and OHLC mirrors the close for those rows.
- **X posts:** 1,858 unique posts scraped with Apify between 2026-06-13 and 2026-10-23 (search terms $BTC, Bitcoin), timestamps aligned to UTC days; engagement (likes, reposts, views) and author follower counts retained. The main scrape used the actor `kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest`; an initial attempt with `apidojo/tweet-scraper` was capped by Apify free-plan limits and contributed only a small early sample.
- **News:** recent Bitcoin coverage via the Tavily API, used as qualitative context for sentiment moves.
- Prices and posts are joined in a SQLite database (`data/agent.db`, tables `prices` and `tweets`).

## 3. Features (sentiment + price only)
- **Sentiment (VADER):** daily mean compound, engagement-weighted mean, follower-weighted mean, post volume, bullish share (compound > 0.05), 3d/7d sentiment momentum, 1-day sentiment change, 7-day volume change.
- **Price:** lagged log returns (1/2/3/5/7d), rolling volatility (7d/14d), momentum (7d/14d), distance from the 14-day moving average.
- **Calendar:** day of week. No other data sources are used.

## 4. Validation design
- Expanding-window walk-forward: initial training window 90 days, test blocks of 7 days, refit every block.
- One regressor per horizon, trained on forward log returns; price predictions are `close * exp(pred_return)`.
- Ridge vs RandomForest is selected **inside** each training window using a time-ordered 80/20 split, then refit on the full window. Test data is never used for model selection.
- Out-of-sample period: 2026-09-11 to 2026-10-18 (38 days).

## 5. Forecast accuracy (out-of-sample)

| Horizon | RMSE (price, USD) | MAE (price, USD) | Directional accuracy |
|---|---|---|---|
| n+0 | 0.00 | 0.00 | n/a (persistence) |
| n+1 | 1632.68 | 1157.96 | 50.0% |
| n+2 | 2299.99 | 1745.37 | 44.7% |
| n+3 | 3103.30 | 2409.14 | 39.5% |
| n+4 | 3838.41 | 2907.84 | 44.7% |
| n+5 | 4694.39 | 3562.61 | 28.9% |

Candidate comparison (OOS RMSE of returns, both models refit per fold; selection itself used only in-window validation):

| Horizon | Ridge | RandomForest |
|---|---|---|
| n+0 | n/a | n/a |
| n+1 | 0.0196 | 0.0188 |
| n+2 | 0.0265 | 0.0261 |
| n+3 | 0.0351 | 0.0354 |
| n+4 | 0.0462 | 0.0446 |
| n+5 | 0.0520 | 0.0537 |

## 6. Strategy backtest (out-of-sample)
Rule: hold BTC when the predicted next-day return is positive, hold cash otherwise. No leverage, no shorting, no transaction costs (see limitations).

| Metric | Sentiment strategy | Buy and hold |
|---|---|---|
| Total return | 11.20% | 7.14% |
| Sharpe ratio (ann.) | 3.26 | 2.00 |
| Sortino ratio (ann.) | 6.37 | 3.74 |
| Max drawdown | -6.01% | -5.99% |
| Volatility (ann.) | 32.96% | 36.25% |
| Days | 38 | 38 |

Charts: `outputs/price_sentiment.png` (price vs sentiment), `outputs/visual_matrix.png` (RMSE matrix by fold x horizon), `outputs/equity_drawdown.png` (equity curve and drawdown).

## 7. Limitations
- VADER is a general-purpose lexicon model; crypto slang and sarcasm are only partially captured. Sentiment is a proxy, not ground truth.
- Scraped X data is a sample, not the full firehose; platform sampling and deleted posts bias the sentiment series.
- Daily bars hide intraday timing: a post late in the UTC day is treated the same as one early in the day.
- The backtest ignores fees, slippage and funding; live results would be worse.
- 180 days is a short sample spanning limited market regimes; metrics should be read as indicative, not as proof of edge.
