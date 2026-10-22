"""End-to-end pipeline orchestrator.

Stages (each can also be run on its own - see README):

1. Ingest prices (yfinance)          -> data/prices.csv, agent.db
2. Ingest X posts (Apify)            -> data/tweets.csv, agent.db
3. Ingest news (Tavily)              -> data/news.json
4. Score + aggregate sentiment       -> daily sentiment frame
5. Engineer features (sentiment+price only)
6. Walk-forward training/evaluation  -> outputs/predictions.csv
7. Metrics + charts                  -> outputs/metrics.json, PNGs

Ingestion stages that need API tokens are skipped with a clear warning
when the token (or the data they would produce) is absent, so the
modelling stages can still run against previously ingested data.

Run: ``python -m src.run_pipeline``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src import charts, features as features_mod, ingest_news, ingest_prices
from src import ingest_tweets, model as model_mod, sentiment as sentiment_mod


def stage_prices() -> pd.DataFrame:
    df = ingest_prices.fetch_prices()
    ingest_prices.save_prices(df)
    print(f"[pipeline] prices: {len(df)} bars")
    return df


def stage_tweets() -> pd.DataFrame:
    try:
        ingest_tweets.main()
    except Exception as exc:  # token missing, actor unset, network, ...
        print(f"[pipeline] WARNING: tweet ingestion skipped ({exc})")
    return pd.read_csv(config.TWEETS_CSV)


def stage_news() -> None:
    try:
        ingest_news.main()
    except Exception as exc:
        print(f"[pipeline] WARNING: news ingestion skipped ({exc})")


def main() -> None:
    prices = stage_prices()
    tweets = stage_tweets()
    stage_news()

    daily_sent = sentiment_mod.daily_sentiment(sentiment_mod.score_tweets(tweets))
    print(f"[pipeline] sentiment: {len(daily_sent)} days aggregated")

    feats = features_mod.build_features(prices, daily_sent)
    feats.to_csv(config.DAILY_FEATURES_CSV, index=False)
    frame = features_mod.modelling_frame(feats)
    print(f"[pipeline] modelling frame: {len(frame)} usable rows")

    preds, metrics = model_mod.walk_forward(frame)
    preds.to_csv(config.OUTPUTS_DIR / "predictions.csv", index=False)
    with open(config.OUTPUTS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[pipeline] walk-forward OOS days: {metrics['oos_days']}")

    paths = charts.make_all_charts(feats, preds)
    for p in paths:
        print(f"[pipeline] chart: {p}")
    print("[pipeline] done - run src.make_report to build the reports")


if __name__ == "__main__":
    main()
