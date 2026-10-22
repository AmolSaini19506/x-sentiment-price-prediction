"""Build the modelling table: features from sentiment + price ONLY.

Design constraints (from the assignment spec):

* Features may use **only** X-sentiment-derived aggregates and price
  history. No on-chain data, no order-book data, no macro series.
* Every feature at day *t* uses information available **at or before**
  day *t*'s close - the features are shifted so a forecast for n+h is
  made with day-t features, never with same-day or future sentiment.
* Targets are the price *levels* at n+0..n+5 (per spec) and, derived
  from them, forward returns per horizon (used for model selection and
  for the trading strategy, where returns are the meaningful quantity).

Price features: lagged log returns (1/2/3/5/7d), rolling volatility
(7d/14d), price momentum (7d/14d cumulative return), distance from the
14-day moving average.

Sentiment features: the daily aggregates from :mod:`src.sentiment`
(level, engagement- and follower-weighted levels, volume + its 7-day
change, momentum 3d/7d, bullish share), plus 1-day lags of the level and
volume so the model sees sentiment *changes*, not just levels.

Calendar features (day of week) are allowed and included.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SENTIMENT_COLUMNS = [
    "sent_mean",
    "sent_engagement_w",
    "sent_follower_w",
    "post_volume",
    "bullish_share",
    "sent_mom_3d",
    "sent_mom_7d",
]

FEATURE_COLUMNS = [
    # price history
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_7d",
    "vol_7d", "vol_14d",
    "mom_7d", "mom_14d",
    "dist_ma14",
    # sentiment (same-day aggregates describe day t; used to forecast t+h)
    *SENTIMENT_COLUMNS,
    "sent_mean_lag1", "sent_chg_1d", "volume_chg_7d",
    # calendar
    "dow",
]


def build_features(prices: pd.DataFrame, daily_sent: pd.DataFrame) -> pd.DataFrame:
    """Join prices with daily sentiment and engineer features + targets."""
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)

    sent = daily_sent.copy()
    if not sent.empty:
        sent["date"] = pd.to_datetime(sent["date"]).dt.date
    df = df.merge(sent, on="date", how="left")

    # Days with no scraped posts are genuinely missing sentiment - fill
    # level-type features with neutral 0 and counts with 0, so the model
    # can still use the price features for those rows.
    for col in ["sent_mean", "sent_engagement_w", "sent_follower_w",
                "bullish_share", "sent_mom_3d", "sent_mom_7d"]:
        if col in df:
            df[col] = df[col].fillna(0.0)
    if "post_volume" in df:
        df["post_volume"] = df["post_volume"].fillna(0)

    close = df["close"].astype(float)
    log_close = np.log(close)

    # --- price-only features ------------------------------------------
    for lag in (1, 2, 3, 5, 7):
        df[f"ret_{lag}d"] = log_close.diff(lag)
    daily_ret = log_close.diff(1)
    df["vol_7d"] = daily_ret.rolling(7).std()
    df["vol_14d"] = daily_ret.rolling(14).std()
    df["mom_7d"] = log_close.diff(7)
    df["mom_14d"] = log_close.diff(14)
    df["dist_ma14"] = close / close.rolling(14).mean() - 1.0

    # --- sentiment dynamics -------------------------------------------
    df["sent_mean_lag1"] = df["sent_mean"].shift(1)
    df["sent_chg_1d"] = df["sent_mean"] - df["sent_mean_lag1"]
    df["volume_chg_7d"] = df["post_volume"] / df["post_volume"].rolling(7).mean() - 1.0

    # --- calendar ------------------------------------------------------
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek

    # --- targets: price levels n+0..n+5 and forward returns ------------
    for h in config.HORIZONS:
        df[f"target_price_n{h}"] = close.shift(-h)
        df[f"target_ret_n{h}"] = log_close.shift(-h) - log_close

    return df


def modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be used for training/evaluation.

    A row is usable when all features exist (needs ~14 days of history)
    and the longest-horizon target exists (needs 5 days of future data).
    """
    max_h = max(config.HORIZONS)
    return df.dropna(subset=FEATURE_COLUMNS + [f"target_ret_n{max_h}"]).reset_index(drop=True)
