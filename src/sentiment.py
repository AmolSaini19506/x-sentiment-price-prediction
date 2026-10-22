"""Turn raw X posts into daily sentiment aggregates.

Each post is scored with VADER (``compound`` in [-1, 1]) - a
lexicon-and-rules model that handles social-media text (slang, caps,
emoji, punctuation emphasis) well without any training data.

Posts are then bucketed by UTC calendar day and aggregated into the
daily sentiment features used by the model:

* ``sent_mean``          - plain mean of VADER compound
* ``sent_engagement_w``  - mean weighted by engagement (likes+reposts+views)
* ``sent_follower_w``    - mean weighted by author follower count
* ``post_volume``        - number of posts that day
* ``sent_mom_3d`` / ``sent_mom_7d`` - rolling means of ``sent_mean`` (momentum)
* ``bullish_share``      - share of posts with compound > 0.05

VADER readouts are *proxies* for market sentiment, not ground truth -
see the limitations section of the evaluation report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

BULLISH_THRESHOLD = 0.05  # standard VADER cut-off for "positive"


def score_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    """Add a ``compound`` VADER score to every tweet row."""
    analyzer = SentimentIntensityAnalyzer()
    df = tweets.copy()
    texts = df["text"].fillna("").astype(str)
    df["compound"] = [analyzer.polarity_scores(t)["compound"] for t in texts]
    return df


def daily_sentiment(tweets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored tweets to one row per UTC day."""
    if tweets.empty:
        return pd.DataFrame(
            columns=[
                "date", "sent_mean", "sent_engagement_w", "sent_follower_w",
                "post_volume", "bullish_share",
            ]
        )

    df = tweets.copy()
    df["date"] = pd.to_datetime(df["created_at"], utc=True).dt.date
    df["engagement"] = df["like_count"] + df["retweet_count"] + df["view_count"]

    def _weighted_mean(group: pd.DataFrame, weight_col: str) -> float:
        weights = group[weight_col].astype(float)
        # +1 keeps zero-engagement posts in the average with minimal weight
        # instead of silently dropping them.
        return float(np.average(group["compound"], weights=weights + 1.0))

    rows = []
    for date, group in df.groupby("date"):
        rows.append(
            {
                "date": date,
                "sent_mean": float(group["compound"].mean()),
                "sent_engagement_w": _weighted_mean(group, "engagement"),
                "sent_follower_w": _weighted_mean(group, "author_followers"),
                "post_volume": int(len(group)),
                "bullish_share": float(
                    (group["compound"] > BULLISH_THRESHOLD).mean()
                ),
            }
        )

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # Sentiment momentum: trailing means of the headline sentiment series.
    daily["sent_mom_3d"] = daily["sent_mean"].rolling(3, min_periods=1).mean()
    daily["sent_mom_7d"] = daily["sent_mean"].rolling(7, min_periods=1).mean()
    return daily


def build_daily_sentiment(tweets_csv=None) -> pd.DataFrame:
    """Load tweets from disk, score them, and return daily aggregates."""
    tweets = pd.read_csv(tweets_csv or config.TWEETS_CSV)
    return daily_sentiment(score_tweets(tweets))
