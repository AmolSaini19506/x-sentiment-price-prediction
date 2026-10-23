"""Ingest X (Twitter) posts about the asset via Apify.

Flow (Apify REST API):
1. Start a run of the configured Twitter-scraper actor with search terms
   covering the last ``LOOKBACK_DAYS`` days.
2. Poll the run until it reaches a terminal state.
3. Download the run's default dataset and normalise each post to a small,
   stable schema so downstream stages never depend on the actor's raw
   field names:

   ``created_at`` (UTC timestamp), ``text``, ``like_count``,
   ``retweet_count``, ``view_count``, ``author_followers``

Outputs: ``data/tweets_raw.json`` (untouched dataset items),
``data/tweets.csv`` (normalised) and SQLite table ``tweets`` in
``data/agent.db``.

Requires ``APIFY_TOKEN`` and a real ``APIFY_TWITTER_ACTOR_ID`` in ``.env``.
Run directly: ``python -m src.ingest_tweets``
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

APIFY_API_BASE = "https://api.apify.com/v2"
POLL_INTERVAL_SECONDS = 10
RUN_TIMEOUT_SECONDS = 60 * 30  # give large scrapes up to 30 minutes


class ApifyError(RuntimeError):
    """Raised when an Apify run fails or times out."""


def _auth_headers() -> dict[str, str]:
    if not config.APIFY_TOKEN:
        raise ApifyError(
            "APIFY_TOKEN is not set. Copy .env.example to .env and add your token."
        )
    return {"Authorization": f"Bearer {config.APIFY_TOKEN}"}


def start_run(since: datetime, until: datetime, max_items: int, sort: str = "Top") -> str:
    """Start the Twitter actor run and return its run ID."""
    actor_id = config.APIFY_TWITTER_ACTOR_ID.replace("/", "~")
    url = f"{APIFY_API_BASE}/acts/{actor_id}/runs"

    # Input keys verified against the apidojo/tweet-scraper build schema:
    # the date range is "start"/"end" (an earlier "since"/"until" payload
    # was silently ignored by the actor, which then scraped only the
    # newest posts).
    payload = {
        "searchTerms": config.TWITTER_SEARCH_TERMS,
        "start": since.strftime("%Y-%m-%d"),
        "end": until.strftime("%Y-%m-%d"),
        "maxItems": max_items,
        "sort": sort,
        "tweetLanguage": "en",
    }
    resp = requests.post(url, json=payload, headers=_auth_headers(), timeout=60)
    resp.raise_for_status()
    return resp.json()["data"]["id"]


def wait_for_run(run_id: str) -> str:
    """Poll a run until it succeeds; return its default dataset ID."""
    url = f"{APIFY_API_BASE}/actor-runs/{run_id}"
    deadline = time.time() + RUN_TIMEOUT_SECONDS
    while time.time() < deadline:
        resp = requests.get(url, headers=_auth_headers(), timeout=60)
        resp.raise_for_status()
        data = resp.json()["data"]
        status = data["status"]
        if status == "SUCCEEDED":
            return data["defaultDatasetId"]
        if status in {"FAILED", "ABORTED", "TIMED-OUT"}:
            raise ApifyError(f"Apify run {run_id} ended with status {status}")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise ApifyError(f"Apify run {run_id} did not finish within timeout")


def fetch_dataset(dataset_id: str) -> list[dict[str, Any]]:
    """Download all items from an Apify dataset."""
    url = f"{APIFY_API_BASE}/datasets/{dataset_id}/items"
    resp = requests.get(
        url, headers=_auth_headers(), params={"clean": "true"}, timeout=300
    )
    resp.raise_for_status()
    return resp.json()


def _first(item: dict[str, Any], *keys: str, default=None):
    """Return the first present, non-null value among candidate keys.

    Different Twitter actors name the same field differently; this helper
    keeps the normaliser tolerant without hiding the canonical schema.
    """
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return default


def normalize(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Map raw actor items onto the pipeline's canonical tweet schema."""
    rows = []
    for item in items:
        author = item.get("author") or {}
        created = _first(item, "createdAt", "created_at", "date", "time")
        if created is None:
            continue  # a post without a timestamp cannot be aligned to prices
        rows.append(
            {
                "created_at": created,
                "text": _first(item, "text", "fullText", "full_text", default=""),
                "like_count": _first(item, "likeCount", "favorite_count", "likes", default=0),
                "retweet_count": _first(item, "retweetCount", "retweet_count", default=0),
                "view_count": _first(item, "viewCount", "views", "impressions", default=0),
                "author_followers": _first(
                    author,
                    "followersCount",
                    "followers_count",
                    "followers",
                    default=_first(item, "authorFollowers", default=0),
                ),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # All timestamps are normalised to UTC - price bars are daily UTC too.
    # format="mixed": actors emit several timestamp spellings; without it
    # pandas infers one format and silently coerces the rest to NaT.
    df["created_at"] = pd.to_datetime(
        df["created_at"], utc=True, errors="coerce", format="mixed"
    )
    df = df.dropna(subset=["created_at"])
    for col in ["like_count", "retweet_count", "view_count", "author_followers"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def save_tweets(raw_items: list[dict[str, Any]], df: pd.DataFrame) -> None:
    if df.empty:
        # Never clobber previously ingested data with an empty result
        # (e.g. the actor returned only noResults markers because the
        # account hit a plan limit). Raising lets run_pipeline fall
        # back to the existing tweets.csv, per its documented design.
        raise ApifyError(
            "ingestion produced 0 usable posts; keeping existing data"
        )
    with open(config.TWEETS_RAW_JSON, "w", encoding="utf-8") as fh:
        json.dump(raw_items, fh, ensure_ascii=False)
    df.to_csv(config.TWEETS_CSV, index=False)
    with sqlite3.connect(config.DB_PATH) as conn:
        df.to_sql("tweets", conn, if_exists="replace", index=False)


# The apidojo actors cap FREE-plan Apify accounts at 10 items per run
# (stated verbatim in the actor log), so a single 180-day run can never
# yield a usable sample. The lookback is therefore scraped in
# WINDOW_DAYS-sized windows, one run each, and the results are merged
# and de-duplicated. "Top" sort spreads each window's items across most
# of its days (verified empirically; "Latest" returns one day's posts).
WINDOW_DAYS = 10
FREE_PLAN_ITEMS_PER_RUN = 10


def main() -> None:
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=config.LOOKBACK_DAYS)

    print(f"[tweets] starting Apify actor {config.APIFY_TWITTER_ACTOR_ID} ...")
    all_items: list[dict[str, Any]] = []
    seen_ids: set = set()
    window_start = since
    while window_start < until:
        window_end = min(window_start + timedelta(days=WINDOW_DAYS), until)
        run_id = start_run(window_start, window_end, FREE_PLAN_ITEMS_PER_RUN)
        print(
            f"[tweets] run {run_id} for "
            f"{window_start.date()}..{window_end.date()}; polling ..."
        )
        dataset_id = wait_for_run(run_id)
        items = fetch_dataset(dataset_id)
        fresh = 0
        for item in items:
            key = item.get("id") or item.get("url")
            if key is not None:
                if key in seen_ids:
                    continue
                seen_ids.add(key)
            all_items.append(item)
            fresh += 1
        print(f"[tweets]   +{fresh} items ({len(all_items)} total)")
        window_start = window_end

    df = normalize(all_items)
    save_tweets(all_items, df)
    print(f"[tweets] {len(all_items)} raw posts -> {len(df)} normalised rows")
    print(f"[tweets] wrote {config.TWEETS_CSV} and table 'tweets' in {config.DB_PATH}")


if __name__ == "__main__":
    main()
