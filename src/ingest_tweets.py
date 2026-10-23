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

    # Input keys for kaitoeasyapi/twitter-x-data-tweet-scraper (pay-per-
    # result; verified live 2026-10-23): a single raw search string in
    # "twitterContent" plus "since"/"until" date bounds. The apidojo
    # actors used before cap free-plan accounts at 10 items per run,
    # which made a usable sample impossible; this actor returned 300+
    # items in a single run on the same free account.
    query = " OR ".join([*config.TWITTER_SEARCH_TERMS, "BTC", "bitcoin"])
    payload = {
        "twitterContent": query,
        "since": since.strftime("%Y-%m-%d"),
        "until": until.strftime("%Y-%m-%d"),
        "maxItems": max_items,
        "queryType": sort,
        "lang": "en",
    }
    resp = requests.post(url, json=payload, headers=_auth_headers(), timeout=60)
    resp.raise_for_status()
    return resp.json()["data"]["id"]


def wait_for_run(run_id: str) -> tuple[str, float]:
    """Poll a run until it succeeds; return (dataset ID, usage in USD)."""
    url = f"{APIFY_API_BASE}/actor-runs/{run_id}"
    deadline = time.time() + RUN_TIMEOUT_SECONDS
    while time.time() < deadline:
        resp = requests.get(url, headers=_auth_headers(), timeout=60)
        resp.raise_for_status()
        data = resp.json()["data"]
        status = data["status"]
        if status == "SUCCEEDED":
            return data["defaultDatasetId"], float(data.get("usageTotalUsd") or 0.0)
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


# Scrape geometry for the pay-per-result actor (2026-10-23 run):
# * The window is anchored to the last available price bar so the tweet
#   sample is a contiguous span ending with the price history, not a
#   moving "now" window.
# * 3-day windows with "Top" sort: one run returns ~300+ posts spread
#   across the window's days at roughly $0.00023/post all-in, so a
#   132-day span costs about $3.50 of the $5 free-plan credit.
# * SPEND_GUARD_USD is a hard stop: no new window is started once the
#   cumulative scrape spend passes it (the free plan has no headroom
#   beyond $5 and no payment method is attached).
SCRAPE_DAYS = 132
WINDOW_DAYS = 3
ITEMS_PER_RUN = 350
MAX_WORKERS = 4
SPEND_GUARD_USD = 4.15


def _target_window() -> tuple[datetime, datetime]:
    """Contiguous scrape span [start, end) ending at the last price bar."""
    prices = pd.read_csv(config.PRICES_CSV, usecols=["date"])
    last_bar = pd.to_datetime(prices["date"]).max().date()
    end = datetime(last_bar.year, last_bar.month, last_bar.day, tzinfo=timezone.utc)
    end += timedelta(days=1)  # 'until' is exclusive; include the last bar's day
    return end - timedelta(days=SCRAPE_DAYS), end


def _existing_covers(start: datetime, end: datetime) -> bool:
    """True when tweets.csv already holds a full scrape of the window.

    Makes re-running the pipeline (scripts/run_all.sh) cheap: the tweet
    stage becomes a no-op instead of paying for the same scrape twice.
    """
    if not config.TWEETS_CSV.exists():
        return False
    try:
        df = pd.read_csv(config.TWEETS_CSV, usecols=["created_at"])
    except Exception:
        return False
    if len(df) < 5000:
        return False
    ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return False
    return ts.min().date() <= (start + timedelta(days=2)).date() and \
        ts.max().date() >= (end - timedelta(days=2)).date()


def main() -> None:
    since, until = _target_window()
    if _existing_covers(since, until):
        print("[tweets] existing tweets.csv already covers "
              f"{since.date()}..{until.date()}; skipping scrape")
        return

    windows = []
    w = since
    while w < until:
        windows.append((w, min(w + timedelta(days=WINDOW_DAYS), until)))
        w += timedelta(days=WINDOW_DAYS)

    print(f"[tweets] starting Apify actor {config.APIFY_TWITTER_ACTOR_ID} ...")
    print(f"[tweets] {len(windows)} windows of {WINDOW_DAYS}d, "
          f"{since.date()}..{until.date()}, up to {ITEMS_PER_RUN} posts each")

    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    state = {"spend": 0.0, "items": [], "seen": set(), "done": 0, "stopped": False}

    def _scrape(window: tuple[datetime, datetime]) -> None:
        w_start, w_end = window
        with lock:
            if state["stopped"]:
                return
        run_id = start_run(w_start, w_end, ITEMS_PER_RUN)
        dataset_id, cost = wait_for_run(run_id)
        items = fetch_dataset(dataset_id)
        with lock:
            state["spend"] += cost
            state["done"] += 1
            fresh = 0
            for item in items:
                key = item.get("id") or item.get("url")
                if key is not None:
                    if key in state["seen"]:
                        continue
                    state["seen"].add(key)
                state["items"].append(item)
                fresh += 1
            print(f"[tweets] window {w_start.date()}..{w_end.date()}: "
                  f"+{fresh} items, ${cost:.3f} "
                  f"({state['done']}/{len(windows)} windows, "
                  f"${state['spend']:.2f} total)")
            if state["spend"] >= SPEND_GUARD_USD:
                state["stopped"] = True
                print(f"[tweets] SPEND GUARD hit at ${state['spend']:.2f}; "
                      "no further windows will be started")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(_scrape, windows))

    all_items = state["items"]
    print(f"[tweets] scrape finished: {len(all_items)} raw posts, "
          f"${state['spend']:.2f} Apify usage")
    df = normalize(all_items)
    save_tweets(all_items, df)
    print(f"[tweets] {len(all_items)} raw posts -> {len(df)} normalised rows")
    print(f"[tweets] wrote {config.TWEETS_CSV} and table 'tweets' in {config.DB_PATH}")


if __name__ == "__main__":
    main()
