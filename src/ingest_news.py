"""Ingest recent news about the asset via the Tavily search API.

News complements the X sentiment signal: it gives context for *why*
sentiment moved and feeds the qualitative sentiment-outlook summary.
Results are stored verbatim in ``data/news.json``.

Requires ``TAVILY_API_KEY`` in ``.env``.
Run directly: ``python -m src.ingest_news``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def fetch_news(
    query: str = config.NEWS_QUERY,
    max_results: int = config.NEWS_MAX_RESULTS,
) -> list[dict]:
    """Run a Tavily search and return the raw result list."""
    if not config.TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    payload = {
        "api_key": config.TAVILY_API_KEY,
        "query": query,
        "topic": "finance",
        "search_depth": "advanced",
        "days": 30,  # recent news only - sentiment context, not an archive
        "max_results": max_results,
        "include_answer": False,
    }
    resp = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json().get("results", [])


def main() -> None:
    results = fetch_news()
    with open(config.NEWS_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"[news] {len(results)} articles -> {config.NEWS_JSON}")


if __name__ == "__main__":
    main()
