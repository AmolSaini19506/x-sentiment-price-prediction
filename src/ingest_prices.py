"""Ingest historical price data for the chosen asset.

Downloads daily OHLCV bars for BTC-USD from Yahoo Finance (via yfinance)
for the configured lookback window, normalises them to a clean daily
frame indexed by UTC date, and persists:

* ``data/prices.csv`` - flat file for quick inspection, and
* ``data/agent.db``   - SQLite table ``prices`` (the shared database the
  tweets table joins against in later stages).

Run directly: ``python -m src.ingest_prices``
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def fetch_prices(
    ticker: str = config.TICKER,
    lookback_days: int = config.LOOKBACK_DAYS,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV bars and return a normalised DataFrame.

    Columns: ``date`` (datetime.date, UTC), ``open``, ``high``, ``low``,
    ``close``, ``volume``. Rows are sorted ascending by date.
    """
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # yfinance may return MultiIndex columns for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.date
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_prices(df: pd.DataFrame) -> None:
    """Persist prices to CSV and to the SQLite database."""
    df.to_csv(config.PRICES_CSV, index=False)

    with sqlite3.connect(config.DB_PATH) as conn:
        df.to_sql("prices", conn, if_exists="replace", index=False)


def main() -> None:
    df = fetch_prices()
    save_prices(df)
    print(
        f"[prices] {len(df)} daily bars for {config.TICKER}: "
        f"{df['date'].iloc[0]} -> {df['date'].iloc[-1]}"
    )
    print(f"[prices] wrote {config.PRICES_CSV} and table 'prices' in {config.DB_PATH}")


if __name__ == "__main__":
    main()
