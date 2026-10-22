"""Ingest historical price data for the chosen asset.

Downloads daily OHLCV bars for BTC-USD from Yahoo Finance (via yfinance)
for the configured lookback window, normalises them to a clean daily
frame indexed by UTC date, and persists:

* ``data/prices.csv`` - flat file for quick inspection, and
* ``data/agent.db``   - SQLite table ``prices`` (the shared database the
  tweets table joins against in later stages).

If Yahoo Finance is unreachable or rate-limits the request (observed
from some cloud egress IPs), the fetch transparently falls back to
CoinGecko's public market_chart API (daily close + volume; open/high/low
are set to the close for those rows, which the downstream features do
not use beyond close/volume). A ``source`` column records where each
frame came from.

Run directly: ``python -m src.ingest_prices``
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# CoinGecko coin id per ticker, for the fallback source.
_COINGECKO_IDS = {"BTC-USD": "bitcoin", "ETH-USD": "ethereum"}


def _fetch_yahoo(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
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
    df["source"] = "yahoo"
    return df


def _fetch_coingecko(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Fallback: daily close + volume from CoinGecko's public API."""
    coin_id = _COINGECKO_IDS.get(ticker)
    if coin_id is None:
        raise RuntimeError(f"no CoinGecko fallback mapping for {ticker}")
    resp = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": lookback_days, "interval": "daily"},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    prices = pd.DataFrame(payload["prices"], columns=["ts_ms", "close"])
    volumes = pd.DataFrame(payload["total_volumes"], columns=["ts_ms", "volume"])
    df = prices.merge(volumes, on="ts_ms", how="left")
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.date
    df = df.drop(columns=["ts_ms"]).drop_duplicates("date", keep="last")
    # market_chart provides closes only; OHLC features are not used
    # downstream beyond close/volume, so mirror the close.
    df["open"] = df["high"] = df["low"] = df["close"]
    df["source"] = "coingecko"
    return df[["date", "open", "high", "low", "close", "volume", "source"]]


def fetch_prices(
    ticker: str = config.TICKER,
    lookback_days: int = config.LOOKBACK_DAYS,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV bars and return a normalised DataFrame.

    Columns: ``date`` (datetime.date, UTC), ``open``, ``high``, ``low``,
    ``close``, ``volume``, ``source``. Rows are sorted ascending by date.
    Primary source is Yahoo Finance; falls back to CoinGecko when Yahoo
    is unreachable or rate-limited.
    """
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    try:
        df = _fetch_yahoo(ticker, start, end)
    except Exception as yahoo_err:  # noqa: BLE001 - deliberate fallback
        print(f"[prices] Yahoo fetch failed ({yahoo_err}); trying CoinGecko")
        df = _fetch_coingecko(ticker, lookback_days)
        df = df[df["date"] >= start.date()]

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
