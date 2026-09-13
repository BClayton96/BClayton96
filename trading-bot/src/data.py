"""Historical and latest price data access via Yahoo Finance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_price_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV history for a ticker.

    Either (start, end) or period ("6mo", "2y", "max", ...) may be given.
    Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Volume.
    """
    if period:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    else:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)

    if df is None or df.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    return df


def fetch_latest_bar(ticker: str) -> pd.Series:
    """Fetch the most recent completed daily bar for a ticker."""
    df = fetch_price_history(ticker, period="5d", interval="1d")
    return df.iloc[-1]
