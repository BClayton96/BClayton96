"""Historical and latest price data via Alpaca's market data API.

Uses the same ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY credentials as
live-trade — Alpaca's market data endpoints require authentication even
for a free/paper account, but no separate account or key is needed since
`live-trade` already requires an Alpaca account.
"""

from __future__ import annotations

import os

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

_PERIOD_UNITS = {"d": 1, "mo": 30, "y": 365}


def _parse_period_to_days(period: str) -> int:
    """Turn a yfinance-style period string ("6mo", "5y", "max", ...) into a
    number of calendar days to look back."""
    if period == "max":
        return 20 * 365
    for suffix, days_per_unit in _PERIOD_UNITS.items():
        if period.endswith(suffix):
            amount = int(period[: -len(suffix)])
            return amount * days_per_unit
    raise ValueError(f"Unrecognized period '{period}'; expected e.g. '5d', '6mo', '5y', 'max'")


def _get_data_client() -> StockHistoricalDataClient:
    api_key = os.environ.get("ALPACA_API_KEY_ID")
    api_secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError(
            "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables "
            "first (see README.md) — historical price data now comes from Alpaca's "
            "market data API, which requires authentication even for paper accounts."
        )
    return StockHistoricalDataClient(api_key, api_secret)


def fetch_price_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch daily OHLCV history for a ticker from Alpaca.

    Either (start, end) or period ("6mo", "2y", "5y", "max", ...) may be
    given. Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Volume.
    """
    if interval != "1d":
        raise NotImplementedError("Only daily ('1d') bars are supported")

    if period:
        start_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=_parse_period_to_days(period))
        end_ts = None
    else:
        start_ts = pd.Timestamp(start, tz="UTC") if start else None
        end_ts = pd.Timestamp(end, tz="UTC") if end else None

    client = _get_data_client()
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start_ts,
        end=end_ts,
    )
    bars = client.get_stock_bars(request).df

    if bars is None or bars.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'")

    bars = bars.reset_index(level="symbol", drop=True)
    df = bars.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index.date)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].dropna()
    return df


def fetch_latest_bar(ticker: str) -> pd.Series:
    """Fetch the most recent completed daily bar for a ticker."""
    df = fetch_price_history(ticker, period="5d", interval="1d")
    return df.iloc[-1]
