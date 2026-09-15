"""Technical-indicator feature engineering and label construction.

All features for row t are computed only from data available up to and
including day t, so they are safe to use for a decision made at the close
of day t (to be acted on at the next available price) without leaking
future information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "sma_10_ratio",
    "sma_50_ratio",
    "ema_12_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "volatility_10",
    "momentum_10",
    "volume_change",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with technical-indicator feature columns added."""
    out = df.copy()
    close = out["Close"]

    out["return_1d"] = close.pct_change()
    sma_10 = close.rolling(10).mean()
    sma_50 = close.rolling(50).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    out["sma_10_ratio"] = close / sma_10 - 1
    out["sma_50_ratio"] = close / sma_50 - 1
    out["ema_12_ratio"] = close / ema_12 - 1
    out["rsi_14"] = _rsi(close, 14)

    macd = ema_12 - ema_26
    out["macd"] = macd
    out["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

    out["volatility_10"] = out["return_1d"].rolling(10).std()
    out["momentum_10"] = close / close.shift(10) - 1
    out["volume_change"] = out["Volume"].pct_change()

    return out


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary `target` column: 1 if next day's close is higher, else 0."""
    out = df.copy()
    out["target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    return out


def build_dataset(df: pd.DataFrame, with_labels: bool = True) -> pd.DataFrame:
    """Full pipeline: add features (and optionally labels), drop warm-up NaNs."""
    out = add_features(df)
    if with_labels:
        out = add_labels(out)
        out = out.dropna(subset=FEATURE_COLUMNS + ["target"])
    else:
        out = out.dropna(subset=FEATURE_COLUMNS)
    return out
