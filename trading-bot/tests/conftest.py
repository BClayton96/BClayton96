import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_price_df():
    """Deterministic synthetic OHLCV series: an upward trend plus noise."""
    rng = np.random.default_rng(seed=7)
    n = 400
    dates = pd.bdate_range("2022-01-03", periods=n)

    trend = np.linspace(100, 160, n)
    noise = rng.normal(0, 1.5, n).cumsum() * 0.2
    close = trend + noise
    close = np.maximum(close, 1.0)

    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df
