import os
from unittest.mock import patch

import pytest

from src.data import _parse_period_to_days, fetch_price_history


@pytest.mark.parametrize(
    "period,expected_days",
    [
        ("5d", 5),
        ("6mo", 180),
        ("2y", 730),
        ("5y", 1825),
        ("max", 20 * 365),
    ],
)
def test_parse_period_to_days(period, expected_days):
    assert _parse_period_to_days(period) == expected_days


def test_parse_period_rejects_unrecognized_string():
    with pytest.raises(ValueError):
        _parse_period_to_days("banana")


def test_fetch_price_history_requires_credentials():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="ALPACA_API_KEY_ID"):
            fetch_price_history("AAPL", period="1y")
