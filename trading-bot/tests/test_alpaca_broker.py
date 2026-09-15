"""Tests for AlpacaBroker's order-sizing and budget-enforcement logic.

No network access or real API keys are used: alpaca.trading.client.TradingClient
is replaced with a mock, so these only verify our own sizing/guard logic.
"""

from unittest.mock import MagicMock, patch

import pytest
from alpaca.common.exceptions import APIError

from src.alpaca_broker import AlpacaBroker


def make_broker(tmp_path, weekly_budget=50.0, cash=1000.0, mock_client=None):
    with patch("src.alpaca_broker.TradingClient") as MockClient:
        client = mock_client or MagicMock()
        MockClient.return_value = client
        broker = AlpacaBroker(
            api_key="fake",
            api_secret="fake",
            paper=True,
            weekly_budget=weekly_budget,
            budget_state_path=tmp_path / "budget.json",
        )
    client.get_account.return_value = MagicMock(cash=str(cash), portfolio_value=str(cash))
    return broker, client


def test_buy_sizes_order_to_weekly_budget_not_full_cash(tmp_path):
    broker, client = make_broker(tmp_path, weekly_budget=50.0, cash=1000.0)
    client.get_open_position.side_effect = APIError("no position")
    client.submit_order.return_value = MagicMock(id="order-1")

    result = broker.buy("AAPL")

    assert result["notional"] == 50.0
    submitted_req = client.submit_order.call_args[0][0]
    assert submitted_req.notional == 50.0
    assert broker.budget.remaining() == 0.0


def test_buy_capped_by_available_cash_when_less_than_budget(tmp_path):
    broker, client = make_broker(tmp_path, weekly_budget=50.0, cash=12.0)
    client.get_open_position.side_effect = APIError("no position")
    client.submit_order.return_value = MagicMock(id="order-2")

    result = broker.buy("AAPL")

    assert result["notional"] == 12.0


def test_buy_noops_when_already_holding_position(tmp_path):
    broker, client = make_broker(tmp_path, weekly_budget=50.0, cash=1000.0)
    client.get_open_position.return_value = MagicMock(qty="3.0")

    result = broker.buy("AAPL")

    assert result is None
    client.submit_order.assert_not_called()


def test_buy_noops_when_weekly_budget_exhausted(tmp_path):
    broker, client = make_broker(tmp_path, weekly_budget=50.0, cash=1000.0)
    client.get_open_position.side_effect = APIError("no position")
    broker.budget.record_spend(50.0)

    result = broker.buy("AAPL")

    assert result is None
    client.submit_order.assert_not_called()


def test_second_buy_in_same_week_is_capped_by_remaining_budget(tmp_path):
    broker, client = make_broker(tmp_path, weekly_budget=50.0, cash=1000.0)
    client.get_open_position.side_effect = APIError("no position")
    client.submit_order.return_value = MagicMock(id="order-3")

    broker.buy("AAPL")  # spends the full $50
    # simulate the position having been sold, freeing it up to buy again
    client.get_open_position.side_effect = APIError("no position")
    second = broker.buy("AAPL")

    assert second is None  # no budget left this week


def test_sell_liquidates_full_open_position(tmp_path):
    broker, client = make_broker(tmp_path, weekly_budget=50.0, cash=1000.0)
    client.get_open_position.return_value = MagicMock(qty="2.5")
    client.submit_order.return_value = MagicMock(id="order-4")

    result = broker.sell("AAPL")

    assert result["qty"] == 2.5
    submitted_req = client.submit_order.call_args[0][0]
    assert submitted_req.qty == 2.5


def test_sell_noops_without_open_position(tmp_path):
    broker, client = make_broker(tmp_path, weekly_budget=50.0, cash=1000.0)
    client.get_open_position.side_effect = APIError("no position")

    result = broker.sell("AAPL")

    assert result is None
    client.submit_order.assert_not_called()


def test_selling_does_not_consume_weekly_budget(tmp_path):
    broker, client = make_broker(tmp_path, weekly_budget=50.0, cash=1000.0)
    client.get_open_position.return_value = MagicMock(qty="2.5")
    client.submit_order.return_value = MagicMock(id="order-5")

    broker.sell("AAPL")

    assert broker.budget.remaining() == 50.0
