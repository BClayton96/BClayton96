import json

from src.broker import PaperBroker


def test_buy_reduces_cash_and_opens_position():
    broker = PaperBroker(cash=1000.0, allocation_pct=0.5, commission_pct=0.0)
    broker.buy("2024-01-01", "TEST", 100.0)
    assert broker.position_qty == 5.0
    assert broker.cash == 500.0
    assert len(broker.trade_log) == 1


def test_buy_while_holding_position_is_noop():
    broker = PaperBroker(cash=1000.0, allocation_pct=0.5, commission_pct=0.0)
    broker.buy("2024-01-01", "TEST", 100.0)
    broker.buy("2024-01-02", "TEST", 90.0)
    assert len(broker.trade_log) == 1
    assert broker.position_qty == 5.0


def test_sell_liquidates_full_position():
    broker = PaperBroker(cash=1000.0, allocation_pct=0.5, commission_pct=0.0)
    broker.buy("2024-01-01", "TEST", 100.0)
    broker.sell("2024-01-02", "TEST", 110.0)
    assert broker.position_qty == 0.0
    assert broker.cash == 500.0 + 5.0 * 110.0


def test_sell_without_position_is_noop():
    broker = PaperBroker(cash=1000.0)
    broker.sell("2024-01-01", "TEST", 100.0)
    assert broker.cash == 1000.0
    assert len(broker.trade_log) == 0


def test_commission_reduces_proceeds():
    broker = PaperBroker(cash=1000.0, allocation_pct=1.0, commission_pct=0.01)
    broker.buy("2024-01-01", "TEST", 100.0)
    # full budget is spent (minus float noise); commission eats into share count
    assert abs(broker.cash) < 1e-9
    assert abs(broker.position_qty - (1000 / 100 / 1.01)) < 1e-6


def test_portfolio_value_reflects_open_position():
    broker = PaperBroker(cash=1000.0, allocation_pct=0.5, commission_pct=0.0)
    broker.buy("2024-01-01", "TEST", 100.0)
    assert broker.portfolio_value(120.0) == 500.0 + 5.0 * 120.0


def test_save_and_load_state_roundtrip(tmp_path):
    broker = PaperBroker(cash=1000.0, allocation_pct=0.5, commission_pct=0.0)
    broker.buy("2024-01-01", "TEST", 100.0)

    state_path = tmp_path / "state.json"
    broker.save_state(state_path)
    assert json.loads(state_path.read_text())["position_qty"] == 5.0

    restored = PaperBroker.load_state(state_path)
    assert restored.cash == broker.cash
    assert restored.position_qty == broker.position_qty
    assert len(restored.trade_log) == 1
