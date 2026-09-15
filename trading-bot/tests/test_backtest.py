from src.backtest import run_backtest
from src.strategy import SmaCrossoverStrategy


def test_run_backtest_returns_sane_result(synthetic_price_df):
    result = run_backtest(synthetic_price_df, ticker="TEST", initial_cash=10_000.0)

    assert result.equity_curve.iloc[0] > 0
    assert len(result.equity_curve) > 0
    assert isinstance(result.total_return_pct, float)
    assert isinstance(result.sharpe_ratio, float)
    assert result.max_drawdown_pct <= 0
    assert result.num_trades >= 0


def test_sma_crossover_signals_are_valid_labels(synthetic_price_df):
    strategy = SmaCrossoverStrategy(fast=10, slow=50)
    signals = strategy.signals(synthetic_price_df)
    assert set(signals.unique()) <= {"BUY", "SELL", "HOLD"}
    assert len(signals) == len(synthetic_price_df)
