"""Walk-forward backtest: train on the past, trade on unseen future data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .broker import PaperBroker
from .features import build_dataset
from .model import TradingModel
from .strategy import MLStrategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_log: list
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int


def _compute_metrics(equity: pd.Series) -> tuple[float, float, float, float]:
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_days = len(equity)
    years = max(n_days / 252, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    daily_returns = equity.pct_change().dropna()
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    return total_return * 100, cagr * 100, sharpe, max_dd * 100


def run_backtest(
    price_df: pd.DataFrame,
    ticker: str,
    train_frac: float = 0.6,
    initial_cash: float = 10_000.0,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> BacktestResult:
    """Train the model on the first `train_frac` of history, then simulate
    trading day-by-day over the remaining (unseen) period.

    A decision made using features known at the close of day t is executed
    at the close of day t+1, to avoid look-ahead bias.
    """
    dataset = build_dataset(price_df, with_labels=True)
    if len(dataset) < 100:
        raise ValueError("Not enough data to backtest (need at least ~100 rows after warm-up)")

    split = int(len(dataset) * train_frac)
    train_set = dataset.iloc[:split]
    test_set = dataset.iloc[split:]

    model = TradingModel().fit(train_set)
    proba_up = model.predict_proba_up(test_set)
    strategy = MLStrategy(buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    signals = strategy.signals(proba_up)

    broker = PaperBroker(cash=initial_cash)
    equity_values = []
    equity_dates = []

    exec_prices = price_df["Close"].reindex(test_set.index).shift(-1)

    for date, signal in signals.items():
        exec_price = exec_prices.loc[date]
        if pd.isna(exec_price):
            exec_price = price_df["Close"].loc[date]

        if signal == "BUY":
            broker.buy(str(date.date()), ticker, float(exec_price))
        elif signal == "SELL":
            broker.sell(str(date.date()), ticker, float(exec_price))

        equity_values.append(broker.portfolio_value(float(price_df["Close"].loc[date])))
        equity_dates.append(date)

    # liquidate any open position at the final price for a fair final mark
    if broker.position_qty > 0:
        broker.sell(str(test_set.index[-1].date()), ticker, float(price_df["Close"].iloc[-1]))
        equity_values[-1] = broker.portfolio_value(float(price_df["Close"].iloc[-1]))

    equity_curve = pd.Series(equity_values, index=equity_dates, name="equity")
    total_return, cagr, sharpe, max_dd = _compute_metrics(equity_curve)

    return BacktestResult(
        equity_curve=equity_curve,
        trade_log=broker.trade_log,
        total_return_pct=total_return,
        cagr_pct=cagr,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        num_trades=len(broker.trade_log),
    )
