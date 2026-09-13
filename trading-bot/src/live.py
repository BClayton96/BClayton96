"""Paper-trading loop: makes one simulated trading decision per run.

This never places real orders or talks to a brokerage. It is meant to be
invoked periodically (e.g. once a day via cron) with `--once`, persisting
its portfolio state to a local JSON file between runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from .broker import PaperBroker
from .data import fetch_price_history
from .features import build_dataset
from .model import TradingModel
from .strategy import MLStrategy


def run_once(
    ticker: str,
    model_path: str | Path,
    state_path: str | Path,
    initial_cash: float = 10_000.0,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    lookback_period: str = "1y",
) -> dict:
    """Fetch latest data, make one decision, update and persist paper state."""
    state_path = Path(state_path)
    if state_path.exists():
        broker = PaperBroker.load_state(state_path)
    else:
        broker = PaperBroker(cash=initial_cash)

    price_df = fetch_price_history(ticker, period=lookback_period)
    dataset = build_dataset(price_df, with_labels=False)
    if dataset.empty:
        raise ValueError("Not enough recent data to compute features")

    model = TradingModel().load(model_path)
    latest_row = dataset.iloc[[-1]]
    proba_up = model.predict_proba_up(latest_row).iloc[0]

    strategy = MLStrategy(buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    signal = strategy.signal(proba_up)

    latest_date = str(dataset.index[-1].date())
    latest_price = float(price_df["Close"].loc[dataset.index[-1]])

    if signal == "BUY":
        broker.buy(latest_date, ticker, latest_price)
    elif signal == "SELL":
        broker.sell(latest_date, ticker, latest_price)

    broker.save_state(state_path)

    result = {
        "date": latest_date,
        "ticker": ticker,
        "price": latest_price,
        "proba_up": float(proba_up),
        "signal": signal,
        "cash": broker.cash,
        "position_qty": broker.position_qty,
        "portfolio_value": broker.portfolio_value(latest_price),
    }
    return result


def print_decision(result: dict) -> None:
    print(json.dumps(result, indent=2))
