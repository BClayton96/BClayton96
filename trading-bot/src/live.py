"""One-shot trading decisions: local paper-portfolio simulation, and a
real (or Alpaca-paper) brokerage path in run_once_alpaca.

`run_once` never leaves this machine: it only updates a local JSON file.
`run_once_alpaca` can place a REAL order if given a non-paper AlpacaBroker
— see main.py's `live-trade` command for the explicit opt-in this requires.
"""

from __future__ import annotations

import json
from pathlib import Path

from .broker import PaperBroker
from .data import fetch_price_history
from .features import build_dataset
from .model import TradingModel
from .strategy import MLStrategy


def decide_signal(
    ticker: str,
    model_path: str | Path,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    lookback_period: str = "1y",
) -> tuple[str, float, float, str]:
    """Fetch latest data and return (date, price, proba_up, signal)."""
    price_df = fetch_price_history(ticker, period=lookback_period)
    dataset = build_dataset(price_df, with_labels=False)
    if dataset.empty:
        raise ValueError("Not enough recent data to compute features")

    model = TradingModel().load(model_path)
    latest_row = dataset.iloc[[-1]]
    proba_up = float(model.predict_proba_up(latest_row).iloc[0])

    strategy = MLStrategy(buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    signal = strategy.signal(proba_up)

    latest_date = str(dataset.index[-1].date())
    latest_price = float(price_df["Close"].loc[dataset.index[-1]])
    return latest_date, latest_price, proba_up, signal


def run_once(
    ticker: str,
    model_path: str | Path,
    state_path: str | Path,
    initial_cash: float = 10_000.0,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    lookback_period: str = "1y",
) -> dict:
    """Fetch latest data, make one decision, update and persist LOCAL
    SIMULATED paper state. Never contacts a brokerage.
    """
    state_path = Path(state_path)
    if state_path.exists():
        broker = PaperBroker.load_state(state_path)
    else:
        broker = PaperBroker(cash=initial_cash)

    latest_date, latest_price, proba_up, signal = decide_signal(
        ticker, model_path, buy_threshold, sell_threshold, lookback_period
    )

    if signal == "BUY":
        broker.buy(latest_date, ticker, latest_price)
    elif signal == "SELL":
        broker.sell(latest_date, ticker, latest_price)

    broker.save_state(state_path)

    return {
        "date": latest_date,
        "ticker": ticker,
        "price": latest_price,
        "proba_up": proba_up,
        "signal": signal,
        "cash": broker.cash,
        "position_qty": broker.position_qty,
        "portfolio_value": broker.portfolio_value(latest_price),
    }


def run_once_alpaca(
    ticker: str,
    model_path: str | Path,
    broker,  # src.alpaca_broker.AlpacaBroker
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    lookback_period: str = "1y",
) -> dict:
    """Fetch latest data, make one decision, and submit it to `broker`.

    If `broker.paper` is False this places a REAL order with REAL money.
    The caller (main.py) is responsible for gating that behind explicit
    user confirmation — this function does not ask again.
    """
    latest_date, latest_price, proba_up, signal = decide_signal(
        ticker, model_path, buy_threshold, sell_threshold, lookback_period
    )

    order = None
    if signal == "BUY":
        order = broker.buy(ticker)
    elif signal == "SELL":
        order = broker.sell(ticker)

    return {
        "date": latest_date,
        "ticker": ticker,
        "price": latest_price,
        "proba_up": proba_up,
        "signal": signal,
        "order": order,
        "paper": broker.paper,
        "weekly_budget_remaining": broker.budget.remaining(),
    }


def print_decision(result: dict) -> None:
    print(json.dumps(result, indent=2))
