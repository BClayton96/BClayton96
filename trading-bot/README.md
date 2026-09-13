# AI Trading Bot (paper-trading / research)

A self-contained Python project that trains a machine-learning model on
historical stock data and uses it to make simulated ("paper") trading
decisions. It includes a walk-forward backtester so you can evaluate the
strategy on unseen historical data before trusting it with even fake money.

**This bot never places real trades.** It has no brokerage integration.
Every "buy" and "sell" only updates numbers in a local JSON file
(`state/<TICKER>.json`) representing a hypothetical portfolio. Treat all
output as educational / research material, not financial advice — trading
carries real risk of loss, and a model that looks good on historical data
can still lose money going forward.

## How it works

1. **Data** (`src/data.py`) — pulls daily OHLCV history from Yahoo Finance
   via `yfinance` (no API key required).
2. **Features** (`src/features.py`) — computes technical indicators (SMA/EMA
   ratios, RSI, MACD, rolling volatility, momentum, volume change) from data
   available up to and including each day, and a label ("did the price rise
   the next day?") for training.
3. **Model** (`src/model.py`) — a `RandomForestClassifier` predicts the
   probability that tomorrow's close is higher than today's.
4. **Strategy** (`src/strategy.py`) — turns that probability into
   BUY / SELL / HOLD using confidence thresholds (with a dead zone around
   50% to avoid overtrading). A simple SMA-crossover baseline is included
   for comparison.
5. **Broker** (`src/broker.py`) — a simulated single-position paper
   portfolio: tracks cash, position size, commission, and a trade log.
   Nothing here talks to a real exchange or brokerage.
6. **Backtest** (`src/backtest.py`) — trains on the first portion of
   history and walks forward through the rest day-by-day, executing each
   day's decision at the *next* day's price to avoid look-ahead bias.
   Reports total return, CAGR, Sharpe ratio, and max drawdown.
7. **Paper trading** (`src/live.py`) — fetches the latest data, makes one
   decision with an already-trained model, and updates the persisted
   simulated portfolio. Meant to be run periodically (e.g. daily via cron),
   with `--once`-style single invocations.

## Setup

```bash
cd trading-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train a model on 5 years of history:

```bash
python main.py train --ticker AAPL --period 5y
```

Backtest it (walk-forward, 60% train / 40% test split by default):

```bash
python main.py backtest --ticker AAPL --period 5y --cash 10000 --plot equity_curve.png
```

Make one simulated paper-trading decision using the latest data
(intended to be run once a day, e.g. via cron):

```bash
python main.py paper-trade --ticker AAPL
```

Each `paper-trade` run prints a JSON decision and updates
`state/AAPL.json` with the simulated portfolio's cash, position, and full
trade history.

## Running tests

Tests use synthetic, deterministic price data — no network access needed:

```bash
pip install -r requirements.txt  # includes pytest
pytest tests/ -v
```

## Design notes / limitations

- **Paper trading only.** Wiring this up to a real broker (e.g. Alpaca,
  Interactive Brokers) would require adding an authenticated order-execution
  client behind the same interface as `PaperBroker` — deliberately not
  included here, since that crosses from "research tool" into "moves real
  money" and should be a separate, carefully reviewed decision.
- **Single ticker, single position** at a time, long-only (no shorting,
  no leverage, no options). Keeps the risk model simple to reason about.
- **Daily bars only.** No intraday/tick-level data or execution.
- A backtest's historical performance is not a promise of future returns.
  Markets change; a model trained on one regime can fail in another.
