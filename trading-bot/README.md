# AI Trading Bot

A self-contained Python project that trains a machine-learning model on
historical stock data and uses it to make trading decisions — either as a
free local simulation (`paper-trade`), or through a real Alpaca brokerage
account (`live-trade`), gated behind a hard weekly spending cap. It also
includes a walk-forward backtester so you can evaluate a strategy on
unseen historical data before trusting it with even fake money.

**Nothing here is financial advice.** Trading carries real risk of loss —
a model that looks good on historical data can still lose money going
forward, and past performance never guarantees future results. Only risk
money you can afford to lose, and treat `live-trade` as something to
approach cautiously (see **Going live with real money** below).

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
7. **Local paper trading** (`src/live.py::run_once`) — fetches the latest
   data, makes one decision with an already-trained model, and updates a
   local JSON portfolio. Never talks to a brokerage. Meant to be run
   periodically (e.g. daily via cron).
8. **Alpaca trading** (`src/alpaca_broker.py`, `src/live.py::run_once_alpaca`)
   — the same decision logic, but the order goes to a real Alpaca account
   via their REST API. Defaults to Alpaca's *paper* endpoint (fake money,
   real API surface) unless you explicitly pass `--live`. Every BUY is
   sized as a dollar-denominated ("notional") order capped by
   `src/budget.py`'s persisted weekly spending limit — regardless of what
   the model says, it physically cannot spend more than the configured
   weekly budget.

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

## Going live with real money

This is the part that actually spends dollars. Set it up carefully and in
order:

### 1. Create your own Alpaca account

Go to [alpaca.markets](https://alpaca.markets), sign up, and complete their
identity verification. **I (the AI) cannot do this step for you** — it
requires your own identity and consent, and no assistant should ever be
handed the keys to move your money without you personally setting up the
account and generating the credentials yourself.

Alpaca gives every account both a **paper** trading endpoint (fake money,
identical API) and a **live** endpoint (real money). Get your paper API
keys first from the dashboard and use those to test everything below
before ever touching live keys.

### 2. Set your API keys as environment variables

Never put these in code or commit them to git (`.gitignore` already
excludes `.env` files and `state/*.json`, but double check before pushing):

```bash
export ALPACA_API_KEY_ID="your-key-id"
export ALPACA_API_SECRET_KEY="your-secret-key"
```

### 3. Test against Alpaca's paper endpoint first

```bash
python main.py train --ticker AAPL --period 5y
python main.py live-trade --ticker AAPL --weekly-budget 50
```

Without `--live`, this always uses Alpaca's paper endpoint — it exercises
the exact same order-submission code path as real trading, but with fake
money. Run it for at least a couple of weeks and sanity-check the orders
in your Alpaca paper dashboard before going further.

### 4. Only then, go live — deliberately

```bash
python main.py live-trade --ticker AAPL --weekly-budget 50 \
    --live --confirm-real-money
```

Both `--live` and `--confirm-real-money` are required together; the CLI
refuses to run with just one. This is intentional friction — there is no
"quiet" way to flip this on by accident.

### The weekly budget guardrail

`--weekly-budget` (default `$50`, matching a "$50/week" plan) is a hard
cap enforced in code, not just documentation:

- Every BUY is placed as a *notional* (dollar-amount) market order, sized
  to whichever is smaller of the remaining weekly budget or your actual
  account cash.
- Spend is tracked in `state/<TICKER>_weekly_budget.json` and resets every
  Monday. Once the week's budget is spent, further BUY signals are
  silently skipped (logged in the decision JSON as `"order": null`) until
  the next week — the bot will never ask you to approve going over budget,
  it just won't place the order.
- Only one open position per ticker at a time (no pyramiding into a
  position with repeated buys).
- Selling is never blocked by the budget — you can always exit a position.

To actually run this weekly (rather than invoking it by hand), schedule
`python main.py live-trade --ticker AAPL --live --confirm-real-money`
via cron or a similar scheduler on a machine you control. This repository
does not run anything on a schedule for you.

### What this does *not* do

- No shorting, leverage, options, or margin — long-only, cash-settled.
- No portfolio diversification logic — one ticker at a time, sized simply.
- No tax handling. Trades in a taxable brokerage account are taxable
  events; keep records and consult a tax professional.
- No protection against the model being wrong. A $50/week cap limits how
  much you can lose per week to buying decisions, but it does not
  guarantee against losing the invested amount if the position's value
  drops before you sell.

## Running tests

Tests use synthetic, deterministic price data — no network access needed:

```bash
pip install -r requirements.txt  # includes pytest
pytest tests/ -v
```

## Design notes / limitations

- **Single ticker, single position** at a time, long-only (no shorting,
  no leverage, no options). Keeps the risk model simple to reason about.
- **Daily bars only.** No intraday/tick-level data or execution.
- The weekly budget cap limits new spending, not risk on money already
  invested — a position bought in week 1 can still lose value in week 2.
- A backtest's historical performance is not a promise of future returns.
  Markets change; a model trained on one regime can fail in another.
- `live-trade` submits real orders through your own Alpaca account and
  your own API keys — this assistant never holds or transmits your
  credentials, and does not run or schedule this command for you.
