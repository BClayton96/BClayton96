#!/usr/bin/env python3
"""AI trading bot CLI.

`train`, `backtest`, and `paper-trade` never leave this machine — they
only simulate trades locally. `live-trade` can place REAL orders through
an Alpaca brokerage account (real money) once you explicitly opt in with
--live --confirm-real-money, and only after you've set your own Alpaca
API keys as environment variables. See README.md for full setup and the
risk disclaimer.

Subcommands:
    train        Train a model on historical data and save it to disk.
    backtest     Walk-forward backtest the model (or SMA baseline) on history.
    paper-trade  Make one simulated (local, fake-money) trading decision.
    live-trade   Make one decision via Alpaca. Defaults to Alpaca's PAPER
                 endpoint (fake money, real API); requires --live plus
                 --confirm-real-money to touch actual funds.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.backtest import run_backtest
from src.data import fetch_price_history
from src.features import build_dataset
from src.live import print_decision, run_once, run_once_alpaca
from src.model import TradingModel
from src.plotting import plot_equity_curve

DEFAULT_MODEL_DIR = Path("models")
DEFAULT_STATE_DIR = Path("state")


def cmd_train(args: argparse.Namespace) -> None:
    price_df = fetch_price_history(args.ticker, period=args.period)
    dataset = build_dataset(price_df, with_labels=True)
    model = TradingModel().fit(dataset)

    DEFAULT_MODEL_DIR.mkdir(exist_ok=True)
    out_path = DEFAULT_MODEL_DIR / f"{args.ticker.upper()}.joblib"
    model.save(out_path)
    print(f"Trained on {len(dataset)} rows of {args.ticker} history.")
    print(f"Model saved to {out_path}")


def cmd_backtest(args: argparse.Namespace) -> None:
    price_df = fetch_price_history(args.ticker, start=args.start, end=args.end, period=args.period)
    result = run_backtest(
        price_df,
        ticker=args.ticker,
        train_frac=args.train_frac,
        initial_cash=args.cash,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )

    print(f"\nBacktest: {args.ticker}  ({result.equity_curve.index[0].date()} to {result.equity_curve.index[-1].date()})")
    print("-" * 60)
    print(f"Starting cash:     ${args.cash:,.2f}")
    print(f"Ending value:      ${result.equity_curve.iloc[-1]:,.2f}")
    print(f"Total return:      {result.total_return_pct:.2f}%")
    print(f"CAGR:              {result.cagr_pct:.2f}%")
    print(f"Sharpe ratio:      {result.sharpe_ratio:.2f}")
    print(f"Max drawdown:      {result.max_drawdown_pct:.2f}%")
    print(f"Number of trades:  {result.num_trades}")

    if args.plot:
        out_path = Path(args.plot)
        plot_equity_curve(result.equity_curve, out_path)
        print(f"\nEquity curve saved to {out_path}")


def cmd_paper_trade(args: argparse.Namespace) -> None:
    model_path = Path(args.model) if args.model else DEFAULT_MODEL_DIR / f"{args.ticker.upper()}.joblib"
    if not model_path.exists():
        print(f"No trained model found at {model_path}. Run 'train' first.", file=sys.stderr)
        sys.exit(1)

    DEFAULT_STATE_DIR.mkdir(exist_ok=True)
    state_path = Path(args.state) if args.state else DEFAULT_STATE_DIR / f"{args.ticker.upper()}.json"

    result = run_once(
        ticker=args.ticker,
        model_path=model_path,
        state_path=state_path,
        initial_cash=args.cash,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    print_decision(result)


def cmd_live_trade(args: argparse.Namespace) -> None:
    if args.live and not args.confirm_real_money:
        print(
            "Refusing to trade: --live requires --confirm-real-money as well.\n"
            "This places REAL orders with REAL money on your Alpaca account.\n"
            "Pass both flags only once you've verified this on Alpaca's paper "
            "endpoint first and are ready to risk real funds.",
            file=sys.stderr,
        )
        sys.exit(1)

    paper = not args.live
    api_key = os.environ.get("ALPACA_API_KEY_ID")
    api_secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not api_key or not api_secret:
        print(
            "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment "
            "variables first (see README.md for how to get these from your "
            "own Alpaca account).",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = Path(args.model) if args.model else DEFAULT_MODEL_DIR / f"{args.ticker.upper()}.joblib"
    if not model_path.exists():
        print(f"No trained model found at {model_path}. Run 'train' first.", file=sys.stderr)
        sys.exit(1)

    DEFAULT_STATE_DIR.mkdir(exist_ok=True)
    budget_state_path = Path(args.budget_state) if args.budget_state else DEFAULT_STATE_DIR / f"{args.ticker.upper()}_weekly_budget.json"

    # Imported lazily: alpaca-py is only required for this command.
    from src.alpaca_broker import AlpacaBroker

    broker = AlpacaBroker(
        api_key=api_key,
        api_secret=api_secret,
        paper=paper,
        weekly_budget=args.weekly_budget,
        budget_state_path=budget_state_path,
    )

    result = run_once_alpaca(
        ticker=args.ticker,
        model_path=model_path,
        broker=broker,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    print_decision(result)
    print(
        "\n[Alpaca PAPER endpoint — no real money moved.]"
        if paper
        else "\n[REAL MONEY order placed via Alpaca.]"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI trading bot (paper-trading / research only)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model on historical data")
    p_train.add_argument("--ticker", required=True)
    p_train.add_argument("--period", default="5y", help="History window, e.g. 2y, 5y, max")
    p_train.set_defaults(func=cmd_train)

    p_bt = sub.add_parser("backtest", help="Walk-forward backtest on historical data")
    p_bt.add_argument("--ticker", required=True)
    p_bt.add_argument("--period", default="5y")
    p_bt.add_argument("--start", default=None)
    p_bt.add_argument("--end", default=None)
    p_bt.add_argument("--cash", type=float, default=10_000.0)
    p_bt.add_argument("--train-frac", type=float, default=0.6)
    p_bt.add_argument("--buy-threshold", type=float, default=0.55)
    p_bt.add_argument("--sell-threshold", type=float, default=0.45)
    p_bt.add_argument("--plot", default=None, help="Path to save an equity-curve PNG")
    p_bt.set_defaults(func=cmd_backtest)

    p_live = sub.add_parser("paper-trade", help="Make one simulated decision using the latest data")
    p_live.add_argument("--ticker", required=True)
    p_live.add_argument("--model", default=None, help="Path to a trained model (default: models/<TICKER>.joblib)")
    p_live.add_argument("--state", default=None, help="Path to portfolio state JSON (default: state/<TICKER>.json)")
    p_live.add_argument("--cash", type=float, default=10_000.0, help="Starting cash if no state file exists yet")
    p_live.add_argument("--buy-threshold", type=float, default=0.55)
    p_live.add_argument("--sell-threshold", type=float, default=0.45)
    p_live.set_defaults(func=cmd_paper_trade)

    p_alpaca = sub.add_parser(
        "live-trade",
        help="Make one decision via Alpaca (paper endpoint by default; --live --confirm-real-money for real money)",
    )
    p_alpaca.add_argument("--ticker", required=True)
    p_alpaca.add_argument("--model", default=None, help="Path to a trained model (default: models/<TICKER>.joblib)")
    p_alpaca.add_argument("--weekly-budget", type=float, default=50.0, help="Max new dollars spent on BUYs per calendar week")
    p_alpaca.add_argument("--budget-state", default=None, help="Path to weekly-budget tracking JSON (default: state/<TICKER>_weekly_budget.json)")
    p_alpaca.add_argument("--buy-threshold", type=float, default=0.55)
    p_alpaca.add_argument("--sell-threshold", type=float, default=0.45)
    p_alpaca.add_argument("--live", action="store_true", help="Use Alpaca's LIVE (real money) endpoint instead of paper")
    p_alpaca.add_argument("--confirm-real-money", action="store_true", help="Required alongside --live to acknowledge this risks real funds")
    p_alpaca.set_defaults(func=cmd_live_trade)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
