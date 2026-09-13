#!/usr/bin/env python3
"""AI trading bot CLI.

IMPORTANT: This tool only ever simulates trades against a local paper
portfolio. It has no integration with any brokerage and never places a
real order. See README.md for details and the educational-use disclaimer.

Subcommands:
    train        Train a model on historical data and save it to disk.
    backtest     Walk-forward backtest the model (or SMA baseline) on history.
    paper-trade  Make one simulated trading decision using the latest data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.backtest import run_backtest
from src.data import fetch_price_history
from src.features import build_dataset
from src.live import print_decision, run_once
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
