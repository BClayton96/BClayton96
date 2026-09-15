"""Real (or Alpaca-paper) brokerage execution, gated by a weekly budget.

Safety model:
  - Every BUY is sized as a dollar ("notional") market order capped at
    whatever is left of the weekly budget (see budget.py), never more.
  - Only one open position per ticker is allowed at a time (no pyramiding).
  - This class talks to Alpaca's *paper* endpoint unless `paper=False` is
    passed explicitly by the caller — see main.py's `live-trade` command
    for the confirmation flags gating that.
  - It never reads secrets from anywhere but environment variables; no
    API key ever belongs in code or in a committed file.
"""

from __future__ import annotations

from pathlib import Path

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from .budget import WeeklyBudget

MIN_ORDER_NOTIONAL = 1.0  # Alpaca's own minimum for fractional/notional orders


class AlpacaBroker:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        weekly_budget: float = 50.0,
        budget_state_path: str | Path = "state/weekly_budget.json",
    ):
        self.client = TradingClient(api_key, api_secret, paper=paper)
        self.paper = paper
        self.budget = WeeklyBudget(budget_state_path, weekly_limit=weekly_budget)

    def get_position_qty(self, ticker: str) -> float:
        try:
            position = self.client.get_open_position(ticker)
        except APIError:
            return 0.0
        return float(position.qty)

    def get_cash(self) -> float:
        account = self.client.get_account()
        return float(account.cash)

    def buy(self, ticker: str) -> dict | None:
        """Place a notional BUY sized to the smaller of remaining weekly
        budget and available cash. No-ops (returns None) if already holding
        a position, if the budget is exhausted, or if cash is too low.
        """
        if self.get_position_qty(ticker) > 0:
            return None

        remaining_budget = self.budget.remaining()
        if remaining_budget < MIN_ORDER_NOTIONAL:
            return None

        cash = self.get_cash()
        notional = round(min(remaining_budget, cash), 2)
        if notional < MIN_ORDER_NOTIONAL:
            return None

        order_req = MarketOrderRequest(
            symbol=ticker,
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(order_req)
        self.budget.record_spend(notional)
        return {"order_id": str(order.id), "side": "BUY", "notional": notional}

    def sell(self, ticker: str) -> dict | None:
        """Liquidate the entire open position, if any. Selling is never
        constrained by the weekly budget — that cap only limits new spend.
        """
        qty = self.get_position_qty(ticker)
        if qty <= 0:
            return None

        order_req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(order_req)
        return {"order_id": str(order.id), "side": "SELL", "qty": qty}

    def portfolio_value(self) -> float:
        account = self.client.get_account()
        return float(account.portfolio_value)
