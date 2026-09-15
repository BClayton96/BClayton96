"""A simulated (paper-trading only) broker. No real money, no live orders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Fill:
    date: str
    side: str
    ticker: str
    qty: float
    price: float
    cash_after: float


@dataclass
class PaperBroker:
    """Tracks simulated cash and a single-ticker position.

    Position sizing: on BUY, invests `allocation_pct` of current cash.
    On SELL, liquidates the entire current position. Applies a flat
    `commission_pct` cost on each trade's notional value.
    """

    cash: float = 10_000.0
    allocation_pct: float = 0.95
    commission_pct: float = 0.001
    position_qty: float = 0.0
    trade_log: list[Fill] = field(default_factory=list)

    def buy(self, date: str, ticker: str, price: float) -> None:
        if price <= 0 or self.position_qty > 0:
            return
        budget = self.cash * self.allocation_pct
        # size the position so cost + commission fits within budget
        qty = budget / (price * (1 + self.commission_pct))
        cost = qty * price
        commission = cost * self.commission_pct
        total = cost + commission
        if qty <= 0 or total > self.cash:
            return
        self.cash -= total
        self.position_qty += qty
        self.trade_log.append(Fill(date, "BUY", ticker, qty, price, self.cash))

    def sell(self, date: str, ticker: str, price: float) -> None:
        if self.position_qty <= 0 or price <= 0:
            return
        proceeds = self.position_qty * price
        commission = proceeds * self.commission_pct
        self.cash += proceeds - commission
        qty = self.position_qty
        self.position_qty = 0.0
        self.trade_log.append(Fill(date, "SELL", ticker, qty, price, self.cash))

    def portfolio_value(self, price: float) -> float:
        return self.cash + self.position_qty * price

    def to_state_dict(self) -> dict:
        return {
            "cash": self.cash,
            "position_qty": self.position_qty,
            "trade_log": [vars(f) for f in self.trade_log],
        }

    def save_state(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_state_dict(), indent=2))

    @classmethod
    def load_state(cls, path: str | Path, **kwargs) -> "PaperBroker":
        data = json.loads(Path(path).read_text())
        broker = cls(cash=data["cash"], position_qty=data["position_qty"], **kwargs)
        broker.trade_log = [Fill(**f) for f in data["trade_log"]]
        return broker
