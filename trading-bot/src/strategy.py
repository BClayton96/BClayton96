"""Turn model predictions (or simple rules) into BUY / SELL / HOLD signals."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MLStrategy:
    """Signal from model confidence, with dead-zone around 0.5 to avoid churn."""

    buy_threshold: float = 0.55
    sell_threshold: float = 0.45

    def signal(self, proba_up: float) -> str:
        if proba_up >= self.buy_threshold:
            return "BUY"
        if proba_up <= self.sell_threshold:
            return "SELL"
        return "HOLD"

    def signals(self, proba_up: pd.Series) -> pd.Series:
        return proba_up.apply(self.signal).rename("signal")


@dataclass
class SmaCrossoverStrategy:
    """Baseline rule-based strategy: fast SMA crossing above/below slow SMA."""

    fast: int = 10
    slow: int = 50

    def signals(self, price_df: pd.DataFrame) -> pd.Series:
        close = price_df["Close"]
        sma_fast = close.rolling(self.fast).mean()
        sma_slow = close.rolling(self.slow).mean()

        above = sma_fast > sma_slow
        crossed_up = above & ~above.shift(1).fillna(False)
        crossed_down = ~above & above.shift(1).fillna(False)

        out = pd.Series("HOLD", index=price_df.index, name="signal")
        out[crossed_up] = "BUY"
        out[crossed_down] = "SELL"
        return out
