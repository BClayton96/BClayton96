"""A persisted, hard weekly spending cap for real-money trading.

This is the safety rail between a model's BUY signal and an actual order:
no matter what the strategy says, a live broker must ask this object how
much it is still allowed to spend before sizing an order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


def _week_start(d: date) -> date:
    """Monday of the week containing d."""
    return d - timedelta(days=d.weekday())


@dataclass
class WeeklyBudget:
    state_path: Path
    weekly_limit: float = 50.0
    _week_start: str = ""
    _spent: float = 0.0

    def __post_init__(self) -> None:
        self.state_path = Path(self.state_path)
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            self._week_start = data.get("week_start", "")
            self._spent = data.get("spent", 0.0)
        self._roll_if_new_week()

    def _roll_if_new_week(self) -> None:
        today_week = _week_start(date.today()).isoformat()
        if self._week_start != today_week:
            self._week_start = today_week
            self._spent = 0.0
            self._save()

    def remaining(self) -> float:
        self._roll_if_new_week()
        return max(0.0, self.weekly_limit - self._spent)

    def record_spend(self, amount: float) -> None:
        if amount <= 0:
            return
        self._roll_if_new_week()
        self._spent += amount
        self._save()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"week_start": self._week_start, "spent": self._spent}, indent=2)
        )
