"""Optional equity-curve plotting for backtest results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_equity_curve(equity_curve: pd.Series, out_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_curve.index, equity_curve.values)
    ax.set_title("Backtest equity curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
