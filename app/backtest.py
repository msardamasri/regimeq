"""
backtest.py — RegimeIQ signal backtest engine.

Strategy:
  - BUY  SPY when regime_score crosses above BULL_THRESHOLD (65)
  - SELL to cash when regime_score crosses below BEAR_THRESHOLD (35)
  - Otherwise hold current position
  - 0 transaction costs (realistic for ETF at this frequency)

Compared against: buy-and-hold SPY over the same period.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

BULL_THRESHOLD = 65
BEAR_THRESHOLD = 35


@dataclass
class BacktestResult:
    equity: pd.Series           # strategy cumulative equity (starts at 100)
    bh_equity: pd.Series        # buy-and-hold equity (starts at 100)
    position: pd.Series         # 1 = invested, 0 = cash
    daily_returns: pd.Series    # strategy daily returns
    # ── Summary stats ──────────────────────────────────────────────────────
    total_return: float         # %
    bh_return: float            # %
    annual_return: float        # % CAGR
    bh_annual_return: float
    sharpe: float               # annualised, rf=0
    bh_sharpe: float
    max_drawdown: float         # % (negative)
    bh_max_drawdown: float
    pct_invested: float         # % of days in market
    n_trades: int               # number of round trips
    years: float


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min() * 100)


def _sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(252))


def run_backtest(history: pd.DataFrame,
                 start_date: Optional[str] = None,
                 end_date:   Optional[str] = None) -> BacktestResult:
    """
    Runs the regime-based strategy on history DataFrame.
    Requires columns: regime_score, spy_close.
    """
    df = history.copy()
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    df = df.dropna(subset=["regime_score", "spy_close"])

    # ── Daily SPY returns ──────────────────────────────────────────────────
    df["spy_ret"] = df["spy_close"].pct_change().fillna(0)

    # ── Position signal ────────────────────────────────────────────────────
    # Start in cash (position = 0)
    position = np.zeros(len(df))
    pos = 0  # 0 = cash, 1 = invested

    for i, score in enumerate(df["regime_score"].values):
        if pos == 0 and score >= BULL_THRESHOLD:
            pos = 1   # enter market
        elif pos == 1 and score <= BEAR_THRESHOLD:
            pos = 0   # exit to cash
        position[i] = pos

    df["position"] = position
    # Returns are earned the day AFTER the signal (no look-ahead)
    df["strat_ret"] = df["position"].shift(1).fillna(0) * df["spy_ret"]

    # ── Equity curves ──────────────────────────────────────────────────────
    df["equity"]    = 100 * (1 + df["strat_ret"]).cumprod()
    df["bh_equity"] = 100 * (1 + df["spy_ret"]).cumprod()

    # ── Stats ──────────────────────────────────────────────────────────────
    years = len(df) / 252
    total_ret    = float((df["equity"].iloc[-1]    / 100 - 1) * 100)
    bh_ret       = float((df["bh_equity"].iloc[-1] / 100 - 1) * 100)
    annual_ret   = float(((df["equity"].iloc[-1]    / 100) ** (1/years) - 1) * 100) if years > 0 else 0
    bh_annual    = float(((df["bh_equity"].iloc[-1] / 100) ** (1/years) - 1) * 100) if years > 0 else 0

    # Count round trips (0→1→0 = 1 trade)
    pos_series = df["position"]
    n_trades   = int(((pos_series.diff() == 1)).sum())
    pct_in     = float(pos_series.mean() * 100)

    return BacktestResult(
        equity        = df["equity"],
        bh_equity     = df["bh_equity"],
        position      = df["position"],
        daily_returns = df["strat_ret"],
        total_return  = total_ret,
        bh_return     = bh_ret,
        annual_return = annual_ret,
        bh_annual_return = bh_annual,
        sharpe        = _sharpe(df["strat_ret"]),
        bh_sharpe     = _sharpe(df["spy_ret"]),
        max_drawdown  = _max_drawdown(df["equity"]),
        bh_max_drawdown = _max_drawdown(df["bh_equity"]),
        pct_invested  = pct_in,
        n_trades      = n_trades,
        years         = years,
    )
