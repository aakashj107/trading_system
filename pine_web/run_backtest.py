"""
run_backtest.py — glue between a transpiled Pine Strategy and your Backtester.
Returns metrics + trade list + equity curve so the UI can plot them.
"""

import os
import sys

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import pandas as pd

from backtester import Backtester
from config import TradingCosts


def run_pine_backtest(strategy, df, initial_capital=100000.0,
                      position_size_pct=0.95, stop_loss_pct=0.05,
                      trailing_stop=True, trailing_stop_pct=0.03,
                      symbol="PINE"):
    """Run a transpiled strategy over df and return a result dict.

    Adds a tiny guard so strategy.generate_signals is called once.
    """
    bt = Backtester(
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        costs=TradingCosts(),
        stop_loss_pct=stop_loss_pct,
        use_trailing_stop=trailing_stop,
        trailing_stop_pct=trailing_stop_pct,
    )
    sig_df = strategy.generate_signals(df)
    res = bt.run_backtest(strategy, sig_df, symbol=symbol)
    return res


def equity_frame(res):
    eq = res.get("equity_curve")
    if eq is None or getattr(eq, "empty", True):
        return pd.DataFrame()
    return eq


def metrics_summary(res):
    """Flatten the metrics dict into the few fields the UI shows by default."""
    m = res.get("metrics", {})
    keys = ["total_trades", "winning_trades", "losing_trades", "win_rate",
            "total_pnl", "total_return_pct", "avg_pnl_per_trade",
            "profit_factor", "avg_holding_days", "total_charges",
            "cagr_pct", "max_drawdown_pct", "sharpe_ratio", "sortino_ratio",
            "initial_capital", "final_capital"]
    return {k: m.get(k) for k in keys}
