"""
strategy_momentum_rotation.py — OUR recreation of the RupeeCase-style
cross-sectional momentum rotation (the most credible public systematic strategy
found on the web; RupeeCase publishes net-of-cost, NSE-BhavCopy-sourced,
survivorship-free backtests, CAGR 14-57%, Sharpe up to 1.81).

We do NOT import RupeeCase's exact (undisclosed) parameters. We implement the
GENERIC, defensible mechanic they describe:
  - rank universe by trailing return (skip the last month = 12-1 momentum)
  - hold the top N names, equal-weight
  - rebalance on a fixed cadence (monthly)
This is a portfolio-rotation strategy, a TYPE our single-name signal strategies
don't cover. It is tested blind OOS before any claim of edge.

NOTE: generate_signals() here works on a multi-symbol dict (universe), not the
single-symbol df interface, because rotation is cross-sectional. The OOS test
(running in this module / validate_momentum_rotation.py) drives PaperTrader
directly.
"""
import numpy as np
import pandas as pd


class MomentumRotation:
    def __init__(self, lookback_months: int = 12, skip_months: int = 1,
                 hold_n: int = 10, rebalance: str = "M"):
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.hold_n = hold_n
        self.rebalance = rebalance  # "M"=monthly, "W"=weekly, "2W"=fortnightly
        self.name = (f"MomentumRotation_L{lookback_months}S{skip_months}"
                     f"_N{hold_n}_{rebalance}")

    def rank(self, data: dict, date: pd.Timestamp) -> list:
        """Return list of (symbol, momentum) for symbols with data up to `date`."""
        scores = []
        for sym, d in data.items():
            hist = d.loc[:date]
            if len(hist) < self.lookback_months * 21:
                continue
            try:
                total = self.lookback_months * 21
                skip = self.skip_months * 21
                price_now = hist["Close"].iloc[-1]
                price_look = hist["Close"].iloc[-total]
                price_skip = hist["Close"].iloc[-skip] if len(hist) > skip else hist["Close"].iloc[0]
                # 12-1 momentum: return over lookback, excluding last `skip` months
                mom = (price_now / price_skip) - (price_look / price_skip)
                scores.append((sym, mom))
            except Exception:
                continue
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: self.hold_n]

    def targets_on(self, data: dict, date: pd.Timestamp) -> set:
        return {sym for sym, _ in self.rank(data, date)}
