"""
strategies_donchian.py — Weekly-style Donchian breakout, daily-native.

From insightbig case study: enter when price exceeds the 50-WEEK high, exit when
it falls below the 40-WEEK low. A long-horizon trend-following system.

Mapped to our DAILY framework (no weekly resampling needed):
  - 50 weeks  ~= 250 trading days  -> entry when Close makes a 250-day high
  - 40 weeks  ~= 200 trading days  -> exit  when Close makes a 200-day low
Long-only. Holds the position until the exit condition fires (trend-follow logic).

This is a SLOW trend overlay — different horizon from our daily consensus
strategies, so it diversifies rather than duplicates them.
"""

import numpy as np
import pandas as pd
from strategies import Strategy, Signal


class DonchianBreakoutStrategy(Strategy):
    """Long-only Donchian breakout: 250d high entry, 200d low exit."""

    def __init__(self, entry_lookback=250, exit_lookback=200):
        super().__init__("DC_Donchian")
        self.entry_lookback = entry_lookback
        self.exit_lookback = exit_lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        signals = np.zeros(n, dtype=int)

        high = df["High"]
        low = df["Low"]

        # prior-window extremes (shift 1 so we compare against PAST, not today)
        hh = high.rolling(self.entry_lookback, min_periods=max(20, self.entry_lookback // 2)).max().shift(1)
        ll = low.rolling(self.exit_lookback, min_periods=max(20, self.exit_lookback // 2)).min().shift(1)

        close = df["Close"]
        in_pos = False
        for i in range(1, n):
            if in_pos:
                if low.iloc[i] <= ll.iloc[i]:
                    signals[i] = Signal.SELL.value
                    in_pos = False
                continue
            if close.iloc[i] > hh.iloc[i]:
                signals[i] = Signal.BUY.value
                in_pos = True

        df["Signal"] = signals
        df["_dc_hh"] = hh
        df["_dc_ll"] = ll
        return df


def get_donchian_strategies():
    return [DonchianBreakoutStrategy()]
