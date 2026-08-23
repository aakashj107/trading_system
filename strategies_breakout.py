"""
strategies_breakout.py — Qullamaggie Breakout family.

PATTERN (contraction -> expansion):
  1. A big move higher (30-100%+) sometime in the past 1-3 months (the "leader").
  2. An orderly pullback / consolidation: HIGHER LOWS, TIGHTENING range,
     price "surfs" the rising 10- and 20-day MA, for ~2 weeks to 2 months.
  3. A range EXPANSION (breakout) out of the base on above-average volume.

TRADE (from the blog, adapted to daily bars + our PaperTrader):
  - Enter on breakout above the consolidation pivot (base high) with volume.
  - Stop: low of the breakout day, never wider than ATR (our PaperTrader does this).
  - Scale 1/3-1/2 after 3-5 days, trail rest on 10-day MA (first daily close
    below = exit). We encode: exit on first close below the 10-day MA after the
    scale-out window, or on a hard hold cap.

LONG-ONLY (validated winning config). Fits our technical book directly.
"""

import numpy as np
import pandas as pd
from strategies import Strategy, Signal


def _highest_lowest_tightening(highs, lows, look):
    """Return True if the last `look` bars show higher lows AND a tightening
    range (last half range < first half range)."""
    if len(highs) < look:
        return False
    seg_h = highs.iloc[-look:]
    seg_l = lows.iloc[-look:]
    # higher lows: compare successive pivot lows (use rolling min of 3-bar)
    roll_min = seg_l.rolling(3, min_periods=1).min()
    higher_lows = roll_min.iloc[-1] >= roll_min.iloc[len(roll_min) // 2]
    # tightening: range of 2nd half < range of 1st half
    first_half = (seg_h - seg_l).iloc[: len(seg_h) // 2].mean()
    second_half = (seg_h - seg_l).iloc[len(seg_h) // 2:].mean()
    tightening = second_half < first_half
    return bool(higher_lows and tightening)


class BreakoutStrategy(Strategy):
    """
    Contraction-breakout on the daily chart. Long-only.
    """

    def __init__(self, base_min=10, base_max=40, vol_mult=2.0,
                 prior_up_min=0.15, trail_ma=10, scale_days=4, hold_days=80):
        """
        base_min/max : consolidation length window (bars) to require
        vol_mult     : breakout-day volume vs 20d median
        prior_up_min : stock must be up >= this over prior ~3mo to be a "leader"
        trail_ma     : MA used to trail after scale-out
        scale_days   : bars after entry before trailing exit arms
        hold_days    : hard max bars
        """
        super().__init__("BO_Breakout")
        self.base_min = base_min
        self.base_max = base_max
        self.vol_mult = vol_mult
        self.prior_up_min = prior_up_min
        self.trail_ma = trail_ma
        self.scale_days = scale_days
        self.hold_days = hold_days

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        signals = np.zeros(n, dtype=int)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        vol = df["Volume"]

        ma10 = close.rolling(10, min_periods=5).mean()
        ma20 = close.rolling(20, min_periods=10).mean()
        ma20_slope = ma20.diff(5) / ma20
        vol_med = vol.rolling(20, min_periods=5).median()
        vol_ratio = vol / vol_med

        # "leader" check: up >= prior_up_min over ~63 bars before window
        prior_up = close.pct_change(63).shift(1)

        # base pivot = highest high of the consolidation window ending yesterday
        base_high = high.rolling(self.base_max, min_periods=self.base_min).max().shift(1)
        # base low for range tightening
        base_low = low.rolling(self.base_max, min_periods=self.base_min).min().shift(1)

        in_pos = False
        entry_i = 0
        bars_held = 0
        scaled = False

        for i in range(self.base_max, n):
            if in_pos:
                bars_held += 1
                if bars_held >= self.hold_days:
                    signals[i] = Signal.SELL.value
                    in_pos = False
                    scaled = False
                elif bars_held >= self.scale_days and close.iloc[i] < ma10.iloc[i]:
                    # trail on 10-day MA after scale-out window
                    signals[i] = Signal.SELL.value
                    in_pos = False
                    scaled = False
                continue

            # Entry: breakout above base high, on volume, with a valid base
            is_leader = (prior_up.iloc[i] if not np.isnan(prior_up.iloc[i]) else 0) >= self.prior_up_min
            ma_rising = ma20_slope.iloc[i] > 0
            surfing_ma = close.iloc[i] >= ma20.iloc[i] * 0.97  # near/rising 20d
            breakout = close.iloc[i] > base_high.iloc[i] * 1.001
            vol_ok = vol_ratio.iloc[i] >= self.vol_mult
            # require a real consolidation existed (range bounded)
            range_ok = (base_high.iloc[i] - base_low.iloc[i]) / base_low.iloc[i] < 0.40

            if is_leader and ma_rising and surfing_ma and breakout and vol_ok and range_ok:
                signals[i] = Signal.BUY.value
                in_pos = True
                entry_i = i
                bars_held = 0
                scaled = False

        df["Signal"] = signals
        df["_bo_base_high"] = base_high
        df["_bo_volr"] = vol_ratio
        return df


def get_breakout_strategies():
    return [BreakoutStrategy()]
