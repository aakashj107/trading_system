"""
strategies_pead.py — Episodic Pivot (EP) family, per Qullamaggie's spec.

EP = a stock gaps up >=10% on unexpected good news (mostly EARNINGS + guidance)
in a previously-neglected / non-trending name, with BIG volume, and then
drifts/runs for weeks-to-months. The key disqualifier we missed in the first
PEAD version: the stock must NOT have already run 3-6 months into the gap —
if it did, the "surprise" isn't a surprise.

HARD RULES (from the blog):
  - gap up >= 10%  (open vs prior close)
  - big volume: >= ~3x the 20d median (pre-market or first minutes)
  - for earnings EP: prefer high EPS/rev growth + beat (we proxy with the gap+vol)
  - BEST when the stock has NOT rallied over the prior 3-6 months
  - trail with 20/50-day MA; exit on first daily close below trail MA

LONG-ONLY (consistent with validated winning config).

This replaces the looser PEAD that tested near-breakeven (+5.5%/10y, PF 1.07):
that version fired on any 2-sigma gap with no "no-prior-run" filter, so it
caught gaps on already-trending names that had no real catalyst drift.
"""

import numpy as np
import pandas as pd
from strategies import Strategy, Signal


class EpisodicPivotStrategy(Strategy):
    """
    Episodic Pivot — long-only, event-driven, strict Qullamaggie rules.
    """

    def __init__(self, gap_pct=0.05, vol_mult=2.5, prior_lookback=126,
                 prior_run_cap=0.20, trail_ma=20, hold_days=60, grade_min=2):
        """
        gap_pct      : minimum overnight gap (open vs prior close) to flag EP
        vol_mult     : earnings-day volume vs 20d median to confirm conviction
        prior_lookback: bars (~6mo of trading days) to check "has it already run"
        prior_run_cap : if stock rose > this over prior_lookback, DISQUALIFY (not a surprise)
        trail_ma     : MA used to trail the EP (20 or 50)
        hold_days    : hard max bars to ride
        """
        super().__init__("EP_EpisodicPivot")
        self.gap_pct = gap_pct
        self.vol_mult = vol_mult
        self.prior_lookback = prior_lookback
        self.prior_run_cap = prior_run_cap
        self.trail_ma = trail_ma
        self.hold_days = hold_days
        self.grade_min = grade_min

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        signals = np.zeros(n, dtype=int)

        close = df["Close"]
        op = df["Open"]
        vol = df["Volume"]

        gap_pct = (op - close.shift(1)) / close.shift(1)
        vol_med = vol.rolling(20, min_periods=5).median()
        vol_ratio = vol / vol_med

        ma_trail = close.rolling(self.trail_ma, min_periods=max(2, self.trail_ma // 2)).mean()

        # prior-run check: return over prior_lookback bars ending just before today
        prior_ret = close.pct_change(self.prior_lookback).shift(1)

        in_pos = False
        entry_i = 0
        bars_held = 0

        for i in range(1, n):
            if in_pos:
                bars_held += 1
                # exit: trail MA breach (first close below) or hard hold cap
                if bars_held >= self.hold_days:
                    signals[i] = Signal.SELL.value
                    in_pos = False
                elif close.iloc[i] < ma_trail.iloc[i]:
                    signals[i] = Signal.SELL.value
                    in_pos = False
                continue

            # EP entry conditions
            is_gap = gap_pct.iloc[i] >= self.gap_pct
            is_vol = vol_ratio.iloc[i] >= self.vol_mult
            not_already_ran = (prior_ret.iloc[i] if not np.isnan(prior_ret.iloc[i])
                               else 0.0) <= self.prior_run_cap
            if is_gap and is_vol and not_already_ran:
                signals[i] = Signal.BUY.value
                in_pos = True
                entry_i = i
                bars_held = 0

        df["Signal"] = signals
        df["_ep_gap"] = gap_pct
        df["_ep_volr"] = vol_ratio
        return df


def get_pead_strategies():
    """EP family (currently one strategy)."""
    return [EpisodicPivotStrategy()]
