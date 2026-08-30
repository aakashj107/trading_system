"""
strategy_opening_range_breakout.py — OUR DAILY adaptation of the "One Candle"
scalping opening-range concept (ProRealAlgos YouTube, 1.7M views).

HONEST SCOPE NOTE: the original is a 15-min/Nasdaq INTRADAY strategy requiring
5-15min bars + a non-Indian asset we don't have. We cannot test it faithfully in
our daily .NS system. What IS mechanically portable is the CORE IDEA:
  - define an opening range (first session bar high/low)
  - trade breakouts of that range in the direction of the prevailing trend
We adapt it to DAILY bars: opening range = first N days' high/low after a
trend filter; long when price breaks above range high in uptrend, short below
range low in downtrend. This is OUR adaptation, labeled as such — NOT the original.

(We do NOT test the "liquidity candle / 25% ATR / hammer reversal" parts — those
are intraday-specific and unfaithful on daily data.)
"""
import numpy as np
import pandas as pd
from strategies import Strategy, Signal


class OpeningRangeBreakout(Strategy):
    def __init__(self, range_days: int = 5, trend_sma: int = 50):
        super().__init__(f"OpeningRangeBreakout_r{range_days}_sma{trend_sma}", "trend")
        self.range_days = range_days
        self.trend_sma = trend_sma

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"]; high = df["High"]; low = df["Low"]
        sma = close.rolling(self.trend_sma).mean()
        # opening range = rolling high/low of last `range_days` bars
        rng_hi = high.rolling(self.range_days).max()
        rng_lo = low.rolling(self.range_days).min()
        df["RangeHigh"] = rng_hi; df["RangeLow"] = rng_lo; df["TrendSMA"] = sma

        sig_col = [Signal.HOLD.value] * len(df)
        for i in range(self.range_days + 1, len(df)):
            if close.iloc[i] > sma.iloc[i] and close.iloc[i] > rng_hi.iloc[i - 1]:
                sig_col[i] = Signal.BUY.value
            elif close.iloc[i] < sma.iloc[i] and close.iloc[i] < rng_lo.iloc[i - 1]:
                sig_col[i] = Signal.SELL.value
        df["Signal"] = sig_col
        return df
