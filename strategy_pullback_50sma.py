"""
strategy_pullback_50sma.py — OUR recreation of the YouTube "Best Pullback Trading
Strategy" (RSI/SMA pullback, daily). The original is S/R+trendline+MA confluence
with subjective 'rejection candlestick' reading. We recreate the QUANTIFIABLE core:

  - TREND filter: price > 50 SMA AND price > 200 SMA (uptrend, higher-highs structure)
  - PULLBACK: price declines to touch the 50 SMA (low <= 50 SMA <= close, i.e. a tag)
  - REVERSAL candle: bullish engulfing OR hammer on that pullback bar
  - ENTRY: BUY next bar

Bearish mirror for shorts (price < 50 & 200 SMA, rally to 50 SMA, bearish engulfing).
This is OUR faithful mechanical adaptation; the 'confluence/area' subjectivity is
intentionally dropped (not faked). OOS-tested blind.
"""
import numpy as np
import pandas as pd
from strategies import Strategy, Signal


def is_bullish_engulfing(o, h, l, c, po, pc):
    return c > o and pc < po and c > po and o < pc and (c - o) > (po - pc) * 0.5


def is_hammer(o, h, l, c):
    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return body > 0 and lower_wick >= 2 * body and upper_wick <= body


class Pullback50SMA(Strategy):
    def __init__(self, fast: int = 50, slow: int = 200):
        super().__init__(f"Pullback50SMA_f{fast}_s{slow}", "trend")
        self.fast = fast; self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        sma_f = df["Close"].rolling(self.fast).mean()
        sma_s = df["Close"].rolling(self.slow).mean()
        df["SMA50"] = sma_f; df["SMA200"] = sma_s

        sig_col = [Signal.HOLD.value] * len(df)
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        for i in range(self.slow + 2, len(df)):
            uptrend = c.iloc[i] > sma_f.iloc[i] and c.iloc[i] > sma_s.iloc[i]
            dntrend = c.iloc[i] < sma_f.iloc[i] and c.iloc[i] < sma_s.iloc[i]
            # pullback tag: 50 SMA sits within this bar's range
            tag = (l.iloc[i] <= sma_f.iloc[i] <= h.iloc[i])
            if uptrend and tag:
                if is_bullish_engulfing(o.iloc[i], h.iloc[i], l.iloc[i], c.iloc[i],
                                        o.iloc[i - 1], c.iloc[i - 1]) or \
                   is_hammer(o.iloc[i], h.iloc[i], l.iloc[i], c.iloc[i]):
                    sig_col[i] = Signal.BUY.value
            elif dntrend and tag:
                if is_bullish_engulfing(o.iloc[i], h.iloc[i], l.iloc[i], c.iloc[i],
                                        o.iloc[i - 1], c.iloc[i - 1]) or \
                   is_hammer(o.iloc[i], h.iloc[i], l.iloc[i], c.iloc[i]):
                    sig_col[i] = Signal.SELL.value
        df["Signal"] = sig_col
        return df
