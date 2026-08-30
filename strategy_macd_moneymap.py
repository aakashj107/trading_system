"""
strategy_macd_moneymap.py — OUR recreation of "The Secret Mindset" YouTube
MACD strategy ("MACD money map", 386K-view video, claims 71% win rate).

We do NOT trust the 71% claim. We recreate the SPECIFIED mechanical rule
(System 1: trend/continuation) and OOS-test it on our data.

Recreated rule (verbatim from transcript):
  - Zero line = direction (MACD > 0 bullish, < 0 bearish)
  - DISTANCE RULE: only take crossovers FAR from zero (above +0.5 / below -0.5
    in the video's price-scale; we normalize as |MACD|/Close > dist_thresh)
  - WAIT 2-3 candles after crossover before entering
System 2 (divergence + histogram) is subjective ("draw lines connecting tops")
and NOT faithfully automatable, so we test only System 1.

This is a single-name signal strategy (fits our Strategy interface).
"""
import numpy as np
import pandas as pd
from strategies import Strategy, Signal


class MACDMoneyMap(Strategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 dist_thresh: float = 0.003, wait: int = 2):
        super().__init__(
            f"MACD_MoneyMap_f{fast}s{slow}_d{dist_thresh}_w{wait}", "trend")
        self.fast = fast; self.slow = slow; self.signal = signal
        self.dist_thresh = dist_thresh  # |MACD|/Close must exceed this to count
        self.wait = wait

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ema_f = df["Close"].ewm(span=self.fast, adjust=False).mean()
        ema_s = df["Close"].ewm(span=self.slow, adjust=False).mean()
        macd = ema_f - ema_s
        sig = macd.ewm(span=self.signal, adjust=False).mean()
        hist = macd - sig
        df["MACD"] = macd; df["SignalLine"] = sig; df["Hist"] = hist

        # crossover direction (zero-line = trend filter)
        bull_cross = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        bear_cross = (macd < sig) & (macd.shift(1) >= sig.shift(1))
        dist = macd.abs() / df["Close"]

        pending_bull = 0; pending_bear = 0
        sig_col = [Signal.HOLD.value] * len(df)
        idx = df.index
        for i in range(1, len(df)):
            if bull_cross.iloc[i] and df["MACD"].iloc[i] > 0 and dist.iloc[i] > self.dist_thresh:
                pending_bull = self.wait  # wait N candles then enter
            elif bear_cross.iloc[i] and df["MACD"].iloc[i] < 0 and dist.iloc[i] > self.dist_thresh:
                pending_bear = self.wait

            if pending_bull > 0:
                pending_bull -= 1
                if pending_bull == 0 and df["MACD"].iloc[i] > 0:
                    sig_col[i] = Signal.BUY.value
            if pending_bear > 0:
                pending_bear -= 1
                if pending_bear == 0 and df["MACD"].iloc[i] < 0:
                    sig_col[i] = Signal.SELL.value
        df["Signal"] = sig_col
        return df
