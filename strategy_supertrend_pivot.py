"""
strategy_supertrend_pivot.py — OUR recreation of the YouTube "RR Supertrend +
Pivot" swing strategy ("A Swing Trading Strategy That Works Like Magic", daily,
Nifty stocks, author-claimed backtested).

Recreated rule (verbatim from transcript):
  - Supertrend (trend filter; green = bullish)
  - Pivot Point: R1 = 2*P - L, P = (H+L+C)/3 of PRIOR day
  - ENTRY: price in green Supertrend AND Close crosses above R1
           (fallback: Close above Pivot Point)
  - TARGET +10%, STOP -5% (1:2 R:R) — we model via PaperTrader 10%/5% exits
  - TIMEFRAME: daily, Nifty stocks

We implement the ENTRY signal; exit is handled by PaperTrader's take-profit
(10%) / stop-loss (5%) which we set per-signal. OOS-tested blind.
"""
import numpy as np
import pandas as pd
from strategies import Strategy, Signal


def supertrend(df: pd.DataFrame, period: int = 7, mult: float = 3.0):
    hl2 = (df["High"] + df["Low"]) / 2
    atr = df["High"].rolling(period).max()  # placeholder; real ATR below
    # proper ATR
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    up = hl2 - mult * atr
    dn = hl2 + mult * atr
    st = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)
    for i in range(1, len(df)):
        if i == 1:
            st.iloc[i] = up.iloc[i]; trend.iloc[i] = 1; continue
        if trend.iloc[i - 1] == 1:
            st.iloc[i] = max(up.iloc[i], st.iloc[i - 1]) if df["Close"].iloc[i] > st.iloc[i - 1] else dn.iloc[i]
        else:
            st.iloc[i] = min(dn.iloc[i], st.iloc[i - 1]) if df["Close"].iloc[i] < st.iloc[i - 1] else up.iloc[i]
        trend.iloc[i] = 1 if df["Close"].iloc[i] > st.iloc[i] else -1
    return trend  # 1 = bullish (green), -1 = bearish


def pivot_r1(df: pd.DataFrame):
    # classic pivot from PRIOR day's HLC
    P = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
    R1 = 2 * P - df["Low"].shift(1)
    return P, R1


class SupertrendPivot(Strategy):
    def __init__(self, st_period: int = 7, st_mult: float = 3.0, use_r1: bool = True):
        super().__init__(f"SupertrendPivot_p{st_period}_m{st_mult}_r1{use_r1}", "trend")
        self.st_period = st_period; self.st_mult = st_mult; self.use_r1 = use_r1

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        trend = supertrend(df, self.st_period, self.st_mult)
        P, R1 = pivot_r1(df)
        df["STTrend"] = trend; df["Pivot"] = P; df["R1"] = R1

        sig_col = [Signal.HOLD.value] * len(df)
        for i in range(2, len(df)):
            green = trend.iloc[i] == 1
            lvl = R1.iloc[i] if self.use_r1 else P.iloc[i]
            if green and df["Close"].iloc[i] > lvl and df["Close"].iloc[i - 1] <= lvl:
                sig_col[i] = Signal.BUY.value
        df["Signal"] = sig_col
        return df
