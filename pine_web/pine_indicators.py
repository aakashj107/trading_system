"""
pine_indicators.py — Python equivalents of the common Pine ta.* toolkit.
Most math is imported from strategies_pine (already validated against TV);
a few extras (MACD, cross, donchian, stochastic, ROC, CCI, MFI) are added here.
"""

import numpy as np
import pandas as pd

# reuse the validated primitives
from .strategies_pine_bridge import (
    _ema as ema, _sma as sma, _rsi as rsi, _atr as atr, _adx as adx,
    _linreg as linreg, hma, alma, kama, vwma, _rma as rma,
)


def wma(series, period):
    """Wilder/weighted MA (Pine ta.wma)."""
    w = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, w) / w.sum(), raw=True)


def highest(series, period):
    return series.rolling(period, min_periods=1).max()


def lowest(series, period):
    return series.rolling(period, min_periods=1).min()


def donchian(high, low, period):
    """Returns (upper, lower, middle) Donchian channels."""
    up = highest(high, period)
    lo = lowest(low, period)
    mid = (up + lo) / 2
    return up, lo, mid


def cross(a, b):
    """ta.cross(a,b): True on bars where a crosses ABOVE b."""
    a, b = a.astype(float), b.astype(float)
    prev = (a.shift(1) <= b.shift(1)) & (a > b)
    return prev.fillna(False).astype(bool)


def crossunder(a, b):
    a, b = a.astype(float), b.astype(float)
    prev = (a.shift(1) >= b.shift(1)) & (a < b)
    return prev.fillna(False).astype(bool)


def macd(close, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram)."""
    ef, es = ema(close, fast), ema(close, slow)
    macd_line = ef - es
    sig = ema(macd_line, signal)
    return macd_line, sig, macd_line - sig


def stoch(high, low, close, k_period=14, d_period=3, smooth=3):
    """Stochastic %K / %D (Pine ta.stoch)."""
    hh = highest(high, k_period)
    # rolling high over k_period (tv uses k_period lookback on highs)
    ll = lowest(low, k_period)
    hh = high.rolling(k_period, min_periods=1).max()
    ll = low.rolling(k_period, min_periods=1).min()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    ks = k.rolling(smooth, min_periods=1).mean()
    d = ks.rolling(d_period, min_periods=1).mean()
    return ks, d


def roc(series, period=12):
    return series.pct_change(period) * 100


def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(period, min_periods=1).mean()
    md = (tp - ma).abs().rolling(period, min_periods=1).mean()
    return (tp - ma) / (0.015 * md).replace(0, np.nan)


def mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    rmf = tp * volume
    up = tp > tp.shift(1)
    down = tp < tp.shift(1)
    pos = rmf.where(up, 0.0).rolling(period, min_periods=1).sum()
    neg = rmf.where(down, 0.0).rolling(period, min_periods=1).sum()
    mr = pos / neg.replace(0, np.nan)
    return 100 - (100 / (1 + mr))


def supertrend(high, low, close, atr_period=10, multiplier=3.0):
    """Returns (supertrend_series, direction(1 up/-1 down))."""
    atr_s = atr(high, low, close, atr_period)
    hl2 = (high + low) / 2
    up = hl2 + multiplier * atr_s
    dn = hl2 - multiplier * atr_s
    n = len(close)
    st = np.full(n, np.nan)
    dirn = np.full(n, 1)
    c = close.values
    upv, dnv, atv = up.values, dn.values, atr_s.values
    st[0], dirn[0] = dnv[0], -1
    for i in range(1, n):
        if c[i] > upv[i - 1]:
            dirn[i] = 1
        elif c[i] < dnv[i - 1]:
            dirn[i] = -1
        else:
            dirn[i] = dirn[i - 1]
            if dirn[i] == 1 and dnv[i] < dnv[i - 1]:
                dnv[i] = dnv[i - 1]
            if dirn[i] == -1 and upv[i] > upv[i - 1]:
                upv[i] = upv[i - 1]
        st[i] = dnv[i] if dirn[i] == -1 else upv[i]
    return pd.Series(st, index=close.index), pd.Series(dirn, index=close.index)
