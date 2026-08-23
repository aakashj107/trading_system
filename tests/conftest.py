"""
Shared pytest fixtures and synthetic-data helpers.

All tests are NETWORK-FREE: they build synthetic OHLCV DataFrames so the
suite runs in CI without Yahoo Finance access and without flakiness.
"""
import numpy as np
import pandas as pd
import pytest


def make_ohlcv(seed: int = 1, periods: int = 260, drift: float = 0.0,
              vol: float = 0.014, base: float = 1000.0) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with realistic intraday range.

    Args:
        seed:   RNG seed for reproducibility.
        periods: number of daily bars.
        drift:  per-bar log-return drift (negative = downtrend).
        vol:    per-bar return stddev.
        base:   starting close price.
    """
    rng = np.random.default_rng(seed)
    price = float(base)
    opens, highs, lows, closes, vols = [], [], [], [], []
    for _ in range(periods):
        price *= (1.0 + drift + rng.normal(0, vol))
        close = price
        high = close * (1.0 + abs(rng.normal(0, 0.01)))
        low = close * (1.0 - abs(rng.normal(0, 0.01)))
        # close position within the bar's range (varies, not always midpoint)
        cp = rng.random()
        close = low + cp * (high - low)
        # volume varies day to day, occasional spikes
        v = 1e6 * (0.6 + rng.random() * 1.4) * (3.0 if rng.random() < 0.08 else 1.0)
        opens.append(close)
        highs.append(high)
        lows.append(low)
        closes.append(close)
        vols.append(v)
    idx = pd.date_range("2022-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows,
         "Close": closes, "Volume": vols},
        index=idx,
    )


@pytest.fixture
def ohlc_uptrend():
    return make_ohlcv(seed=11, drift=+0.0015, periods=260)


@pytest.fixture
def ohlc_downtrend():
    return make_ohlcv(seed=22, drift=-0.0016, periods=260)


@pytest.fixture
def ohlc_sharp_crash():
    """A strong, volatile downtrend with volume spikes — guarantees the
    bear strategies have the EMA-cross / breakdown / overbought-bounce
    setups they need to actually fire SHORT signals."""
    rng = np.random.default_rng(5)
    price = 1000.0
    opens, highs, lows, closes, vols = [], [], [], [], []
    for i in range(300):
        # sharp sustained downtrend with noise
        price *= (1.0 - 0.0035 + rng.normal(0, 0.02))
        close = price
        high = close * (1.0 + abs(rng.normal(0, 0.015)))
        low = close * (1.0 - abs(rng.normal(0, 0.015)))
        cp = rng.random()
        close = low + cp * (high - low)
        # frequent volume spikes (needed by volume-confirmation filters)
        v = 1e6 * (0.6 + rng.random() * 1.4) * (3.0 if rng.random() < 0.25 else 1.0)
        opens.append(close); highs.append(high); lows.append(low)
        closes.append(close); vols.append(v)
    idx = pd.date_range("2022-01-01", periods=300, freq="D")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows,
         "Close": closes, "Volume": vols}, index=idx)


@pytest.fixture
def universe_cache():
    """A small in-memory data_cache dict for parallel-scan tests."""
    return {f"STK{i}": make_ohlcv(seed=10 + i, periods=260) for i in range(8)}
