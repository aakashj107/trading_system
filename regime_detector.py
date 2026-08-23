"""
Market Regime Detector — Bull/Bear/Sideways/Volatile.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from enum import Enum


class MarketRegime(Enum):
    BULL="BULL"; BEAR="BEAR"; SIDEWAYS="SIDEWAYS"; VOLATILE="VOLATILE"


class RegimeDetector:
    STRATEGY_WEIGHTS = {
        MarketRegime.BULL: {"trend":1.2, "momentum":1.2, "mean_reversion":0.5},
        MarketRegime.BEAR: {"trend":0.5, "momentum":0.3, "mean_reversion":1.2},
        MarketRegime.SIDEWAYS: {"trend":0.3, "momentum":0.3, "mean_reversion":1.5},
        MarketRegime.VOLATILE: {"trend":0.3, "momentum":0.3, "mean_reversion":0.8},
    }

    def detect_regime(self, index_data: pd.DataFrame) -> Tuple[MarketRegime, Dict]:
        df = index_data.copy()
        close = df["Close"]
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean() if len(close) >= 200 else sma50

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        h = df["High"]; l = df["Low"]
        tr = pd.concat([h-l, (h-close.shift()).abs(), (l-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = (atr / close) * 100

        lc = close.iloc[-1]; ls50 = sma50.iloc[-1]
        ls200 = sma200.iloc[-1]; lr = rsi.iloc[-1]; la = atr_pct.iloc[-1]

        details = {"close":lc, "sma_50":ls50, "sma_200":ls200, "rsi":lr, "atr_pct":la}

        if la > 2.5: regime = MarketRegime.VOLATILE
        elif lc > ls200 and ls50 > ls200: regime = MarketRegime.BULL
        elif lc < ls200 and ls50 < ls200: regime = MarketRegime.BEAR
        else: regime = MarketRegime.SIDEWAYS

        details["regime"] = regime.value
        return regime, details

    def detect(self, index_data: pd.DataFrame) -> Dict:
        """
        Convenience wrapper used by daily_runner.py.

        Returns the details dict (which already contains the "regime" key)
        rather than the (MarketRegime, dict) tuple, so callers can do
        `result.get("regime")` directly.
        """
        _regime, details = self.detect_regime(index_data)
        return details

    def print_regime_report(self, index_data):
        regime, d = self.detect_regime(index_data)
        icons = {MarketRegime.BULL:"↑", MarketRegime.BEAR:"↓",
                 MarketRegime.SIDEWAYS:"→", MarketRegime.VOLATILE:"⚡"}
        print(f"\n{'='*50}")
        print(f"  MARKET REGIME: {icons[regime]} {regime.value}")
        print(f"{'='*50}")
        print(f"  Nifty: {d['close']:,.2f}")
        print(f"  SMA50: {d['sma_50']:,.2f}")
        print(f"  SMA200: {d['sma_200']:,.2f}")
        print(f"  RSI: {d['rsi']:.1f}")
        print(f"  ATR%: {d['atr_pct']:.2f}%")
        weights = self.STRATEGY_WEIGHTS.get(regime, {})
        print(f"\n  Strategy weights ({regime.value}):")
        for k, v in weights.items():
            bar = "█" * int(v * 10)
            print(f"    {k:<16} {v:.1f}x {bar}")
        print(f"{'='*50}")
