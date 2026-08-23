"""
Indian Market Trading System — Strategy Engine
v3.1 — Short-selling support, bear-specific strategies, regime awareness.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
from config import BQ_CONFIG,ADX_GATE_CONFIG,EXHAUSTION_CONFIG,TREND_RANGE_CONFIG

# ============================================================
#  PHASE 1A: BREAK QUALITY FILTER
#  Place after imports, before Signal class
# ============================================================


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    """
    Compute Average True Range using Wilder's smoothing (RMA).
    Matches TradingView's ta.atr() behavior.

    If your system already computes ATR elsewhere, you can
    replace calls to this function with your existing one.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing (RMA) — equivalent to EMA with alpha = 1/period
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period,
                          adjust=False).mean()
    return atr


class BreakQualityFilter:
    """
    Universal candle-quality filter.

    Evaluates candle "conviction" by checking:
    1. Body Ratio: |Close - Open| / (High - Low)
       — How much of the candle range is actual directional body vs wicks.
       — A doji (tiny body, long wicks) has a ratio near 0.
       — A marubozu (full body, tiny wicks) has a ratio near 1.

    2. Body / ATR: |Close - Open| / ATR(14)
       — Ensures the candle body has substance relative to recent volatility.
       — A big-looking candle on a volatile stock may actually be normal.

    Usage:
        bq = BreakQualityFilter()

        # Check one bar
        result = bq.check_bar(df, bar_index=50)

        # Filter entire signal series
        filtered = bq.filter_signal_series(df, raw_signals)

        # Get quality report for debugging
        report = bq.get_signal_quality_report(df, raw_signals)
    """

    def __init__(self, config=None):
        self.config = config or BQ_CONFIG
        self._atr_cache = {}

    def _get_atr(self, df: pd.DataFrame) -> pd.Series:
        """Get or compute ATR with caching to avoid recomputation."""
        cache_key = (id(df), self.config.atr_length)
        if cache_key not in self._atr_cache:
            self._atr_cache[cache_key] = compute_atr(
                df['High'], df['Low'], df['Close'],
                self.config.atr_length
            )
        return self._atr_cache[cache_key]

    def _compute_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute body ratio and body/ATR for every bar."""
        body_abs = (df['Close'] - df['Open']).abs()
        candle_range = (df['High'] - df['Low']).replace(0, np.nan)
        body_ratio = body_abs / candle_range

        atr = self._get_atr(df)
        body_atr_mult = body_abs / atr.replace(0, np.nan)

        return pd.DataFrame({
            'body_ratio': body_ratio,
            'body_abs': body_abs,
            'atr': atr,
            'body_atr_mult': body_atr_mult
        }, index=df.index)

    def _get_thresholds(self):
        """Get the active thresholds based on mode."""
        if self.config.mode == "HARD":
            return self.config.hard_min_body_ratio, self.config.hard_min_body_atr
        else:
            return self.config.soft_min_body_ratio, self.config.soft_min_body_atr

    def check_bar(self, df: pd.DataFrame, bar_index: int) -> dict:
        """
        Check if a specific bar passes the quality filter.

        Returns dict with keys:
            passed:      bool — True if signal should be counted
            mode:        str  — "OFF", "SOFT", "HARD"
            body_ratio:  float — the candle's body ratio
            body_atr_mult: float — body size as multiple of ATR
            reason:      str  — "BQ_OK", "LOW_BODY_RATIO", "LOW_BODY_ATR", etc.
            weak:        bool — True if SOFT mode and candle failed quality
        """
        if self.config.mode == "OFF":
            return {
                'passed': True, 'mode': 'OFF', 'weak': False,
                'body_ratio': None, 'body_atr_mult': None,
                'reason': 'BQ_OFF'
            }

        metrics = self._compute_metrics(df)

        if bar_index >= len(metrics):
            return {
                'passed': False, 'mode': self.config.mode, 'weak': False,
                'body_ratio': None, 'body_atr_mult': None,
                'reason': 'INDEX_OUT_OF_RANGE'
            }

        row = metrics.iloc[bar_index]
        body_ratio = row['body_ratio']
        body_atr_mult = row['body_atr_mult']

        min_ratio, min_atr = self._get_thresholds()

        # Evaluate
        if pd.isna(body_ratio) or pd.isna(body_atr_mult):
            reason = 'INSUFFICIENT_DATA'
            passed = (self.config.mode == "SOFT")
        elif body_ratio < min_ratio:
            reason = 'LOW_BODY_RATIO'
            passed = (self.config.mode == "SOFT")
        elif body_atr_mult < min_atr:
            reason = 'LOW_BODY_ATR'
            passed = (self.config.mode == "SOFT")
        else:
            reason = 'BQ_OK'
            passed = True

        weak = (self.config.mode == "SOFT" and reason != 'BQ_OK')

        return {
            'passed': passed,
            'mode': self.config.mode,
            'body_ratio': round(body_ratio, 4) if not pd.isna(body_ratio) else None,
            'body_atr_mult': round(body_atr_mult, 4) if not pd.isna(body_atr_mult) else None,
            'reason': reason,
            'weak': weak
        }

    def filter_signal_series(self, df: pd.DataFrame,
                              signals: pd.Series) -> pd.Series:
        """
        Filter a signal series through break quality.

        Args:
            df:      OHLCV DataFrame
            signals: Series with values 1 (BUY), -1 (SELL), 0 (HOLD),
                     and optionally -2 (SHORT), 2 (COVER)

        Returns:
            Filtered signal series:
            - HARD mode: signals on weak candles are zeroed out
            - SOFT mode: signals unchanged (weak tagging done elsewhere)
            - OFF mode:  signals unchanged
        """
        if self.config.mode == "OFF":
            return signals.copy()

        filtered = signals.copy()
        metrics = self._compute_metrics(df)

        if self.config.mode == "HARD":
            min_ratio, min_atr = self._get_thresholds()

            signal_mask = filtered != 0
            fail_mask = (
                signal_mask & (
                    (metrics['body_ratio'] < min_ratio) |
                    (metrics['body_atr_mult'] < min_atr) |
                    metrics['body_ratio'].isna() |
                    metrics['body_atr_mult'].isna()
                )
            )
            filtered[fail_mask] = 0

        return filtered

    def get_signal_quality_report(self, df: pd.DataFrame,
                                    signals: pd.Series) -> pd.DataFrame:
        """
        Generate quality report for all signal bars.
        Useful for debugging — shows which signals were weak and why.
        """
        metrics = self._compute_metrics(df)
        signal_mask = signals != 0

        report = pd.DataFrame({
            'signal': signals,
            'direction': signals.map({
                1: 'BUY', -1: 'SELL', -2: 'SHORT', 2: 'COVER', 0: 'NEUTRAL'
            }),
            'body_ratio': metrics['body_ratio'],
            'body_atr_mult': metrics['body_atr_mult'],
            'atr': metrics['atr'],
        }, index=df.index)

        report = report[signal_mask].copy()

        if self.config.mode == "OFF":
            report['quality'] = 'BQ_OFF'
            report['passed'] = True
            report['weak'] = False
        else:
            min_ratio, min_atr = self._get_thresholds()

            conditions = [
                report['body_ratio'].isna() | report['body_atr_mult'].isna(),
                report['body_ratio'] < min_ratio,
                report['body_atr_mult'] < min_atr,
            ]
            choices = ['INSUFFICIENT_DATA', 'LOW_BODY_RATIO', 'LOW_BODY_ATR']
            report['quality'] = np.select(conditions, choices, default='BQ_OK')

            if self.config.mode == "HARD":
                report['passed'] = (report['quality'] == 'BQ_OK')
            else:
                report['passed'] = True  # SOFT always passes

            report['weak'] = (report['quality'] != 'BQ_OK')

        return report

class Signal(Enum):
    BUY = 1       # Open long position
    SELL = -1     # Close long position / bearish signal
    HOLD = 0      # No action
    SHORT = -2    # NEW: Open short position (profit when price falls)
    COVER = 2     # NEW: Close short position (buy to cover)


class Strategy(ABC):
    def __init__(self, name: str, category: str = "unknown"):
        self.name = name
        self.category = category

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def __repr__(self):
        return f"Strategy({self.name})"


# ============================================================
#  ORIGINAL STRATEGIES (unchanged logic, added category)
# ============================================================

class SMACrossover(Strategy):
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(f"SMA_Crossover_{fast_period}_{slow_period}", "trend")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["SMA_Fast"] = df["Close"].rolling(window=self.fast_period).mean()
        df["SMA_Slow"] = df["Close"].rolling(window=self.slow_period).mean()
        df["Signal"] = Signal.HOLD.value
        df.loc[
            (df["SMA_Fast"] > df["SMA_Slow"]) &
            (df["SMA_Fast"].shift(1) <= df["SMA_Slow"].shift(1)),
            "Signal"
        ] = Signal.BUY.value
        df.loc[
            (df["SMA_Fast"] < df["SMA_Slow"]) &
            (df["SMA_Fast"].shift(1) >= df["SMA_Slow"].shift(1)),
            "Signal"
        ] = Signal.SELL.value
        return df


class EMACrossover(Strategy):
    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        super().__init__(f"EMA_Crossover_{fast_period}_{slow_period}", "trend")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["EMA_Fast"] = df["Close"].ewm(span=self.fast_period, adjust=False).mean()
        df["EMA_Slow"] = df["Close"].ewm(span=self.slow_period, adjust=False).mean()
        df["Signal"] = Signal.HOLD.value
        df.loc[
            (df["EMA_Fast"] > df["EMA_Slow"]) &
            (df["EMA_Fast"].shift(1) <= df["EMA_Slow"].shift(1)),
            "Signal"
        ] = Signal.BUY.value
        df.loc[
            (df["EMA_Fast"] < df["EMA_Slow"]) &
            (df["EMA_Fast"].shift(1) >= df["EMA_Slow"].shift(1)),
            "Signal"
        ] = Signal.SELL.value
        return df


class RSIStrategy(Strategy):
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(f"RSI_{period}_{oversold}_{overbought}", "mean_reversion")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        df["Signal"] = Signal.HOLD.value
        df.loc[
            (df["RSI"] > self.oversold) & (df["RSI"].shift(1) <= self.oversold),
            "Signal"
        ] = Signal.BUY.value
        df.loc[
            (df["RSI"] < self.overbought) & (df["RSI"].shift(1) >= self.overbought),
            "Signal"
        ] = Signal.SELL.value
        return df


class MACDStrategy(Strategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9):
        super().__init__(f"MACD_{fast}_{slow}_{signal_period}", "momentum")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ema_fast = df["Close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.slow, adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD"].ewm(span=self.signal_period, adjust=False).mean()
        df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
        df["Signal"] = Signal.HOLD.value
        df.loc[
            (df["MACD"] > df["MACD_Signal"]) &
            (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1)),
            "Signal"
        ] = Signal.BUY.value
        df.loc[
            (df["MACD"] < df["MACD_Signal"]) &
            (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1)),
            "Signal"
        ] = Signal.SELL.value
        return df


class BollingerBands(Strategy):
    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(f"Bollinger_{period}_{num_std}", "mean_reversion")
        self.period = period
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["BB_Middle"] = df["Close"].rolling(window=self.period).mean()
        rolling_std = df["Close"].rolling(window=self.period).std()
        df["BB_Upper"] = df["BB_Middle"] + (self.num_std * rolling_std)
        df["BB_Lower"] = df["BB_Middle"] - (self.num_std * rolling_std)
        df["Signal"] = Signal.HOLD.value
        df.loc[
            (df["Close"] > df["BB_Lower"]) &
            (df["Close"].shift(1) <= df["BB_Lower"].shift(1)),
            "Signal"
        ] = Signal.BUY.value
        df.loc[
            (df["Close"] < df["BB_Upper"]) &
            (df["Close"].shift(1) >= df["BB_Upper"].shift(1)),
            "Signal"
        ] = Signal.SELL.value
        return df


class BreakoutStrategy(Strategy):
    def __init__(self, lookback: int = 20, volume_multiplier: float = 1.5):
        super().__init__(f"Breakout_{lookback}_Vol{volume_multiplier}", "trend")
        self.lookback = lookback
        self.volume_multiplier = volume_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Channel_High"] = df["High"].rolling(window=self.lookback).max().shift(1)
        df["Channel_Low"] = df["Low"].rolling(window=self.lookback).min().shift(1)
        df["Avg_Volume"] = df["Volume"].rolling(window=self.lookback).mean()
        df["Signal"] = Signal.HOLD.value
        df.loc[
            (df["Close"] > df["Channel_High"]) &
            (df["Volume"] > df["Avg_Volume"] * self.volume_multiplier),
            "Signal"
        ] = Signal.BUY.value
        df.loc[
            (df["Close"] < df["Channel_Low"]),
            "Signal"
        ] = Signal.SELL.value
        return df


class MeanReversion(Strategy):
    def __init__(self, period: int = 50, entry_std: float = 2.0, exit_std: float = 0.5):
        super().__init__(f"MeanReversion_{period}_{entry_std}std", "mean_reversion")
        self.period = period
        self.entry_std = entry_std
        self.exit_std = exit_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["MA"] = df["Close"].rolling(window=self.period).mean()
        df["Std"] = df["Close"].rolling(window=self.period).std()
        df["Z_Score"] = (df["Close"] - df["MA"]) / df["Std"]
        df["Signal"] = Signal.HOLD.value
        df.loc[
            (df["Z_Score"] < -self.entry_std) &
            (df["Z_Score"].shift(1) >= -self.entry_std),
            "Signal"
        ] = Signal.BUY.value
        df.loc[
            (df["Z_Score"] > -self.exit_std) &
            (df["Z_Score"].shift(1) <= -self.exit_std),
            "Signal"
        ] = Signal.SELL.value
        return df


class MomentumStrategy(Strategy):
    def __init__(self, roc_period: int = 20, volume_period: int = 20):
        super().__init__(f"Momentum_ROC{roc_period}", "momentum")
        self.roc_period = roc_period
        self.volume_period = volume_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ROC"] = ((df["Close"] - df["Close"].shift(self.roc_period))
                     / df["Close"].shift(self.roc_period)) * 100
        df["Vol_Avg"] = df["Volume"].rolling(window=self.volume_period).mean()
        df["Signal"] = Signal.HOLD.value
        df.loc[
            (df["ROC"] > 0) & (df["ROC"].shift(1) <= 0) &
            (df["Volume"] > df["Vol_Avg"]),
            "Signal"
        ] = Signal.BUY.value
        df.loc[
            (df["ROC"] < 0) & (df["ROC"].shift(1) >= 0),
            "Signal"
        ] = Signal.SELL.value
        return df


# ============================================================
#  NEW: BEAR-SPECIFIC SHORT-SELLING STRATEGIES
# ============================================================

class BearTrendShort(Strategy):
    """
    Bear Trend Short — Short when price breaks below key moving average
    with volume confirmation. Covers when price reclaims the MA.

    Logic:
    - SHORT: Price crosses below EMA20 + RSI < 45 + Volume > average
    - COVER: Price crosses above EMA20 OR RSI < 25 (oversold bounce)

    Best for: Sustained downtrends with momentum
    """
    def __init__(self, ema_period: int = 20, rsi_period: int = 14,
                 rsi_short_below: int = 45, rsi_cover_below: int = 25,
                 volume_mult: float = 1.2):
        super().__init__(
            f"BearTrendShort_EMA{ema_period}_RSI{rsi_short_below}",
            "bear_short"
        )
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.rsi_short_below = rsi_short_below
        self.rsi_cover_below = rsi_cover_below
        self.volume_mult = volume_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMA
        df["EMA"] = df["Close"].ewm(span=self.ema_period, adjust=False).mean()

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Volume average
        df["Vol_Avg"] = df["Volume"].rolling(window=20).mean()

        df["Signal"] = Signal.HOLD.value

        # Stateful SHORT/COVER pairing (no overlap):
        #   SHORT only fires when flat; COVER only fires when already short.
        in_short = False
        for i in range(1, len(df)):
            if not in_short:
                # SHORT: EMA cross-down + RSI bearish + volume confirmation
                if (df["Close"].iloc[i] < df["EMA"].iloc[i] and
                        df["Close"].iloc[i - 1] >= df["EMA"].iloc[i - 1] and
                        df["RSI"].iloc[i] < self.rsi_short_below and
                        df["Volume"].iloc[i] > df["Vol_Avg"].iloc[i] * self.volume_mult):
                    df.iloc[i, df.columns.get_loc("Signal")] = Signal.SHORT.value
                    in_short = True
            else:
                # COVER: price reclaims EMA (uptrend) or deeply oversold bounce
                if (df["Close"].iloc[i] > df["EMA"].iloc[i] or
                        df["RSI"].iloc[i] < self.rsi_cover_below):
                    df.iloc[i, df.columns.get_loc("Signal")] = Signal.COVER.value
                    in_short = False

        return df


class BearMomentumShort(Strategy):
    """
    Bear Momentum Short — Short when MACD turns bearish with histogram
    expansion. Covers on MACD crossover or histogram contraction.

    Logic:
    - SHORT: MACD crosses below signal + histogram expanding negative + below SMA50
    - COVER: MACD crosses above signal OR histogram 3 consecutive contractions

    Best for: Momentum-driven selloffs
    """
    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9,
                 sma_period: int = 50):
        super().__init__(
            f"BearMomentumShort_MACD_{fast}_{slow}_{signal_period}",
            "bear_short"
        )
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period
        self.sma_period = sma_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # MACD
        ema_fast = df["Close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.slow, adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD"].ewm(span=self.signal_period, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # Histogram direction (expanding or contracting)
        df["Hist_Abs"] = df["MACD_Hist"].abs()
        df["Hist_Expanding"] = df["Hist_Abs"] > df["Hist_Abs"].shift(1)

        # Consecutive histogram contraction count
        df["Hist_Contracting"] = df["Hist_Abs"] < df["Hist_Abs"].shift(1)
        df["Contract_Count"] = 0
        count = 0
        for i in range(len(df)):
            if df["Hist_Contracting"].iloc[i]:
                count += 1
            else:
                count = 0
            df.iloc[i, df.columns.get_loc("Contract_Count")] = count

        # SMA for trend filter
        df["SMA50"] = df["Close"].rolling(window=self.sma_period).mean()

        df["Signal"] = Signal.HOLD.value

        # Stateful SHORT/COVER pairing (no overlap):
        #   SHORT only fires when flat; COVER only fires when already short.
        in_short = False
        for i in range(1, len(df)):
            if not in_short:
                # SHORT: MACD bearish crossover + histogram expanding + below SMA50
                if (df["MACD"].iloc[i] < df["MACD_Signal"].iloc[i] and
                        df["MACD"].iloc[i - 1] >= df["MACD_Signal"].iloc[i - 1] and
                        df["Hist_Expanding"].iloc[i] and
                        df["MACD_Hist"].iloc[i] < 0 and
                        df["Close"].iloc[i] < df["SMA50"].iloc[i]):
                    df.iloc[i, df.columns.get_loc("Signal")] = Signal.SHORT.value
                    in_short = True
            else:
                # COVER: MACD bullish crossover OR 3+ bars histogram contraction
                if ((df["MACD"].iloc[i] > df["MACD_Signal"].iloc[i] and
                        df["MACD"].iloc[i - 1] <= df["MACD_Signal"].iloc[i - 1]) or
                        df["Contract_Count"].iloc[i] >= 3):
                    df.iloc[i, df.columns.get_loc("Signal")] = Signal.COVER.value
                    in_short = False

        return df


class BearBreakdownShort(Strategy):
    """
    Bear Breakdown Short — Short on high-volume breakdowns below key support.
    Covers when price reclaims the breakdown level.

    Logic:
    - SHORT: Price breaks below 20-day low + volume > 1.5x average + close near low
    - COVER: Price reclaims breakdown level OR 5% profit OR RSI < 20

    Best for: Sharp breakdown moves with volume
    """
    def __init__(self, lookback: int = 20, volume_mult: float = 1.5,
                 profit_target: float = 0.05, rsi_period: int = 14):
        super().__init__(
            f"BearBreakdownShort_LB{lookback}_Vol{volume_mult}",
            "bear_short"
        )
        self.lookback = lookback
        self.volume_mult = volume_mult
        self.profit_target = profit_target
        self.rsi_period = rsi_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Support level (20-day low)
        df["Support"] = df["Low"].rolling(window=self.lookback).min().shift(1)

        # Volume
        df["Vol_Avg"] = df["Volume"].rolling(window=self.lookback).mean()

        # Close position relative to range (near low = bearish)
        df["Range"] = df["High"] - df["Low"]
        df["Close_Position"] = np.where(
            df["Range"] > 0,
            (df["Close"] - df["Low"]) / df["Range"],
            0.5
        )

        # RSI for oversold cover
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        df["Signal"] = Signal.HOLD.value

        # Track breakdown level for profit target
        df["Breakdown_Level"] = 0.0
        in_short = False
        breakdown_price = 0

        for i in range(1, len(df)):
            prev_signal = df["Signal"].iloc[i - 1]

            # Track state
            if prev_signal == Signal.SHORT.value:
                in_short = True
                breakdown_price = df["Support"].iloc[i - 1]
            elif prev_signal == Signal.COVER.value:
                in_short = False
                breakdown_price = 0

            if in_short:
                df.iloc[i, df.columns.get_loc("Breakdown_Level")] = breakdown_price

            # SHORT: Break below support + high volume + close near low
            if (not in_short and
                df["Close"].iloc[i] < df["Support"].iloc[i] and
                df["Volume"].iloc[i] > df["Vol_Avg"].iloc[i] * self.volume_mult and
                df["Close_Position"].iloc[i] < 0.3):
                df.iloc[i, df.columns.get_loc("Signal")] = Signal.SHORT.value

            # COVER: Reclaim support OR profit target hit OR deeply oversold
            elif in_short:
                profit_pct = (breakdown_price - df["Close"].iloc[i]) / breakdown_price
                if (df["Close"].iloc[i] > breakdown_price or
                    profit_pct >= self.profit_target or
                    df["RSI"].iloc[i] < 20):
                    df.iloc[i, df.columns.get_loc("Signal")] = Signal.COVER.value

        return df


class BearMeanReversionShort(Strategy):
    """
    Bear Mean Reversion Short — Short on overbought bounces in a downtrend.
    The "sell the rally" strategy.

    Logic:
    - SHORT: Price in downtrend (below SMA50) + RSI > 60 (overbought bounce)
             + Price touches upper Bollinger Band
    - COVER: RSI < 35 OR Price touches lower Bollinger Band

    Best for: Ranging bear markets with sharp relief rallies
    """
    def __init__(self, sma_period: int = 50, bb_period: int = 20,
                 bb_std: float = 2.0, rsi_period: int = 14,
                 rsi_short_above: int = 60, rsi_cover_below: int = 35):
        super().__init__(
            f"BearMeanRevShort_SMA{sma_period}_RSI{rsi_short_above}",
            "bear_short"
        )
        self.sma_period = sma_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_short_above = rsi_short_above
        self.rsi_cover_below = rsi_cover_below

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # SMA for trend
        df["SMA50"] = df["Close"].rolling(window=self.sma_period).mean()

        # Bollinger Bands
        df["BB_Mid"] = df["Close"].rolling(window=self.bb_period).mean()
        bb_std = df["Close"].rolling(window=self.bb_period).std()
        df["BB_Upper"] = df["BB_Mid"] + (self.bb_std * bb_std)
        df["BB_Lower"] = df["BB_Mid"] - (self.bb_std * bb_std)

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        df["Signal"] = Signal.HOLD.value

        # Stateful SHORT/COVER pairing (no overlap):
        #   SHORT only fires when flat; COVER only fires when already short.
        in_short = False
        for i in range(1, len(df)):
            if not in_short:
                # SHORT: Downtrend + overbought bounce + near upper BB
                if (df["Close"].iloc[i] < df["SMA50"].iloc[i] and
                        df["RSI"].iloc[i] > self.rsi_short_above and
                        df["Close"].iloc[i] >= df["BB_Upper"].iloc[i] * 0.98):
                    df.iloc[i, df.columns.get_loc("Signal")] = Signal.SHORT.value
                    in_short = True
            else:
                # COVER: Oversold OR near lower BB
                if (df["RSI"].iloc[i] < self.rsi_cover_below or
                        df["Close"].iloc[i] <= df["BB_Lower"].iloc[i] * 1.02):
                    df.iloc[i, df.columns.get_loc("Signal")] = Signal.COVER.value
                    in_short = False

        return df


# ============================================================
#  PHASE 1B: REGRESSION SUPERTREND STRATEGY (Strategy #14)
#  Place after BearMeanReversionShort, before STRATEGY REGISTRY
# ============================================================


class RegressionSuperTrendStrategy(Strategy):
    """
    Strategy #14: Regression SuperTrend

    A trend-following strategy using linear regression with RMSE-based
    adaptive bands. More sophisticated than SMA/EMA crossovers because
    band width adapts to how well price is following the regression line.

    How it works:
    1. Fit a linear regression line over the last N bars (default 20)
    2. Compute RMSE (Root Mean Squared Error) — measures how much price
       deviates from the regression line
    3. Create bands at regression_line ± (RMSE × multiplier)
    4. Apply Supertrend tightening logic:
       - In uptrend:  lower band can only move UP (never down)
       - In downtrend: upper band can only move DOWN (never up)
    5. Signal on band violation:
       - BUY:  close crosses above upper band (bearish → bullish flip)
       - SELL: close crosses below lower band (bullish → bearish flip)

    Why this is different from SMA/EMA strategies:
    - SMA/EMA: band width is fixed (e.g., 20-period average)
    - This: band width adapts to how cleanly price follows the trend.
      Wide bands = choppy price action = fewer signals.
      Narrow bands = clean trend = more responsive signals.

    Category: trend
    Regime weights: BULL=1.5x, BEAR=0.3x, SIDEWAYS=1.0x
    """

    def __init__(self, reg_length: int = 20, rmse_multiplier: float = 2.0):
        super().__init__(
            f"Regression_SuperTrend_{reg_length}_{rmse_multiplier}",
            "trend"
        )
        self.reg_length = reg_length
        self.rmse_multiplier = rmse_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate BUY/SELL/HOLD signals.

        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']

        Returns:
            DataFrame with additional columns:
                'Reg_Line', 'Reg_Upper', 'Reg_Lower', 'Reg_Trend', 'Signal'
        """
        df = df.copy()
        close = df['Close'].values.astype(float)
        n = len(close)
        length = self.reg_length
        mul = self.rmse_multiplier

        # Precompute x values for linear regression
        x = np.arange(length, dtype=float)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)

        # Initialize arrays
        reg_line = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        trend = np.ones(n, dtype=int)       # 1 = bullish, -1 = bearish
        signals = np.zeros(n, dtype=float)  # 1=BUY, -1=SELL, 0=HOLD

        for i in range(length - 1, n):
            # Extract window
            window = close[i - length + 1:i + 1]
            y_mean = window.mean()

            # Linear regression: slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
            if x_var == 0:
                continue
            slope = np.sum((x - x_mean) * (window - y_mean)) / x_var
            intercept = y_mean - slope * x_mean

            # Regression values for each point in window
            reg_values = slope * x + intercept

            # Current bar's regression value (last point in window)
            reg_line[i] = reg_values[-1]

            # RMSE — matches Pine Script formula: sqrt(Σerr² / (len-1))
            residuals = window - reg_values
            rmse = np.sqrt(np.sum(residuals ** 2) / (length - 1))

            # Raw bands
            raw_upper = reg_line[i] + rmse * mul
            raw_lower = reg_line[i] - rmse * mul

            if i > length - 1 and not np.isnan(upper[i - 1]):
                prev_trend = trend[i - 1]

                # Supertrend tightening logic
                # In downtrend: upper band can only tighten (move down)
                if prev_trend == -1:
                    upper[i] = min(raw_upper, upper[i - 1])
                else:
                    upper[i] = raw_upper

                # In uptrend: lower band can only tighten (move up)
                if prev_trend == 1:
                    lower[i] = max(raw_lower, lower[i - 1])
                else:
                    lower[i] = raw_lower

                # Trend flip detection
                if prev_trend == 1 and close[i] < lower[i]:
                    trend[i] = -1  # Bullish → Bearish
                elif prev_trend == -1 and close[i] > upper[i]:
                    trend[i] = 1   # Bearish → Bullish
                else:
                    trend[i] = prev_trend

                # Generate signal on trend flip
                if trend[i] != prev_trend:
                    signals[i] = Signal.BUY.value if trend[i] == 1 \
                                 else Signal.SELL.value
            else:
                # First valid bar — initialize
                upper[i] = raw_upper
                lower[i] = raw_lower
                trend[i] = 1

        # Add to DataFrame
        df['Reg_Line'] = reg_line
        df['Reg_Upper'] = upper
        df['Reg_Lower'] = lower
        df['Reg_Trend'] = trend
        df['Signal'] = signals

        return df

    # ============================================================
#  PHASE 2A: CHOCH / BOS WITH RETEST + CONFIRMATION
#  (Strategy #15)
# ============================================================


class CHOCHBOSRetest(Strategy):
    """
    Strategy #15: CHOCH/BOS with Retest + Confirmation

    Structural analysis strategy that identifies high-probability
    entries through three layers of confirmation.

    Key Concepts:
    ┌─────────────────────────────────────────────────────────┐
    │  BOS (Break of Structure):                              │
    │    Price breaks a swing level in the SAME direction     │
    │    as current structure. = Trend continuation signal.   │
    │                                                         │
    │  CHOCH (Change of Character):                           │
    │    Price breaks a swing level in the OPPOSITE direction │
    │    of current structure. = Trend reversal signal.       │
    └─────────────────────────────────────────────────────────┘

    Entry Sequence (3 confirmations):
    1. BREAK: Close through a confirmed swing high/low
       → Sets the neckline level (body of break candle)
    2. RETEST: Price returns to neckline level
       → Body must touch the neckline (wick doesn't count)
    3. CONFIRM: Price crosses neckline again in break direction
       → This is when the signal fires

    Why 3 confirmations?
    - Break alone: could be a fakeout
    - Break + Retest: shows price respected the level
    - Break + Retest + Confirm: shows genuine follow-through

    Filters:
    - ADX must be above minimum (trending market)
    - Max wait timeout after break (avoid stale setups)

    Category: structure
    Regime weights: BULL=1.0x, BEAR=1.0x, SIDEWAYS=1.0x
    """

    def __init__(self, pivot_left: int = 5, pivot_right: int = 5,
                 max_wait_bars: int = 40, adx_period: int = 14,
                 adx_min: float = 20.0, use_adx_filter: bool = True):
        super().__init__(
            f"CHOCH_BOS_Retest_P{pivot_left}_{pivot_right}",
            "structure"
        )
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.max_wait_bars = max_wait_bars
        self.adx_period = adx_period
        self.adx_min = adx_min
        self.use_adx_filter = use_adx_filter

    def _detect_pivots(self, high: np.ndarray, low: np.ndarray):
        """
        Detect pivot highs and lows.

        A pivot high at bar i means high[i] is the highest in the
        window [i - pivot_left, i + pivot_right].

        The pivot is CONFIRMED at bar (i + pivot_right) because we
        need future bars to validate it.

        Returns:
            swing_high: array where swing_high[j] = pivot price
                        if a pivot was confirmed at bar j, else NaN
            swing_low:  same for pivot lows
        """
        n = len(high)
        left = self.pivot_left
        right = self.pivot_right
        swing_high = np.full(n, np.nan)
        swing_low = np.full(n, np.nan)

        for i in range(left, n - right):
            # Check if high[i] is the max in its window
            window_highs = high[i - left:i + right + 1]
            if high[i] == window_highs.max() and np.sum(window_highs == high[i]) == 1:
                swing_high[i + right] = high[i]

            # Check if low[i] is the min in its window
            window_lows = low[i - left:i + right + 1]
            if low[i] == window_lows.min() and np.sum(window_lows == low[i]) == 1:
                swing_low[i + right] = low[i]

        return swing_high, swing_low

    def _compute_adx(self, high_series, low_series, close_series, period):
        """Compute ADX using Wilder's smoothing (matches TradingView)."""
        prev_close = close_series.shift(1)
        tr = pd.concat([
            high_series - low_series,
            (high_series - prev_close).abs(),
            (low_series - prev_close).abs()
        ], axis=1).max(axis=1)

        up_move = high_series - high_series.shift(1)
        down_move = low_series.shift(1) - low_series

        plus_dm = pd.Series(0.0, index=high_series.index)
        minus_dm = pd.Series(0.0, index=high_series.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        sm_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        sm_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        plus_di = 100 * sm_plus / atr.replace(0, np.nan)
        minus_di = 100 * sm_minus / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        return adx

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)

        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        open_ = df['Open'].values

        # Compute ADX
        adx = self._compute_adx(df['High'], df['Low'], df['Close'],
                                self.adx_period).values

        # Detect pivots
        swing_high, swing_low = self._detect_pivots(high, low)

        # State machine
        signals = np.zeros(n, dtype=float)
        break_type = np.full(n, '', dtype=object)  # 'BOS' or 'CHOCH'

        last_swing_h = np.nan
        last_swing_l = np.nan
        struct_dir = 0      # 1=bullish structure, -1=bearish, 0=undefined

        # Active setup tracking
        active = False
        active_dir = 0      # 1=looking for long, -1=looking for short
        neckline = np.nan
        retested = False
        setup_bar = 0

        warmup = self.pivot_left + self.pivot_right

        for i in range(warmup, n):
            # Update swing levels from confirmed pivots
            if not np.isnan(swing_high[i]):
                last_swing_h = swing_high[i]
            if not np.isnan(swing_low[i]):
                last_swing_l = swing_low[i]

            # Need at least one swing H and one swing L
            if np.isnan(last_swing_h) or np.isnan(last_swing_l):
                continue

            # ADX filter
            adx_ok = (not self.use_adx_filter) or (adx[i] >= self.adx_min)

            # Timeout check for active setup
            if active and (i - setup_bar) > self.max_wait_bars:
                active = False
                active_dir = 0
                neckline = np.nan
                retested = False

            # ── If we have an active setup, check retest + entry ──
            if active:
                if not retested:
                    # Retest: body touches neckline
                    if active_dir == 1:
                        # Long retest: candle body comes down to neckline
                        if min(open_[i], close[i]) <= neckline:
                            retested = True
                    else:
                        # Short retest: candle body comes up to neckline
                        if max(open_[i], close[i]) >= neckline:
                            retested = True
                else:
                    # Confirmation: price crosses neckline again
                    if active_dir == 1:
                        if close[i] > neckline and close[i - 1] <= neckline:
                            if adx_ok:
                                signals[i] = Signal.BUY.value
                                break_type[i] = "CHOCH_BOS"
                                struct_dir = 1
                            active = False
                            active_dir = 0
                            neckline = np.nan
                            retested = False
                    else:
                        if close[i] < neckline and close[i - 1] >= neckline:
                            if adx_ok:
                                signals[i] = Signal.SELL.value
                                break_type[i] = "CHOCH_BOS"
                                struct_dir = -1
                            active = False
                            active_dir = 0
                            neckline = np.nan
                            retested = False
                continue

            # ── Check for new breaks ──
            broke_up = (close[i] > last_swing_h and
                        close[i - 1] <= last_swing_h)
            broke_down = (close[i] < last_swing_l and
                          close[i - 1] >= last_swing_l)

            if broke_up:
                active = True
                active_dir = 1
                neckline = max(open_[i], close[i])  # Body level
                retested = False
                setup_bar = i

            elif broke_down:
                active = True
                active_dir = -1
                neckline = min(open_[i], close[i])  # Body level
                retested = False
                setup_bar = i

        df['Signal'] = signals
        df['Swing_High'] = swing_high
        df['Swing_Low'] = swing_low

        return df


# ============================================================
#  PHASE 2B: LIQUIDITY SWEEP + RETEST + BREAKOUT
#  (Strategy #16)
# ============================================================


class LiquiditySweepRetest(Strategy):
    """
    Strategy #16: Liquidity Sweep + Retest + Breakout

    An institutional-style strategy that detects when "smart money"
    sweeps liquidity at swing levels, then trades the reversal.

    Key Concept:
    ┌─────────────────────────────────────────────────────────┐
    │  LIQUIDITY POOL: Swing highs and lows where stop losses │
    │  cluster. Institutional players target these levels to  │
    │  fill large orders.                                     │
    │                                                         │
    │  SWEEP: Price WICKS beyond a swing level (triggers      │
    │  stops) but CLOSES back inside. The wick beyond is the  │
    │  "liquidity grab."                                      │
    │                                                         │
    │  RETEST: Price returns to the swept level. This is the  │
    │  "smart money" testing if the level holds as new        │
    │  support/resistance.                                    │
    │                                                         │
    │  BREAKOUT: Price breaks the retest candle's extreme in  │
    │  the sweep direction — confirming the reversal.         │
    └─────────────────────────────────────────────────────────┘

    Entry Sequence (3 stages):
    1. SWEEP: Bullish — low < swing_low AND close > swing_low
              Bearish — high > swing_high AND close < swing_high
    2. RETEST: Price returns to swept level within ATR tolerance
    3. BREAKOUT: Close breaks retest candle's high (long) or low (short)

    Why this works:
    - The sweep traps retail traders who placed stops at swing levels
    - The reversal after sweep shows institutional intent
    - The retest + breakout confirms the new direction

    Category: structure
    Regime weights: BULL=1.0x, BEAR=1.0x, SIDEWAYS=1.0x
    """

    def __init__(self, pivot_left: int = 5, pivot_right: int = 5,
                 retest_tol_atr_mult: float = 0.15,
                 max_wait_bars: int = 25, atr_length: int = 14):
        super().__init__(
            f"Liquidity_Sweep_P{pivot_left}_{pivot_right}",
            "structure"
        )
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.retest_tol_atr_mult = retest_tol_atr_mult
        self.max_wait_bars = max_wait_bars
        self.atr_length = atr_length

    def _detect_pivots(self, high: np.ndarray, low: np.ndarray):
        """Detect pivot highs and lows (same as CHOCHBOSRetest)."""
        n = len(high)
        left = self.pivot_left
        right = self.pivot_right
        swing_high = np.full(n, np.nan)
        swing_low = np.full(n, np.nan)

        for i in range(left, n - right):
            window_highs = high[i - left:i + right + 1]
            if high[i] == window_highs.max() and np.sum(window_highs == high[i]) == 1:
                swing_high[i + right] = high[i]

            window_lows = low[i - left:i + right + 1]
            if low[i] == window_lows.min() and np.sum(window_lows == low[i]) == 1:
                swing_low[i + right] = low[i]

        return swing_high, swing_low

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)

        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        open_ = df['Open'].values

        # ATR for retest tolerance
        atr = compute_atr(df['High'], df['Low'], df['Close'],
                          self.atr_length).values

        # Detect pivots
        swing_high, swing_low = self._detect_pivots(high, low)

        # State machine
        signals = np.zeros(n, dtype=float)

        last_swing_h = np.nan
        last_swing_l = np.nan

        # Active sweep tracking
        mode = 0            # 0=idle, 1=bull sweep active, -1=bear sweep active
        liq_level = np.nan  # The swept level
        sweep_bar = 0
        retested = False
        retest_hi = np.nan  # High of retest candle
        retest_lo = np.nan  # Low of retest candle

        warmup = self.pivot_left + self.pivot_right + self.atr_length

        for i in range(warmup, n):
            # Update swing levels
            if not np.isnan(swing_high[i]):
                last_swing_h = swing_high[i]
            if not np.isnan(swing_low[i]):
                last_swing_l = swing_low[i]

            tol = atr[i] * self.retest_tol_atr_mult if not np.isnan(atr[i]) else 0

            # Timeout check
            if mode != 0 and (i - sweep_bar) > self.max_wait_bars:
                mode = 0
                liq_level = np.nan
                retested = False
                retest_hi = np.nan
                retest_lo = np.nan

            # ── If sweep is active, check retest + breakout ──
            if mode != 0:
                if not retested:
                    if mode == 1:
                        # Bull retest: price dips to liq_level with rejection
                        if low[i] <= (liq_level + tol) and close[i] > liq_level:
                            retested = True
                            retest_hi = high[i]
                            retest_lo = low[i]
                    else:
                        # Bear retest: price rises to liq_level with rejection
                        if high[i] >= (liq_level - tol) and close[i] < liq_level:
                            retested = True
                            retest_hi = high[i]
                            retest_lo = low[i]
                else:
                    # Breakout confirmation
                    if mode == 1:
                        # Bull breakout: close above retest candle high
                        if close[i] > retest_hi:
                            signals[i] = Signal.BUY.value
                            mode = 0
                            liq_level = np.nan
                            retested = False
                    else:
                        # Bear breakout: close below retest candle low
                        if close[i] < retest_lo:
                            signals[i] = Signal.SELL.value
                            mode = 0
                            liq_level = np.nan
                            retested = False
                continue

            # ── Check for new sweeps ──
            if not np.isnan(last_swing_l):
                # Bullish sweep: wick below swing low, close back above
                if low[i] < last_swing_l and close[i] > last_swing_l:
                    mode = 1
                    liq_level = last_swing_l
                    sweep_bar = i
                    retested = False
                    retest_hi = np.nan
                    retest_lo = np.nan
                    continue

            if not np.isnan(last_swing_h):
                # Bearish sweep: wick above swing high, close back below
                if high[i] > last_swing_h and close[i] < last_swing_h:
                    mode = -1
                    liq_level = last_swing_h
                    sweep_bar = i
                    retested = False
                    retest_hi = np.nan
                    retest_lo = np.nan

        df['Signal'] = signals
        df['Swing_High'] = swing_high
        df['Swing_Low'] = swing_low

        return df

    # ============================================================
#  PHASE 3A: EXHAUSTION DETECTOR
#  Not a strategy — a consensus modifier.
#  Detects when a trend is running out of momentum.
# ============================================================


class ExhaustionDetector:
    """
    Detects trend exhaustion — when a trend is losing steam.

    This is NOT a strategy. It doesn't generate BUY/SELL signals.
    Instead, it produces an "exhaustion score" that the scanner
    uses to downgrade trend-aligned signals.

    How it works:
    ┌─────────────────────────────────────────────────────────┐
    │  UPTREND EXHAUSTION:                                    │
    │    1. Price is above EMA50 (uptrend confirmed)          │
    │    2. Distance from price to EMA50 is CONTRACTING       │
    │       (price drifting closer to the moving average)     │
    │    3. RSI is FALLING (momentum fading)                  │
    │    → Both conditions = trend is tiring                  │
    │                                                         │
    │  DOWNTREND EXHAUSTION:                                  │
    │    1. Price is below EMA50 (downtrend confirmed)        │
    │    2. Distance from price to EMA50 is CONTRACTING       │
    │    3. RSI is RISING (selling pressure fading)           │
    │    → Both conditions = downtrend is tiring              │
    └─────────────────────────────────────────────────────────┘

    The output is a per-bar exhaustion boolean that the scanner
    uses to reduce confidence in trend-aligned consensus signals.

    Usage:
        detector = ExhaustionDetector(config)
        result = detector.analyze(df)
        # result has 'exhausting', 'exhaustion_score', 'direction' per bar
    """

    def __init__(self, config=None):
        self.config = config or EXHAUSTION_CONFIG

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze a stock's data for trend exhaustion.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with new columns:
                'Exh_Trend_EMA':     The trend reference line
                'Exh_Distance':      |close - trend_ema|
                'Exh_Dist_EMA':      Smoothed distance
                'Exh_Dist_Falling':  True if dist_ema is contracting
                'Exh_RSI':           RSI values
                'Exh_RSI_Weakening': True if RSI moving against trend
                'Exhausting':        True if both conditions met
                'Exhaustion_Dir':    'BULL_EXH' or 'BEAR_EXH' or None
        """
        df = df.copy()
        c = self.config

        if not c.enabled:
            df['Exhausting'] = False
            df['Exhaustion_Dir'] = None
            return df

        close = df['Close']

        # ── Trend reference line ──
        trend_ema = close.ewm(span=c.trend_ema_length, adjust=False).mean()
        df['Exh_Trend_EMA'] = trend_ema

        # ── Distance from price to trend ──
        distance = (close - trend_ema).abs()
        dist_ema = distance.ewm(span=c.distance_ema_length, adjust=False).mean()
        df['Exh_Distance'] = distance
        df['Exh_Dist_EMA'] = dist_ema

        # Distance contraction: dist_ema falling for N consecutive bars
        dist_falling = pd.Series(True, index=df.index)
        for k in range(1, c.contraction_bars + 1):
            dist_falling = dist_falling & (dist_ema < dist_ema.shift(k))
        df['Exh_Dist_Falling'] = dist_falling

        # ── RSI ──
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=c.rsi_period).mean()
        avg_loss = loss.rolling(window=c.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df['Exh_RSI'] = rsi

        # ── Trend direction ──
        bull = close > trend_ema
        bear = close < trend_ema

        # RSI weakening:
        #   In uptrend: RSI falling for N bars = exhaustion
        #   In downtrend: RSI rising for N bars = exhaustion (selling pressure fading)
        rsi_weakening = pd.Series(False, index=df.index)

        # Uptrend exhaustion: RSI falling
        rsi_falling = pd.Series(True, index=df.index)
        for k in range(1, c.rsi_weakness_bars + 1):
            rsi_falling = rsi_falling & (rsi < rsi.shift(k))
        bull_exhaustion = bull & dist_falling & rsi_falling

        # Downtrend exhaustion: RSI rising
        rsi_rising = pd.Series(True, index=df.index)
        for k in range(1, c.rsi_weakness_bars + 1):
            rsi_rising = rsi_rising & (rsi > rsi.shift(k))
        bear_exhaustion = bear & dist_falling & rsi_rising

        # Combined
        exhausting = bull_exhaustion | bear_exhaustion
        df['Exh_RSI_Weakening'] = rsi_weakening
        df['Exhausting'] = exhausting

        # Direction label
        df['Exhaustion_Dir'] = None
        df.loc[bull_exhaustion, 'Exhaustion_Dir'] = 'BULL_EXH'
        df.loc[bear_exhaustion, 'Exhaustion_Dir'] = 'BEAR_EXH'

        return df

    def is_exhausting(self, df: pd.DataFrame) -> dict:
        """
        Quick check: is the LAST bar exhausting?

        Returns:
            {
                'exhausting': bool,
                'direction': str or None,
                'rsi': float,
                'trend_ema': float,
                'distance_contracting': bool,
                'rsi_weakening': bool,
            }
        """
        result = self.analyze(df)
        last = result.iloc[-1]

        return {
            'exhausting': bool(last.get('Exhausting', False)),
            'direction': last.get('Exhaustion_Dir'),
            'rsi': round(float(last.get('Exh_RSI', 0)), 1),
            'trend_ema': round(float(last.get('Exh_Trend_EMA', 0)), 2),
            'distance_contracting': bool(last.get('Exh_Dist_Falling', False)),
            'rsi_weakening': bool(last.get('Exhausting', False)),
        }


# ============================================================
#  PHASE 3B: TREND / RANGE CLASSIFIER
#  Not a strategy — a sub-regime detector.
#  Works alongside your existing Bull/Bear/Sideways regime.
# ============================================================


class TrendRangeClassifier:
    """
    Classifies market behavior as TREND or RANGE.

    This is a SECOND layer of regime detection that answers
    a different question than your existing regime_detector.py:

    - regime_detector: "Is the market Bull, Bear, or Sideways?"
      → Direction question

    - TrendRangeClassifier: "Is the market trending or ranging?"
      → Character question

    Classification is based on three independent factors:
    ┌─────────────────────────────────────────────────────────┐
    │  1. FLIP COUNT: Count EMA crossover flips in last N    │
    │     bars. More flips = more ranging (choppy).          │
    │     Threshold: >= 4 flips in 60 bars = RANGE           │
    │                                                         │
    │  2. ADX LEVEL: Measures trend strength (not direction). │
    │     ADX <= 20 = no directional strength = RANGE         │
    │     ADX > 20 = trending                                 │
    │                                                         │
    │  3. EMA SLOPE: How steep is the 50 EMA?                │
    │     |slope| < 0.05% per bar = flat = RANGE              │
    │     |slope| >= 0.05% = trending                         │
    │                                                         │
    │  DECISION: If at least 2 of 3 factors say RANGE → RANGE │
    │  Otherwise → TREND                                      │
    └─────────────────────────────────────────────────────────┘

    Usage:
        classifier = TrendRangeClassifier(config)
        result = classifier.classify(df)
        # result has 'sub_regime' = 'TREND' or 'RANGE'
    """

    def __init__(self, config=None):
        self.config = config or TREND_RANGE_CONFIG

    def _compute_adx(self, high, low, close, period):
        """Compute ADX using Wilder's smoothing."""
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        sm_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        sm_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        plus_di = 100 * sm_plus / atr.replace(0, np.nan)
        minus_di = 100 * sm_minus / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        return adx

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each bar as TREND or RANGE.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with new columns:
                'TR_Fast_EMA':      Fast EMA
                'TR_Slow_EMA':      Slow EMA
                'TR_Flip':          True on crossover bars
                'TR_Flip_Count':    Flips in lookback window
                'TR_ADX':           ADX values
                'TR_EMA_Slope':     Slope of slow EMA (% change per bar)
                'TR_Range_Score':   How many factors say RANGE (0-3)
                'Sub_Regime':       'TREND' or 'RANGE'
        """
        df = df.copy()
        c = self.config

        if not c.enabled:
            df['Sub_Regime'] = 'TREND'
            df['TR_Range_Score'] = 0
            return df

        close = df['Close']

        # ── Factor 1: Flip count ──
        fast_ema = close.ewm(span=c.fast_ema, adjust=False).mean()
        slow_ema = close.ewm(span=c.slow_ema, adjust=False).mean()

        # Detect crossovers (flip = cross event)
        cross_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        cross_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        flip = cross_up | cross_down

        # Count flips in lookback window (rolling sum)
        flip_count = flip.astype(int).rolling(window=c.flip_lookback, min_periods=1).sum()

        is_range_flips = flip_count >= c.flip_threshold

        df['TR_Fast_EMA'] = fast_ema
        df['TR_Slow_EMA'] = slow_ema
        df['TR_Flip'] = flip
        df['TR_Flip_Count'] = flip_count

        # ── Factor 2: ADX ──
        adx = self._compute_adx(df['High'], df['Low'], df['Close'], c.adx_period)
        is_range_adx = adx <= c.adx_range_max

        df['TR_ADX'] = adx

        # ── Factor 3: EMA slope ──
        slope_ema = close.ewm(span=c.slope_ema_length, adjust=False).mean()
        slope_abs = (slope_ema - slope_ema.shift(c.slope_lookback)).abs()
        slope_pct = (slope_abs / slope_ema.replace(0, np.nan)) * 100
        # Normalize to per-bar slope
        slope_pct_per_bar = slope_pct / c.slope_lookback
        is_range_slope = slope_pct_per_bar < c.slope_min_pct

        df['TR_EMA_Slope'] = slope_pct_per_bar

        # ── Combined classification ──
        # Score: how many factors say RANGE (0-3)
        range_score = (is_range_flips.astype(int) +
                       is_range_adx.astype(int) +
                       is_range_slope.astype(int))

        # If 2 or more factors say RANGE → RANGE, else TREND
        sub_regime = np.where(range_score >= 2, 'RANGE', 'TREND')

        df['TR_Range_Score'] = range_score
        df['Sub_Regime'] = sub_regime

        return df

    def get_sub_regime(self, df: pd.DataFrame) -> dict:
        """
        Quick check: what is the current sub-regime for the last bar?

        Returns:
            {
                'sub_regime': 'TREND' or 'RANGE',
                'range_score': int (0-3),
                'adx': float,
                'flip_count': int,
                'slope_pct': float,
                'factors': dict with individual factor results
            }
        """
        result = self.classify(df)
        last = result.iloc[-1]

        adx_val = float(last.get('TR_ADX', 0))
        flip_ct = int(last.get('TR_Flip_Count', 0))
        slope_val = float(last.get('TR_EMA_Slope', 0))
        score = int(last.get('TR_Range_Score', 0))

        c = self.config
        return {
            'sub_regime': str(last.get('Sub_Regime', 'TREND')),
            'range_score': score,
            'adx': round(adx_val, 1),
            'flip_count': flip_ct,
            'slope_pct': round(slope_val, 4),
            'factors': {
                'flips_range': flip_ct >= c.flip_threshold,
                'adx_range': adx_val <= c.adx_range_max,
                'slope_range': slope_val < c.slope_min_pct,
            }
        }


# ============================================================
#  PHASE 4A: CCI PREP STRATEGY (Strategy #17)
#  Two-stage mean reversion: ARM on extreme → FIRE on trigger
# ============================================================

# ============================================================
#  PHASE 4A: CCI PREP STRATEGY (Strategy #17)
#  Two-stage mean reversion: ARM on extreme → FIRE on trigger
# ============================================================


class CCIPrepStrategy(Strategy):
    """
    Strategy #17: CCI PREP (Preparation) — Two-Stage Mean Reversion

    A unique mean reversion approach that separates "preparation"
    from "entry" into two distinct stages. This reduces false
    signals compared to single-threshold CCI strategies.

    How it works:
    ┌─────────────────────────────────────────────────────────┐
    │  STAGE 1 — ARM (Preparation):                           │
    │    CCI crosses beyond ±extreme (default ±200).          │
    │    This doesn't generate a signal — it just "arms"      │
    │    the setup, meaning "an extreme has been reached,      │
    │    watch for the reversal."                              │
    │                                                         │
    │    Long arm:  CCI crosses below -200 (deeply oversold)  │
    │    Short arm: CCI crosses above +200 (deeply overbought)│
    │                                                         │
    │  STAGE 2 — FIRE (Entry):                                │
    │    CCI crosses back through ±trigger (default ±100).    │
    │    This fires the actual signal.                        │
    │                                                         │
    │    Long fire:  CCI crosses above -100 (recovery)        │
    │    Short fire: CCI crosses below +100 (rejection)       │
    │                                                         │
    │  COOLDOWN:                                              │
    │    After a signal fires, wait N bars before allowing    │
    │    another one. Prevents signal spam in choppy markets. │
    │                                                         │
    │  WHY TWO STAGES?                                        │
    │    Single-threshold CCI strategies fire on every cross  │
    │    of ±100, which generates many false signals.         │
    │    By first requiring an extreme (±200), then waiting   │
    │    for the trigger (±100), we only trade reversions     │
    │    from genuinely stretched conditions.                 │
    └─────────────────────────────────────────────────────────┘

    Category: mean_reversion
    Regime weights: BULL=0.5x, BEAR=1.2x, SIDEWAYS=1.0x
    """

    def __init__(self, cci_length: int = 20, extreme_level: int = 200,
                 trigger_level: int = 100, cooldown_bars: int = 12):
        super().__init__(
            f"CCI_Prep_{cci_length}_{extreme_level}_{trigger_level}",
            "mean_reversion"
        )
        self.cci_length = cci_length
        self.extreme_level = extreme_level
        self.trigger_level = trigger_level
        self.cooldown_bars = cooldown_bars

    def _compute_cci(self, high, low, close, period):
        """
        Compute CCI (Commodity Channel Index).

        CCI = (Typical Price - SMA(TP)) / (0.015 × Mean Deviation)
        Typical Price = (High + Low + Close) / 3
        """
        tp = (high + low + close) / 3.0
        sma_tp = tp.rolling(window=period).mean()

        # Mean deviation (not standard deviation)
        mean_dev = tp.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))),
            raw=True
        )

        cci = (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
        return cci

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate BUY/SELL/HOLD signals using two-stage CCI logic.

        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']

        Returns:
            DataFrame with additional columns:
                'CCI', 'CCI_Armed_Long', 'CCI_Armed_Short', 'Signal'
        """
        df = df.copy()

        # Compute CCI
        df['CCI'] = self._compute_cci(
            df['High'], df['Low'], df['Close'], self.cci_length
        )

        n = len(df)
        signals = np.zeros(n, dtype=float)

        # State tracking
        armed_long = False   # Long preparation armed
        armed_short = False  # Short preparation armed
        last_signal_bar = -self.cooldown_bars - 1  # Allow first signal

        cci_vals = df['CCI'].values
        armed_l = np.zeros(n, dtype=bool)
        armed_s = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if np.isnan(cci_vals[i]) or np.isnan(cci_vals[i - 1]):
                continue

            cci = cci_vals[i]
            prev_cci = cci_vals[i - 1]
            cooldown_ok = (i - last_signal_bar) >= self.cooldown_bars

            # ── Stage 1: ARM on extremes ──
            # Long arm: CCI drops below negative extreme
            if prev_cci >= -self.extreme_level and cci < -self.extreme_level:
                armed_long = True
                armed_short = False  # Only one direction at a time

            # Short arm: CCI rises above positive extreme
            elif prev_cci <= self.extreme_level and cci > self.extreme_level:
                armed_short = True
                armed_long = False

            # ── Stage 2: FIRE on trigger crossover ──
            # Long fire: armed + CCI crosses above negative trigger
            if armed_long and cooldown_ok:
                if prev_cci <= -self.trigger_level and cci > -self.trigger_level:
                    signals[i] = Signal.BUY.value
                    armed_long = False
                    last_signal_bar = i

            # Short fire: armed + CCI crosses below positive trigger
            if armed_short and cooldown_ok:
                if prev_cci >= self.trigger_level and cci < self.trigger_level:
                    signals[i] = Signal.SELL.value
                    armed_short = False
                    last_signal_bar = i

            armed_l[i] = armed_long
            armed_s[i] = armed_short

        df['Signal'] = signals
        df['CCI_Armed_Long'] = armed_l
        df['CCI_Armed_Short'] = armed_s

        return df


# ============================================================
#  PHASE 4B: ADX QUALITY GATE
#  Not a strategy — a universal signal quality modifier.
#  Suppresses signals when ADX is low, boosts when strong.
# ============================================================


class ADXQualityGate:
    """
    Universal ADX-based signal quality gate.

    Unlike the other modifiers (exhaustion, sub-regime) which are
    directional or regime-specific, the ADX gate is a pure STRENGTH
    filter that applies to ALL signals regardless of direction.

    Think of it as a "market energy" meter:
    - Low ADX = low energy = signals are likely noise
    - High ADX = high energy = signals have momentum behind them

    The gate assigns a quality zone and corresponding weight
    multiplier to each signal:

    ADX < 15  → NO_TRADE  (×0.3) — market is dead, skip
    15 ≤ 20   → WEAK      (×0.6) — some life, reduced confidence
    20 ≤ 25   → OK        (×1.0) — normal, baseline confidence
    25 ≤ 35   → CLEAN     (×1.2) — strong trend, high confidence
    ADX ≥ 35  → LATE      (×0.9) — very strong but may be exhausted

    ADX DIRECTION bonus:
    - ADX rising  + trend signal → extra boost (trend strengthening)
    - ADX falling + trend signal → extra penalty (trend weakening)
    - ADX falling + mean reversion → no penalty (actually favorable)

    Usage:
        gate = ADXQualityGate(config)
        result = gate.evaluate(df)
        # result has 'zone', 'weight_mult', 'adx', 'adx_rising'

        # Apply to signal weights:
        adjusted_weight = base_weight * result['weight_mult']
    """

    def __init__(self, config=None):

        self.config = config or ADX_GATE_CONFIG

    def _compute_adx(self, high, low, close, period):
        """Compute ADX using Wilder's smoothing."""
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        sm_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        sm_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        plus_di = 100 * sm_plus / atr.replace(0, np.nan)
        minus_di = 100 * sm_minus / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / \
             (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        return adx

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluate ADX quality for the last bar.

        Args:
            df: OHLCV DataFrame

        Returns:
            {
                'adx': float,
                'zone': str ('NO_TRADE', 'WEAK', 'OK', 'CLEAN', 'LATE'),
                'weight_mult': float,
                'adx_rising': bool,
                'adx_direction_mult': float (for trend signals),
                'plus_di': float,
                'minus_di': float,
            }
        """
        c = self.config
        adx = self._compute_adx(df['High'], df['Low'], df['Close'],
                                 c.adx_period)
        last_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0
        prev_adx = float(adx.iloc[-2]) if len(adx) > 1 and not np.isnan(adx.iloc[-2]) else last_adx

        adx_rising = last_adx > prev_adx

        # Determine zone
        if last_adx < c.no_trade_max:
            zone = "NO_TRADE"
            weight_mult = c.no_trade_mult
        elif last_adx < c.weak_max:
            zone = "WEAK"
            weight_mult = c.weak_mult
        elif last_adx < c.ok_max:
            zone = "OK"
            weight_mult = c.ok_mult
        elif last_adx < c.clean_max:
            zone = "CLEAN"
            weight_mult = c.clean_mult
        else:
            zone = "LATE"
            weight_mult = c.late_mult

        # ADX direction adjustment for trend signals
        if adx_rising:
            adx_dir_mult = c.adx_rising_boost
        else:
            adx_dir_mult = c.adx_falling_penalty

        return {
            'adx': round(last_adx, 1),
            'zone': zone,
            'weight_mult': weight_mult,
            'adx_rising': adx_rising,
            'adx_direction_mult': adx_dir_mult,
        }

    def apply_to_signals(self, stock_signals: dict,
                          adx_result: dict) -> dict:
        """
        Apply ADX quality gate to a stock's signal map.

        Modifies weights based on ADX zone. Also applies
        ADX direction bonus/penalty for trend-category signals.

        Args:
            stock_signals: {strategy_name: {signal, weight, category, ...}}
            adx_result:    output from self.evaluate()

        Returns:
            Modified stock_signals
        """
        zone = adx_result['zone']
        base_mult = adx_result['weight_mult']
        adx_dir_mult = adx_result['adx_direction_mult']

        for name, sig_info in stock_signals.items():
            category = sig_info['category']

            # Apply base zone multiplier
            sig_info['weight'] *= base_mult

            # Apply ADX direction adjustment for trend/momentum signals
            if category in ('trend', 'momentum'):
                sig_info['weight'] *= adx_dir_mult
                if adx_dir_mult > 1.0:
                    sig_info['adx_boost'] = True
                elif adx_dir_mult < 1.0:
                    sig_info['adx_penalty'] = True

            # For mean reversion in LATE zone: slight boost
            # (overextended trends often mean-revert)
            if zone == "LATE" and category == 'mean_reversion':
                sig_info['weight'] *= 1.1
                sig_info['adx_boost'] = True

            sig_info['adx_zone'] = zone

        return stock_signals

    def apply(self, stock_signals: dict, df: pd.DataFrame) -> dict:
        """
        Convenience wrapper: evaluate ADX on the DataFrame and apply
        the gate to stock_signals in one call.
        """
        adx_result = self.evaluate(df)
        return self.apply_to_signals(stock_signals, adx_result)

    def get_quality_level(self, df: pd.DataFrame) -> str:
        """
        Convenience wrapper: return the ADX quality zone string for
        the last bar ('NO_TRADE', 'WEAK', 'OK', 'CLEAN', 'LATE').
        """
        return self.evaluate(df)["zone"]

# ============================================================
#  STRATEGY REGISTRY
# ============================================================

def get_all_strategies() -> list:
    """All strategies — original + bear short-selling + Phase 1 additions"""
    return [
        # Original 9
        SMACrossover(20, 50),
        SMACrossover(50, 200),
        EMACrossover(12, 26),
        RSIStrategy(14, 30, 70),
        MACDStrategy(12, 26, 9),
        BollingerBands(20, 2.0),
        BreakoutStrategy(20, 1.5),
        MeanReversion(50, 2.0, 0.5),
        MomentumStrategy(20, 20),
        # Phase 1B: New strategy
        RegressionSuperTrendStrategy(20, 2.0),
        # Phase 2A: CHOCH/BOS Retest
        CHOCHBOSRetest(),
        # Phase 2B: Liquidity Sweep Retest
        LiquiditySweepRetest(),
        # Phase 4A: CCI PREP/READY
        CCIPrepStrategy(),
    ]


def get_bear_strategies() -> list:
    """NEW: Only bear-specific short-selling strategies"""
    return [
        BearTrendShort(),
        BearMomentumShort(),
        BearBreakdownShort(),
        BearMeanReversionShort(),
    ]


def get_all_strategies_with_shorts() -> list:
    """NEW: All strategies including bear shorts"""
    return get_all_strategies() + get_bear_strategies()


def get_regime_strategies(regime: str) -> list:
    """
    NEW: Get appropriate strategies for the current market regime.

    BULL:     Original 9 (long-biased)
    BEAR:     Original 9 + 4 bear short strategies
    SIDESWAYS: Original 9 only
    """
    if regime == "BEAR":
        return get_all_strategies_with_shorts()
    else:
        return get_all_strategies()


def get_strategy_by_name(name: str) -> Optional[Strategy]:
    for s in get_all_strategies_with_shorts():
        if s.name == name:
            return s
    print(f"Strategy '{name}' not found.")
    return None


def get_strategy_info() -> pd.DataFrame:
    """NEW: Print all available strategies with category info"""
    strategies = get_all_strategies_with_shorts()
    info = []
    for s in strategies:
        info.append({
            "Name": s.name,
            "Category": s.category,
            "Type": "bear_short" if "bear" in s.category else "original",
            "Signals": "SHORT/COVER" if "bear" in s.category else "BUY/SELL",
        })
    return pd.DataFrame(info)
