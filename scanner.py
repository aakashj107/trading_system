"""
Indian Market Trading System — Signal Scanner
v3.1 + Phase 1/2/3 — BQ Filter, Structural Strategies, Exhaustion, Sub-Regime.

This module:
  1. Scans all stocks × all strategies
  2. Applies Break Quality Filter (Phase 1A)
  3. Detects exhaustion (Phase 3A) and sub-regime (Phase 3B)
  4. Builds weighted consensus with regime-aware thresholds
"""

import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from strategies import (
    Signal,
    get_all_strategies,
    get_regime_strategies,
    BreakQualityFilter,
    ExhaustionDetector,
    TrendRangeClassifier,
    CCIPrepStrategy,        # Phase 4A — NEW
    ADXQualityGate,         # Phase 4B — NEW
)
from data_fetcher import YFinanceDataFetcher  # default fetcher for the scanner
from config import (
    BQ_CONFIG,
    EXHAUSTION_CONFIG,
    TREND_RANGE_CONFIG,
    CCI_PREP_CONFIG,        # Phase 4A — NEW
    ADX_GATE_CONFIG,        # Phase 4B — NEW
    STOCK_UNIVERSE,
    get_correlation_group,
)


# ============================================================
#  STRATEGY QUALITY WEIGHTS
#  Each strategy gets a quality weight and a category.
#  The category determines how regime and sub-regime affect it.
# ============================================================

STRATEGY_QUALITY = {
    # ── Original 9 ──────────────────────────────────────────
    "SMA_Crossover_20_50":                    {"weight": 0.4, "category": "trend"},
    "SMA_Crossover_50_200":                   {"weight": 0.3, "category": "trend"},
    "EMA_Crossover_12_26":                    {"weight": 0.7, "category": "trend"},
    "RSI_14_30_70":                           {"weight": 1.0, "category": "mean_reversion"},
    "MACD_12_26_9":                           {"weight": 1.3, "category": "momentum"},
    "Bollinger_20_2.0":                       {"weight": 1.5, "category": "mean_reversion"},
    "Breakout_20_Vol1.5":                     {"weight": 0.5, "category": "trend"},
    "MeanReversion_50_2.0std":                {"weight": 1.2, "category": "mean_reversion"},
    "Momentum_ROC20":                         {"weight": 0.4, "category": "momentum"},

    # ── Bear Short-Selling (4) ──────────────────────────────
    "BearTrendShort_EMA20_RSI45":             {"weight": 1.3, "category": "bear_short"},
    "BearMomentumShort_MACD_12_26_9":         {"weight": 1.2, "category": "bear_short"},
    "BearBreakdownShort_LB20_Vol1.5":         {"weight": 1.1, "category": "bear_short"},
    "BearMeanRevShort_SMA50_RSI60":           {"weight": 1.0, "category": "bear_short"},

    # ── Phase 1B: Regression SuperTrend ─────────────────────
    "Regression_SuperTrend_20_2.0":           {"weight": 1.2, "category": "trend"},

    # ── Phase 2: Structural Strategies ──────────────────────
    "CHOCH_BOS_Retest_P5_5":                  {"weight": 1.1, "category": "structure"},
    "Liquidity_Sweep_P5_5":                   {"weight": 1.0, "category": "structure"},

    # ── Pine Script Strategies (if loaded) ──────────────────
    "Structure_Reg20_Choch_ADX20":            {"weight": 1.1, "category": "structure"},
    "Combined_Structure_RTS":                 {"weight": 0.7, "category": "structure"},
    "RTS_Adaptive_ALMA_21_T2.0":             {"weight": 0.9, "category": "adaptive"},

    # ── Phase 4A: CCI PREP/READY  ──────────────────
    "CCI_Prep_20_200_100":                    {"weight": 1.0, "category": "mean_reversion"},
}


# ============================================================
#  REGIME-BASED CATEGORY WEIGHTS
#  Multipliers applied to each strategy category per regime.
# ============================================================

REGIME_CATEGORY_WEIGHTS = {
    "BULL": {
        "trend":          1.5,
        "momentum":       1.2,
        "mean_reversion": 0.5,
        "structure":      1.0,
        "adaptive":       1.0,
        "bear_short":     0.0,
    },
    "BEAR": {
        "trend":          0.3,
        "momentum":       0.3,
        "mean_reversion": 1.2,
        "structure":      1.0,
        "adaptive":       0.8,
        "bear_short":     1.3,
    },
    "SIDEWAYS": {
        "trend":          1.0,
        "momentum":       1.0,
        "mean_reversion": 1.0,
        "structure":      1.0,
        "adaptive":       1.0,
        "bear_short":     0.5,
    },
}


# ============================================================
#  REGIME-AWARE CONSENSUS THRESHOLDS
#  Number of strategies required for each consensus level.
# ============================================================

CONSENSUS_THRESHOLDS = {
    "BULL":     {"CONVICTION": 4, "STRONG": 3, "MODERATE": 2, "WEAK": 1},
    "BEAR":     {"CONVICTION": 3, "STRONG": 2, "MODERATE": 2, "WEAK": 1},
    "SIDEWAYS": {"CONVICTION": 4, "STRONG": 3, "MODERATE": 2, "WEAK": 1},
}


class SignalScanner:
    """
    Scans all stocks through all strategies, applies quality filters,
    detects exhaustion and sub-regime, and builds weighted consensus.

    Pipeline per stock:
        1. Fetch data
        2. Check exhaustion (Phase 3A)
        3. Classify sub-regime TREND/RANGE (Phase 3B)
        4. Run each strategy → get raw signal
        5. Apply Break Quality Filter (Phase 1A)
        6. Apply regime-based category weight
        7. Apply strategy quality weight
        8. Apply exhaustion penalty (Phase 3A)
        9. Apply sub-regime adjustment (Phase 3B)
        10. Build consensus from weighted signals
    """

    def __init__(
        self,
        fetcher=None,
        regime: str = "BULL",
        period: str = "2y",
        min_volume: float = 100000,
        bq_config=None,
        exhaustion_config=None,
        trend_range_config=None,
        adx_gate_config=None,          # Phase 4B — NEW
        symbols=None,
        strategies=None,
    ):
        """
        Args:
            fetcher: DataFetcher instance with get_historical_data(symbol, period).
                     Defaults to a fresh YFinanceDataFetcher if omitted.
            regime:  Current market regime — "BULL", "BEAR", or "SIDESWAYS"
            period:  Data period to fetch (e.g. "2y", "1y")
            min_volume: Minimum average 20-day volume to include a stock
            bq_config: Optional BreakQualityConfig override
            exhaustion_config: Optional ExhaustionConfig override
            trend_range_config: Optional TrendRangeConfig override
            symbols:  Optional explicit list of symbols to scan (overrides universe)
            strategies: Optional explicit strategy list (overrides regime-based set)
        """
        self.fetcher = fetcher if fetcher is not None else YFinanceDataFetcher()
        self.regime = regime
        self.period = period
        self.min_volume = min_volume

        # ── Strategies ──
        # Use caller-provided strategies when given, else regime-based set.
        self.strategies = list(strategies) if strategies is not None \
            else get_regime_strategies(regime)

        # ── Phase 1A: Break Quality Filter ──
        self.bq_filter = BreakQualityFilter(config=bq_config or BQ_CONFIG)

        # ── Phase 3A: Exhaustion Detector ──
        self.exhaustion_detector = ExhaustionDetector(
            config=exhaustion_config or EXHAUSTION_CONFIG
        )

        # ── Phase 3B: Trend/Range Classifier ──
        self.tr_classifier = TrendRangeClassifier(
            config=trend_range_config or TREND_RANGE_CONFIG
        )

        # ── Phase 4B: ADX Quality Gate ──
        self.adx_gate = ADXQualityGate(
            config=adx_gate_config or ADX_GATE_CONFIG
        )

        # ── Regime weights ──
        self.regime_weights = REGIME_CATEGORY_WEIGHTS.get(
            regime, REGIME_CATEGORY_WEIGHTS["SIDEWAYS"]
        )

        # ── Consensus thresholds ──
        self.thresholds = CONSENSUS_THRESHOLDS.get(
            regime, CONSENSUS_THRESHOLDS["SIDEWAYS"]
        )

        # ── Symbols ──
        # Honor an explicit symbol list when provided.
        if symbols:
            self.symbols = list(symbols)
        else:
            self.symbols = list(STOCK_UNIVERSE.keys())

    # ==========================================================
    #  PHASE 3A: EXHAUSTION PENALTY
    # ==========================================================

    def _apply_exhaustion_penalty(
        self, stock_signals: dict, exhaustion_result: dict
    ) -> dict:
        """
        Downgrade trend-aligned signals when exhaustion is detected.

        When a trend is tiring:
        - Trend/momentum signals that ALIGN with the current trend
          get their weight reduced by the exhaustion penalty factor.
        - Counter-trend signals are NOT penalized (they might be
          early reversal signals worth listening to).

        Args:
            stock_signals:     {strategy_name: {signal, weight, category, ...}}
            exhaustion_result: output from ExhaustionDetector.is_exhausting()

        Returns:
            Modified stock_signals with adjusted weights
        """
        if not exhaustion_result.get("exhausting", False):
            return stock_signals

        direction = exhaustion_result.get("direction")
        penalty = EXHAUSTION_CONFIG.weight_penalty  # default 0.70

        for name, sig_info in stock_signals.items():
            signal = sig_info["signal"]
            category = sig_info["category"]

            if direction == "BULL_EXH":
                # Bullish exhaustion — penalize BUY signals from trend/momentum
                if signal == 1 and category in ("trend", "momentum"):
                    sig_info["weight"] *= penalty
                    sig_info["exhaustion_penalty"] = True

            elif direction == "BEAR_EXH":
                # Bearish exhaustion — penalize SELL/SHORT from trend/bear
                if signal in (-1, -2) and category in (
                    "trend", "momentum", "bear_short"
                ):
                    sig_info["weight"] *= penalty
                    sig_info["exhaustion_penalty"] = True

        return stock_signals

    # ==========================================================
    #  PHASE 3B: SUB-REGIME WEIGHT ADJUSTMENT
    # ==========================================================

    def _apply_sub_regime_weights(
        self, stock_signals: dict, sub_regime: str
    ) -> dict:
        """
        Adjust strategy weights based on TREND/RANGE classification.

        TREND mode:
          - trend/momentum strategies get boosted  (×1.3)
          - mean_reversion strategies get penalized (×0.7)

        RANGE mode:
          - mean_reversion strategies get boosted  (×1.3)
          - trend/momentum strategies get penalized (×0.7)

        structure, bear_short, adaptive: unaffected (work in both).

        Args:
            stock_signals: {strategy_name: {signal, weight, category, ...}}
            sub_regime:    'TREND' or 'RANGE'

        Returns:
            Modified stock_signals with adjusted weights
        """
        c = TREND_RANGE_CONFIG

        for name, sig_info in stock_signals.items():
            category = sig_info["category"]

            if sub_regime == "TREND":
                if category in ("trend", "momentum"):
                    sig_info["weight"] *= c.trend_boost  # default 1.3
                    sig_info["sub_regime_boost"] = True
                elif category == "mean_reversion":
                    sig_info["weight"] *= c.mr_penalty  # default 0.7
                    sig_info["sub_regime_penalty"] = True

            elif sub_regime == "RANGE":
                if category in ("trend", "momentum"):
                    sig_info["weight"] *= c.trend_penalty  # default 0.7
                    sig_info["sub_regime_penalty"] = True
                elif category == "mean_reversion":
                    sig_info["weight"] *= c.mr_boost  # default 1.3
                    sig_info["sub_regime_boost"] = True

            # structure, bear_short, adaptive: no adjustment

        return stock_signals

    # ==========================================================
    #  CONSENSUS BUILDER
    # ==========================================================

    def _build_consensus(
        self, stock_signals: dict
    ) -> dict:
        """
        Build consensus from weighted signals.

        Counts BUY and SELL/SHORT signals, sums their adjusted weights,
        and determines the consensus level and direction.

        Args:
            stock_signals: {strategy_name: {signal, weight, category, ...}}

        Returns:
            {
                'buy_count': int,
                'sell_count': int,
                'buy_score': float,
                'sell_score': float,
                'consensus_level': str,
                'direction': str,
            }
        """
        buy_count = 0
        sell_count = 0
        buy_score = 0.0
        sell_score = 0.0

        for name, sig_info in stock_signals.items():
            signal = sig_info["signal"]
            weight = sig_info["weight"]

            if signal == 1:       # BUY
                buy_count += 1
                buy_score += weight
            elif signal in (-1, -2):  # SELL or SHORT
                sell_count += 1
                sell_score += weight

        dominant_count = max(buy_count, sell_count)

        # Determine consensus level
        if dominant_count >= self.thresholds["CONVICTION"]:
            consensus_level = "CONVICTION"
        elif dominant_count >= self.thresholds["STRONG"]:
            consensus_level = "STRONG"
        elif dominant_count >= self.thresholds["MODERATE"]:
            consensus_level = "MODERATE"
        elif dominant_count >= self.thresholds["WEAK"]:
            consensus_level = "WEAK"
        else:
            consensus_level = "NONE"

        # Determine direction by weighted score
        if buy_score > sell_score:
            direction = "BUY"
        elif sell_score > buy_score:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        return {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_score": round(buy_score, 3),
            "sell_score": round(sell_score, 3),
            "consensus_level": consensus_level,
            "direction": direction,
        }

    # ==========================================================
    #  MAIN SCAN — SINGLE STOCK
    # ==========================================================

    def scan_stock(self, symbol: str, data: pd.DataFrame) -> Optional[dict]:
        """
        Scan a single stock through all strategies with all filters.

        Pipeline:
            1. Exhaustion check (Phase 3A)
            2. Sub-regime classification (Phase 3B)
            3. ADX quality level (Phase 4B)
            4. Per-strategy: signal → BQ filter → regime weight → quality weight
            5. Exhaustion penalty (Phase 3A)
            6. Sub-regime adjustment (Phase 3B)
            7. ADX quality gate (Phase 4B)
            8. Build consensus
        """
        if data.empty or len(data) < 50:
            return None

        if data["Volume"].tail(20).mean() < self.min_volume:
            return None

        # ── Phase 3A: Exhaustion check ──
        exhaustion = self.exhaustion_detector.is_exhausting(data)

        # ── Phase 3B: Sub-regime classification ──
        sub_regime_info = self.tr_classifier.get_sub_regime(data)
        sub_regime = sub_regime_info["sub_regime"]

        # ── Phase 4B: ADX quality level ──
        adx_quality = self.adx_gate.get_quality_level(data)

        # ── Per-strategy signal map ──
        stock_signals = {}

        for strategy in self.strategies:
            try:
                df = strategy.generate_signals(data)
                if df.empty:
                    continue

                last_signal = int(df["Signal"].iloc[-1])
                if last_signal == Signal.HOLD.value:
                    continue

                signal_map = {
                    Signal.BUY.value:    "BUY",
                    Signal.SELL.value:   "SELL",
                    Signal.SHORT.value:  "SHORT",
                    Signal.COVER.value:  "COVER",
                }
                signal_text = signal_map.get(last_signal, "HOLD")

                if signal_text == "COVER":
                    continue

                # ── Phase 1A: Break Quality Filter ──
                bq_result = self.bq_filter.check_bar(
                    data, bar_index=len(data) - 1
                )
                if not bq_result["passed"]:
                    continue

                # Get base quality weight and category
                sq = STRATEGY_QUALITY.get(
                    strategy.name,
                    {"weight": 1.0, "category": "unknown"},
                )
                weight = sq["weight"]
                category = sq["category"]

                # Apply regime-based category multiplier
                regime_mult = self.regime_weights.get(category, 1.0)
                weight *= regime_mult

                # SOFT penalty for weak candles
                if bq_result["weak"]:
                    weight *= 0.5

                stock_signals[strategy.name] = {
                    "signal": last_signal,
                    "signal_text": signal_text,
                    "weight": weight,
                    "category": category,
                    "bq_quality": bq_result["reason"],
                    "bq_weak": bq_result["weak"],
                    "exhaustion_penalty": False,
                    "sub_regime_boost": False,
                    "sub_regime_penalty": False,
                    "adx_adjusted": False,
                    "adx_suppressed": False,
                }

            except Exception:
                continue

        if not stock_signals:
            return None

        # ── Phase 4B: ADX Quality Gate ──
        adx_result = self.adx_gate.evaluate(data)
        if ADX_GATE_CONFIG.enabled:
            stock_signals = self.adx_gate.apply_to_signals(
                stock_signals, adx_result
            )

        # ── Phase 3A: Apply exhaustion penalty ──
        if exhaustion["exhausting"]:
            stock_signals = self._apply_exhaustion_penalty(
                stock_signals, exhaustion
            )

        # ── Phase 3B: Apply sub-regime weights ──
        if TREND_RANGE_CONFIG.enabled:
            stock_signals = self._apply_sub_regime_weights(
                stock_signals, sub_regime
            )

        # ── Build consensus ──
        consensus = self._build_consensus(stock_signals)

        return {
            "symbol": symbol,
            "stock_signals": stock_signals,
            "exhaustion": exhaustion,
            "sub_regime": sub_regime,
            "sub_regime_info": sub_regime_info,
            "adx_quality": adx_quality,
            "consensus": consensus,
        }

    # ==========================================================
    #  MAIN SCAN — ALL STOCKS
    # ==========================================================

    def scan(self, data_cache: Optional[Dict[str, pd.DataFrame]] = None,
              max_workers: int = 8) -> pd.DataFrame:
        """
        Scan all stocks in the universe through all strategies.

        Args:
            data_cache: optional dict {symbol: DataFrame} of pre-loaded data.
                        If provided, stocks in the cache skip network fetch.
            max_workers: number of threads for parallel per-stock scanning.

        Returns:
            DataFrame with one row per (stock, strategy) pair that
            produced a signal. Columns include signal type, weight,
            BQ quality, sub-regime, exhaustion status, and consensus.
        """
        results = []
        symbols = self.symbols

        def _fetch(symbol: str):
            if data_cache is not None and symbol in data_cache:
                return data_cache[symbol]
            try:
                return self.fetcher.get_historical_data(symbol, period=self.period)
            except Exception as e:
                print(f"  ⚠ Data error for {symbol}: {e}")
                return None

        # Parallel fetch (network-bound; CPU work happens in scan_stock)
        fetched = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_sym = {ex.submit(_fetch, s): s for s in symbols}
            for fut in as_completed(future_to_sym):
                s = future_to_sym[fut]
                try:
                    fetched[s] = fut.result()
                except Exception as e:
                    print(f"  ⚠ Fetch failed for {s}: {e}")
                    fetched[s] = None

        # Parallel per-stock scan
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_sym = {
                ex.submit(self.scan_stock, s, fetched[s]): s
                for s in symbols if fetched.get(s) is not None
            }
            for fut in as_completed(future_to_sym):
                s = future_to_sym[fut]
                try:
                    stock_result = fut.result()
                except Exception as e:
                    print(f"  ⚠ Scan error for {s}: {e}")
                    continue
                if stock_result is None:
                    continue

                consensus = stock_result["consensus"]
                exhaustion = stock_result["exhaustion"]
                sub_regime = stock_result["sub_regime"]
                data = fetched[s]

                for name, sig_info in stock_result["stock_signals"].items():
                    results.append({
                        # ── Core ──
                        "Date":        str(data.index[-1].date()),
                        "Symbol":      s,
                        "Strategy":    name,
                        "Signal":      sig_info["signal_text"],
                        "Price":       round(data.iloc[-1]["Close"], 2),
                        "Weight":      round(sig_info["weight"], 3),
                        "Category":    sig_info["category"],

                        # ── Phase 1A: Break Quality ──
                        "BQ_Quality":  sig_info["bq_quality"],
                        "BQ_Weak":     sig_info["bq_weak"],

                        # ── Phase 3A: Exhaustion ──
                        "Exhausting":    exhaustion["exhausting"],
                        "Exhaustion_Dir": exhaustion.get("direction"),
                        "Exh_Penalty":   sig_info["exhaustion_penalty"],

                        # ── Phase 3B: Sub-Regime ──
                        "Sub_Regime":  sub_regime,
                        "SR_Boost":    sig_info["sub_regime_boost"],
                        "SR_Penalty":  sig_info["sub_regime_penalty"],

                        # ── Phase 4B: ADX Gate ──
                        "ADX_Zone":    sig_info.get("adx_zone", ""),
                        "ADX_Boost":   sig_info.get("adx_boost", False),
                        "ADX_Penalty": sig_info.get("adx_penalty", False),

                        # ── Consensus (same for all rows of this stock) ──
                        "Consensus":     consensus["consensus_level"],
                        "Consensus_Dir": consensus["direction"],
                        "Buy_Count":     consensus["buy_count"],
                        "Sell_Count":    consensus["sell_count"],
                        "Buy_Score":     consensus["buy_score"],
                        "Sell_Score":    consensus["sell_score"],
                    })

        return pd.DataFrame(results) if results else pd.DataFrame()

    # ==========================================================
    #  CONVENIENCE: SCAN AND GET STOCK-LEVEL SUMMARY
    # ==========================================================

    def scan_with_summary(self) -> tuple:
        """
        Run the full scan and return both the signal-level DataFrame
        and a stock-level summary DataFrame.

        Returns:
            (signals_df, summary_df)
            signals_df: one row per (stock, strategy) signal
            summary_df: one row per stock with consensus and key metrics
        """
        signals_df = self.scan()

        if signals_df.empty:
            return signals_df, pd.DataFrame()

        # Build stock-level summary
        summary_records = []
        for symbol in signals_df["Symbol"].unique():
            stock_rows = signals_df[signals_df["Symbol"] == symbol]
            first = stock_rows.iloc[0]

            summary_records.append({
                "Symbol":        symbol,
                "Price":         first["Price"],
                "Direction":     first["Consensus_Dir"],
                "Consensus":     first["Consensus"],
                "Buy_Count":     first["Buy_Count"],
                "Sell_Count":    first["Sell_Count"],
                "Buy_Score":     first["Buy_Score"],
                "Sell_Score":    first["Sell_Score"],
                "Sub_Regime":    first["Sub_Regime"],
                "Exhausting":    first["Exhausting"],
                "Signals":       len(stock_rows),
                "BQ_Weak_Total": stock_rows["BQ_Weak"].sum(),
                "Exh_Penalties": stock_rows["Exh_Penalty"].sum(),
                "SR_Boosts":     stock_rows["SR_Boost"].sum(),
                "SR_Penalties":  stock_rows["SR_Penalty"].sum(),
            })

        summary_df = pd.DataFrame(summary_records)

        # Sort by consensus priority then by buy score
        consensus_order = {
            "CONVICTION": 0, "STRONG": 1, "MODERATE": 2, "WEAK": 3, "NONE": 4
        }
        summary_df["_sort"] = summary_df["Consensus"].map(consensus_order)
        summary_df = summary_df.sort_values(
            ["_sort", "Buy_Score"], ascending=[True, False]
        ).drop(columns=["_sort"]).reset_index(drop=True)

        return signals_df, summary_df

    # ==========================================================
    #  CONSENSUS API (used by main.py / daily_runner.py / run_all.py)
    # ==========================================================

    def scan_for_consensus(self) -> pd.DataFrame:
        """
        Run the full scan and return a STOCK-LEVEL consensus DataFrame
        (one row per stock that produced signals), sorted by consensus
        strength. Mirrors scan_with_summary()'s summary portion.

        Returns:
            DataFrame with columns: Symbol, Price, Direction, Consensus,
            Buy_Count, Sell_Count, Buy_Score, Sell_Score, Sub_Regime,
            Exhausting, Signals, BQ_Weak_Total, Exh_Penalties, SR_Boosts,
            SR_Penalties  — or empty DataFrame if no signals.
        """
        _signals_df, summary_df = self.scan_with_summary()
        return summary_df

    def resolve_conflicts(self, consensus_df: pd.DataFrame) -> pd.DataFrame:
        """
        De-conflict overlapping BUY/SELL signals per stock.

        When a stock shows both BUY and SELL/SHORT consensus, we keep the
        direction with the higher weighted score. The DataFrame is returned
        unchanged if it has no per-stock conflict (or is empty).
        """
        if consensus_df is None or consensus_df.empty:
            return consensus_df

        resolved = consensus_df.copy()
        if "Direction" in resolved.columns and "Buy_Score" in resolved.columns \
                and "Sell_Score" in resolved.columns:
            # Direction already reflects dominant weighted score, so nothing
            # further is required here — but we flag conflicts for transparency.
            resolved["Conflict"] = (
                (resolved["Buy_Score"] > 0) & (resolved["Sell_Score"] > 0)
            )
        return resolved

    def print_consensus(self, consensus_df: pd.DataFrame):
        """Print the stock-level consensus table."""
        self.print_summary(consensus_df)

    def print_signals(self, df: pd.DataFrame):
        if df.empty:
            print("No signals today.")
            return

        bq_mode = self.bq_filter.config.mode

        print(f"\n{'=' * 95}")
        print(f"  DAILY SIGNAL SCAN — {datetime.now().strftime('%Y-%m-%d')}")
        print(f"  Regime: {self.regime}  |  BQ: {bq_mode}  |  "
              f"Strategies: {len(self.strategies)}")
        print(f"{'=' * 95}")

        for sig in ["BUY", "SELL", "SHORT"]:
            sub = df[df["Signal"] == sig]
            if sub.empty:
                continue

            print(f"\n  {sig} SIGNALS ({len(sub)}):")
            print(f"  {'Symbol':<16} {'Strategy':<30} {'Price':>10} "
                  f"{'Weight':>7} {'Sub':>6} {'ADX':>6}  Notes")
            print(f"  {'─' * 90}")

            for _, r in sub.iterrows():
                notes = []
                if r.get("BQ_Weak"):
                    notes.append("BQ:WEAK")
                if r.get("Exhausting"):
                    exh_dir = r.get("Exhaustion_Dir", "") or ""
                    notes.append(f"EXH:{exh_dir}")
                if r.get("Exh_Penalty"):
                    notes.append("↓exh")
                if r.get("SR_Boost"):
                    notes.append(f"↑{r.get('Sub_Regime', '')}")
                if r.get("SR_Penalty"):
                    notes.append(f"↓{r.get('Sub_Regime', '')}")
                if r.get("ADX_Adjusted"):
                    adx_lvl = r.get("ADX_Level", "")
                    notes.append(f"ADX:{adx_lvl}")
                if r.get("ADX_Suppressed"):
                    notes.append("ADX:OFF")
                if r.get("ADX_Zone") and r["ADX_Zone"] not in ("OK", ""):
                    notes.append(f"ADX:{r['ADX_Zone']}")
                if r.get("ADX_Boost"):
                    notes.append("↑adx")
                if r.get("ADX_Penalty"):
                    notes.append("↓adx")

                note_str = ", ".join(notes) if notes else ""

                print(
                    f"  {r['Symbol']:<16} {r['Strategy']:<30} "
                    f"{'₹{:,.2f}'.format(r['Price']):>10} "
                    f"{r['Weight']:>7.2f} "
                    f"{r.get('Sub_Regime', ''):>6} "
                    f"{r.get('ADX', 0):>5.0f}  "
                    f"{note_str}"
                )

        # ── Consensus Summary ──
        if "Consensus" in df.columns:
            print(f"\n{'─' * 95}")
            print(f"  CONSENSUS SUMMARY")
            print(f"{'─' * 95}")

            consensus_data = df.groupby("Symbol").first().reset_index()

            for level in ["CONVICTION", "STRONG", "MODERATE", "WEAK"]:
                stocks = consensus_data[consensus_data["Consensus"] == level]
                if stocks.empty:
                    continue
                print(f"\n  {level} ({len(stocks)} stocks):")
                for _, r in stocks.iterrows():
                    exh_warn = " ⚠EXH" if r.get("Exhausting") else ""
                    adx_note = f" ADX:{r.get('ADX', 0):.0f}({r.get('ADX_Level', '')})"
                    print(
                        f"    {r['Symbol']:<16} {r['Consensus_Dir']:<8} "
                        f"B:{int(r['Buy_Count'])} S:{int(r['Sell_Count'])} "
                        f"score:{r['Buy_Score']:.1f}/{r['Sell_Score']:.1f} "
                        f"sub:{r['Sub_Regime']}{exh_warn}{adx_note}"
                    )

        # ── Phase 3+4 Warnings ──
        print(f"\n{'─' * 95}")

        if "Exhausting" in df.columns:
            exh_stocks = df[df["Exhausting"] == True]["Symbol"].unique()
            if len(exh_stocks) > 0:
                print(f"  ⚠ EXHAUSTION: {', '.join(exh_stocks)}")

        if "Sub_Regime" in df.columns:
            range_stocks = df[df["Sub_Regime"] == "RANGE"]["Symbol"].unique()
            trend_stocks = df[df["Sub_Regime"] == "TREND"]["Symbol"].unique()
            print(f"  Sub-Regime: {len(trend_stocks)} TREND, "
                  f"{len(range_stocks)} RANGE")

        if "ADX_Level" in df.columns:
            suppressed = df[df["ADX_Suppressed"] == True]["Symbol"].unique()
            if len(suppressed) > 0:
                print(f"  ⚠ ADX NO_TRADE (suppressed): {', '.join(suppressed)}")

            caution = df[df["ADX_Level"] == "CAUTION"]["Symbol"].unique()
            if len(caution) > 0:
                print(f"  ⚠ ADX CAUTION (overextended): {', '.join(caution)}")

        if "ADX_Zone" in df.columns:
            adx_summary = df[df["ADX_Zone"] != ""]["ADX_Zone"].value_counts()
            if not adx_summary.empty:
                adx_str = ", ".join(
                    f"{zone}:{count}" for zone, count in adx_summary.items()
                )
                print(f"  ADX Zones: {adx_str}")

        # Adjustment summary
        if "Exh_Penalty" in df.columns:
            bq_weak = int(df["BQ_Weak"].sum()) if "BQ_Weak" in df.columns else 0
            exh_pen = int(df["Exh_Penalty"].sum())
            sr_boost = int(df["SR_Boost"].sum())
            sr_pen = int(df["SR_Penalty"].sum())
            adx_adj = int(df["ADX_Adjusted"].sum()) if "ADX_Adjusted" in df.columns else 0
            adx_sup = int(df["ADX_Suppressed"].sum()) if "ADX_Suppressed" in df.columns else 0
            print(
                f"  Adjustments: "
                f"{bq_weak} BQ weak, "
                f"{exh_pen} exh penalty, "
                f"{sr_boost}↑/{sr_pen}↓ SR, "
                f"{adx_adj} ADX adj, "
                f"{adx_sup} ADX suppress"
            )

        print(f"\n{'=' * 95}")

    # ==========================================================
    #  PRINT STOCK SUMMARY
    # ==========================================================

    def print_summary(self, summary_df: pd.DataFrame):
        """
        Print stock-level consensus summary table.

        Args:
            summary_df: DataFrame from scan_with_summary()
        """
        if summary_df.empty:
            print("No stocks with active signals.")
            return

        print(f"\n{'=' * 95}")
        print(f"  STOCK CONSENSUS SUMMARY — {datetime.now().strftime('%Y-%m-%d')}")
        print(f"  Regime: {self.regime}")
        print(f"{'=' * 95}")

        print(
            f"\n  {'Symbol':<16} {'Price':>10} {'Direction':<8} "
            f"{'Consensus':<12} {'B/S':>5} {'Scores':>12} "
            f"{'Sub':>6} {'Exh':>5} {'Adj':>6}"
        )
        print(f"  {'─' * 90}")

        for _, r in summary_df.iterrows():
            exh_flag = "YES" if r["Exhausting"] else ""
            adj_count = (
                int(r["BQ_Weak_Total"])
                + int(r["Exh_Penalties"])
                + int(r["SR_Penalties"])
            )
            adj_str = str(adj_count) if adj_count > 0 else ""

            print(
                f"  {r['Symbol']:<16} "
                f"{'₹{:,.2f}'.format(r['Price']):>10} "
                f"{r['Direction']:<8} "
                f"{r['Consensus']:<12} "
                f"{int(r['Buy_Count'])}/{int(r['Sell_Count']):<4} "
                f"{r['Buy_Score']:.1f}/{r['Sell_Score']:.1f}  "
                f"{r['Sub_Regime']:>6} "
                f"{exh_flag:>5} "
                f"{adj_str:>6}"
            )

        # Statistics
        total = len(summary_df)
        by_consensus = summary_df["Consensus"].value_counts()
        by_dir = summary_df["Direction"].value_counts()

        print(f"\n  {'─' * 90}")
        print(f"  Total stocks with signals: {total}")
        print(f"  Direction: {dict(by_dir)}")
        print(f"  Consensus: {dict(by_consensus)}")

        by_sub = summary_df["Sub_Regime"].value_counts()
        exh_count = summary_df["Exhausting"].sum()
        print(f"  Sub-regime: {dict(by_sub)}")
        print(f"  Exhausting: {int(exh_count)} stocks")
        print(f"{'=' * 95}")


# ============================================================
#  STANDALONE RUNNER — for quick testing
# ============================================================

if __name__ == "__main__":
    """
    Quick standalone test.
    Requires: data_fetcher.py with a working DataFetcher class.

    Usage:
        python scanner.py
    """
    from config import print_sector_distribution

    print_sector_distribution()

    try:
        from data_fetcher import DataFetcher
        fetcher = DataFetcher()

        # Detect regime
        from regime_detector import RegimeDetector
        detector = RegimeDetector()
        regime = detector.detect_regime()
        print(f"\n  Detected regime: {regime}")

        # Run scanner
        scanner = SignalScanner(fetcher=fetcher, regime=regime)
        signals_df, summary_df = scanner.scan_with_summary()

        # Print results
        scanner.print_signals(signals_df)
        scanner.print_summary(summary_df)

    except ImportError as e:
        print(f"\n  Cannot run standalone: {e}")
        print(f"  Import SignalScanner in your daily_runner.py instead.")
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()