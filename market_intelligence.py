"""
Indian Market Trading System — Market Intelligence Layer
=========================================================
Option C: Full system intelligence upgrade.

Components:
  1. RegimeAdaptiveManager   — weights strategies by market regime
  2. IndiaVIXRiskGate        — dynamic position sizing by fear level
  3. ConsensusFilter         — requires N strategies to agree
  4. MarketBreadth           — % of stocks above key MAs
  5. WatchlistManager        — smart watchlist with buy zones
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"
REGIME_RANGE = "RANGE"


# ═══════════════════════════════════════════════════════════════════════
#  1. REGIME-ADAPTIVE STRATEGY MANAGER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StrategyWeight:
    name: str
    category: str          # "trend", "momentum", "mean_reversion", "structure", "adaptive"
    base_weight: float = 1.0
    regime_weight: float = 1.0
    effective_weight: float = 1.0


class RegimeAdaptiveManager:
    """
    Adjusts strategy weights based on market regime.

    TREND strategies (SMA crossover, EMA crossover, breakout) perform
    well in BULL markets but lose money in BEAR/RANGE.

    MOMENTUM strategies (ROC, MACD) work in BULL but dangerous in BEAR.

    MEAN REVERSION strategies (Bollinger, RSI) work in BEAR/RANGE
    but underperform in strong BULL trends.

    STRUCTURE strategies (CHOCH/BOS, RTS) adapt to whatever regime
    they detect per stock, so they get moderate weight in all regimes.
    """

    # Strategy → category mapping
    STRATEGY_CATEGORIES = {
        "SMA_Crossover_20_50": "trend",
        "SMA_Crossover_50_200": "trend",
        "EMA_Crossover_12_26": "trend",
        "Breakout_20_Vol1.5": "trend",
        "RSI_14_30_70": "mean_reversion",
        "Bollinger_20_2.0": "mean_reversion",
        "MeanReversion_50_2.0std": "mean_reversion",
        "MACD_12_26_9": "momentum",
        "Momentum_ROC20": "momentum",
        # Pine strategies
        "RTS_Adaptive_ALMA_21_T2.0": "adaptive",
        "RTS_Adaptive_HMA_21_T2.0": "adaptive",
        "Structure_Reg20_Choch_ADX20": "structure",
        "Combined_Structure_RTS": "structure",
    }

    # Regime → category → weight multiplier
    REGIME_WEIGHTS = {
        REGIME_BULL: {
            "trend": 2.0,
            "momentum": 1.5,
            "mean_reversion": 0.5,
            "structure": 1.0,
            "adaptive": 1.2,
        },
        REGIME_BEAR: {
            "trend": 0.3,
            "momentum": 0.3,
            "mean_reversion": 1.2,
            "structure": 1.0,
            "adaptive": 0.8,
        },
        REGIME_RANGE: {
            "trend": 0.2,
            "momentum": 0.5,
            "mean_reversion": 1.5,
            "structure": 1.2,
            "adaptive": 1.0,
        },
    }

    def get_category(self, strategy_name: str) -> str:
        for key, cat in self.STRATEGY_CATEGORIES.items():
            if key in strategy_name:
                return cat
        return "trend"  # default

    def get_weights(self, regime: str, strategies: list) -> List[StrategyWeight]:
        """Return weighted list of strategies for the given regime."""
        regime = regime.upper()
        if regime not in self.REGIME_WEIGHTS:
            regime = REGIME_BEAR  # default to conservative

        weights = self.REGIME_WEIGHTS[regime]
        result = []

        for strat in strategies:
            name = strat.name if hasattr(strat, "name") else str(strat)
            cat = self.get_category(name)
            rw = weights.get(cat, 1.0)

            result.append(StrategyWeight(
                name=name,
                category=cat,
                base_weight=1.0,
                regime_weight=rw,
                effective_weight=rw,
            ))

        return result

    def filter_strategies(self, regime: str, strategies: list,
                          min_weight: float = 0.3) -> list:
        """Return only strategies with weight >= min_weight for this regime."""
        weights = self.get_weights(regime, strategies)
        allowed_names = {w.name for w in weights if w.effective_weight >= min_weight}
        return [s for s in strategies if s.name in allowed_names]

    def print_weights(self, regime: str, strategies: list):
        """Print strategy weight table."""
        weights = self.get_weights(regime, strategies)
        weights.sort(key=lambda w: w.effective_weight, reverse=True)

        print(f"\n  STRATEGY WEIGHTS ({regime} regime):")
        print(f"  {'Strategy':<35} {'Category':<15} {'Weight':<8} {'Bar'}")
        print(f"  {'─' * 70}")

        for w in weights:
            bar_len = int(w.effective_weight * 10)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            skip = "SKIP" if w.effective_weight < 0.3 else ""
            print(f"  {w.name:<35} {w.category:<15} {w.effective_weight:<8.1f} "
                  f"{bar} {skip}")


# ═══════════════════════════════════════════════════════════════════════
#  2. INDIA VIX RISK GATE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RiskProfile:
    vix_level: float
    vix_regime: str       # "LOW", "NORMAL", "ELEVATED", "HIGH", "EXTREME"
    position_pct: float   # 0.04 to 0.15
    max_positions: int    # 1 to 5
    risk_label: str       # "Aggressive", "Normal", "Cautious", "Defensive", "Cash"
    message: str


class IndiaVIXRiskGate:
    """
    Dynamic position sizing based on India VIX (fear index).

    VIX measures implied volatility of Nifty options.
    High VIX = market is fearful = reduce risk.
    Low VIX = market is complacent = normal risk.
    """

    VIX_LEVELS = [
        (13, "LOW",      0.15, 5, "Aggressive",  "Low fear — full position sizing"),
        (17, "NORMAL",   0.12, 4, "Normal",      "Normal volatility — standard sizing"),
        (22, "ELEVATED", 0.10, 3, "Cautious",    "Elevated fear — reduced positions"),
        (28, "HIGH",     0.07, 2, "Defensive",   "High fear — minimum positions"),
        (99, "EXTREME",  0.04, 1, "Cash Pres.",  "Extreme fear — cash preservation"),
    ]

    def get_risk_profile(self, vix_value: float) -> RiskProfile:
        for threshold, regime, pct, max_pos, label, msg in self.VIX_LEVELS:
            if vix_value < threshold:
                return RiskProfile(
                    vix_level=vix_value,
                    vix_regime=regime,
                    position_pct=pct,
                    max_positions=max_pos,
                    risk_label=label,
                    message=msg,
                )
        # Fallback for extreme VIX
        return RiskProfile(
            vix_level=vix_value,
            vix_regime="EXTREME",
            position_pct=0.04,
            max_positions=1,
            risk_label="Cash Preservation",
            message="Extreme fear — preserve capital",
        )

    def print_risk_profile(self, vix_value: float):
        rp = self.get_risk_profile(vix_value)
        print(f"\n  INDIA VIX RISK GATE:")
        print(f"  {'─' * 50}")
        print(f"  VIX Level:    {rp.vix_level:.1f}")
        print(f"  Regime:       {rp.vix_regime}")
        print(f"  Risk Mode:    {rp.risk_label}")
        print(f"  Position:     {rp.position_pct * 100:.0f}% per trade")
        print(f"  Max Positions:{rp.max_positions}")
        print(f"  Message:      {rp.message}")

        # Show all levels for reference
        print(f"\n  VIX THRESHOLDS:")
        for thresh, regime, pct, max_p, label, _ in self.VIX_LEVELS:
            marker = " ◄" if vix_value < thresh and not any(
                vix_value < t for t, *_ in self.VIX_LEVELS if t < thresh
            ) else ""
            print(f"    < {thresh:<5} → {label:<15} "
                  f"Position: {pct*100:.0f}%  Max: {max_p}{marker}")

        return rp


# ═══════════════════════════════════════════════════════════════════════
#  3. MARKET BREADTH
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BreadthResult:
    pct_above_20sma: float
    pct_above_50sma: float
    breadth_regime: str    # "STRONG", "MODERATE", "WEAK", "DIRE"
    total_stocks: int
    above_20: int
    above_50: int
    message: str


class MarketBreadth:
    """
    Measures how many stocks are participating in the trend.

    Breadth > 60% → Strong participation, trend is real
    Breadth 40-60% → Moderate, selective entries
    Breadth 20-40% → Weak, defensive mode
    Breadth < 20% → Dire, very few stocks holding up the market
    """

    def calculate(self, stock_data: Dict[str, pd.DataFrame]) -> BreadthResult:
        """
        Calculate breadth from dict of {symbol: DataFrame}.
        Each DataFrame must have 'Close' column.
        """
        above_20 = 0
        above_50 = 0
        total = 0

        for symbol, df in stock_data.items():
            if df.empty or len(df) < 50:
                continue

            total += 1
            close = df["Close"]
            current = close.iloc[-1]

            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]

            if current > sma20:
                above_20 += 1
            if current > sma50:
                above_50 += 1

        if total == 0:
            return BreadthResult(0, 0, "UNKNOWN", 0, 0, 0, "No data")

        pct20 = above_20 / total * 100
        pct50 = above_50 / total * 100

        if pct20 >= 60:
            regime, msg = "STRONG", "Broad participation — trend is real"
        elif pct20 >= 40:
            regime, msg = "MODERATE", "Mixed participation — selective entries"
        elif pct20 >= 20:
            regime, msg = "WEAK", "Narrow participation — defensive mode"
        else:
            regime, msg = "DIRE", "Minimal participation — capital preservation"

        return BreadthResult(
            pct_above_20sma=pct20,
            pct_above_50sma=pct50,
            breadth_regime=regime,
            total_stocks=total,
            above_20=above_20,
            above_50=above_50,
            message=msg,
        )

    def print_breadth(self, result: BreadthResult):
        print(f"\n  MARKET BREADTH:")
        print(f"  {'─' * 50}")
        print(f"  Stocks analyzed:   {result.total_stocks}")
        print(f"  Above 20 SMA:      {result.above_20}/{result.total_stocks} "
              f"({result.pct_above_20sma:.1f}%)")
        print(f"  Above 50 SMA:      {result.above_50}/{result.total_stocks} "
              f"({result.pct_above_50sma:.1f}%)")
        print(f"  Breadth Regime:    {result.breadth_regime}")
        print(f"  Message:           {result.message}")

        # Visual bar
        bar_len = int(result.pct_above_20sma / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  [{bar}] {result.pct_above_20sma:.0f}%")

        return result


# ═══════════════════════════════════════════════════════════════════════
#  4. CONSENSUS SIGNAL FILTER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ConsensusSignal:
    symbol: str
    signal_type: str       # "BUY" or "SELL"
    agreeing_strategies: List[str]
    count: int
    strength: str          # "WEAK", "MODERATE", "STRONG", "CONVICTION"
    avg_weight: float      # average effective weight of agreeing strategies


class ConsensusFilter:
    """
    Requires multiple strategies to agree before generating a signal.

    Signal Strength:
      1 strategy  → WEAK (monitor only)
      2 strategies → MODERATE (paper trade candidate)
      3 strategies → STRONG (real trade candidate)
      5+ strategies → CONVICTION (highest priority)
    """

    STRENGTH_THRESHOLDS = [
        (5, "CONVICTION"),
        (3, "STRONG"),
        (2, "MODERATE"),
        (1, "WEAK"),
    ]

    def find_consensus(self, signals_df: pd.DataFrame,
                       strategy_weights: List[StrategyWeight] = None) -> List[ConsensusSignal]:
        """
        Find consensus signals across strategies.

        signals_df must have columns: Symbol, Signal, Strategy
        Signal values: BUY (1), SELL (-1), HOLD (0)
        """
        if signals_df.empty:
            return []

        # Create weight lookup
        weight_lookup = {}
        if strategy_weights:
            for sw in strategy_weights:
                weight_lookup[sw.name] = sw.effective_weight

        consensus = []

        # Group by symbol
        for symbol in signals_df["Symbol"].unique():
            sym_signals = signals_df[signals_df["Symbol"] == symbol]

            # Find BUY consensus
            buy_strats = sym_signals[sym_signals["Signal"] == "BUY"]["Strategy"].tolist()
            if len(buy_strats) >= 1:
                strength = self._get_strength(len(buy_strats))
                avg_w = np.mean([weight_lookup.get(s, 1.0) for s in buy_strats])
                consensus.append(ConsensusSignal(
                    symbol=symbol,
                    signal_type="BUY",
                    agreeing_strategies=buy_strats,
                    count=len(buy_strats),
                    strength=strength,
                    avg_weight=avg_w,
                ))

            # Find SELL consensus
            sell_strats = sym_signals[sym_signals["Signal"] == "SELL"]["Strategy"].tolist()
            if len(sell_strats) >= 1:
                strength = self._get_strength(len(sell_strats))
                avg_w = np.mean([weight_lookup.get(s, 1.0) for s in sell_strats])
                consensus.append(ConsensusSignal(
                    symbol=symbol,
                    signal_type="SELL",
                    agreeing_strategies=sell_strats,
                    count=len(sell_strats),
                    strength=strength,
                    avg_weight=avg_w,
                ))

        # Sort by count (strongest consensus first)
        consensus.sort(key=lambda c: (c.count, c.avg_weight), reverse=True)
        return consensus

    def _get_strength(self, count: int) -> str:
        for threshold, label in self.STRENGTH_THRESHOLDS:
            if count >= threshold:
                return label
        return "WEAK"

    def print_consensus(self, consensus: List[ConsensusSignal]):
        if not consensus:
            print(f"\n  CONSENSUS SIGNALS: None")
            return

        print(f"\n  CONSENSUS SIGNALS:")
        print(f"  {'─' * 80}")
        print(f"  {'Signal':<6} {'Symbol':<16} {'Strength':<12} {'Count':<6} "
              f"{'Avg Weight':<12} {'Strategies'}")
        print(f"  {'─' * 80}")

        for c in consensus:
            icon = "🟢" if c.signal_type == "BUY" else "🔴"
            strats = ", ".join(c.agreeing_strategies[:3])
            if len(c.agreeing_strategies) > 3:
                strats += f" +{len(c.agreeing_strategies) - 3} more"
            print(f"  {icon} {c.signal_type:<4} {c.symbol:<16} {c.strength:<12} "
                  f"{c.count:<6} {c.avg_weight:<12.2f} {strats}")

        # Action summary
        strong_buy = [c for c in consensus if c.signal_type == "BUY"
                      and c.strength in ("STRONG", "CONVICTION")]
        strong_sell = [c for c in consensus if c.signal_type == "SELL"
                       and c.strength in ("STRONG", "CONVICTION")]

        if strong_buy:
            print(f"\n  🟢 STRONG BUY CANDIDATES:")
            for c in strong_buy:
                print(f"    {c.symbol}: {c.count} strategies agree "
                      f"({c.strength})")

        if strong_sell:
            print(f"\n  🔴 STRONG SELL CANDIDATES:")
            for c in strong_sell:
                print(f"    {c.symbol}: {c.count} strategies agree "
                      f"({c.strength})")

        if not strong_buy and not strong_sell:
            print(f"\n  ⚠️ No STRONG/CONVICTION signals — waiting for alignment")


# ═══════════════════════════════════════════════════════════════════════
#  5. WATCHLIST MANAGER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WatchlistItem:
    symbol: str
    current_price: float
    buy_zone: float       # price level to start watching
    interesting_price: float  # fundamental attractive price
    pct_to_buy_zone: float
    conviction: float     # from equity research
    action: str           # from equity research
    status: str           # "APPROACHING", "IN_ZONE", "PASSED", "MONITORING"
    notes: str = ""


class WatchlistManager:
    """
    Manages an intelligent watchlist of stocks approaching buy zones.

    Stocks are categorized:
      APPROACHING — within 15% of buy zone
      IN_ZONE     — within buy zone (ready to consider)
      PASSED       — already passed buy zone (was it a miss or trap?)
      MONITORING  — more than 15% away from buy zone
    """

    def __init__(self):
        self.watchlist: List[WatchlistItem] = []
        self.removed: List[dict] = []  # track removed stocks

    def build_from_verdicts(self, verdicts: list,
                            data_quality: Dict[str, bool] = None):
        """Build watchlist from equity research verdicts."""
        self.watchlist = []

        for v in verdicts:
            if not hasattr(v, "symbol"):
                continue

            current = getattr(v, "current_price", 0)
            interesting = getattr(v, "interesting_price", 0)
            conviction = getattr(v, "conviction_score", 0)
            action = getattr(v, "action", None)
            action_val = action.value if hasattr(action, "value") else str(action)

            # Skip stocks with no data
            if current <= 0:
                if data_quality is not None:
                    data_quality[v.symbol] = False
                continue

            # Buy zone = interesting price (where fundamentals are attractive)
            buy_zone = interesting if interesting > 0 else current * 0.85
            pct_to_zone = ((current - buy_zone) / current * 100) if current > 0 else 0

            # Status
            if pct_to_zone <= 0:
                status = "IN_ZONE"
            elif pct_to_zone <= 15:
                status = "APPROACHING"
            elif current > buy_zone * 1.2:
                status = "MONITORING"
            else:
                status = "PASSED"

            # Notes
            notes = ""
            if action_val in ("STRONG_AVOID", "AVOID"):
                notes = f"AVOID — {action_val}"
            elif status == "IN_ZONE":
                notes = "In buy zone — consider entry with consensus"
            elif status == "APPROACHING":
                notes = f"Approaching — {pct_to_zone:.1f}% away"

            self.watchlist.append(WatchlistItem(
                symbol=v.symbol,
                current_price=current,
                buy_zone=buy_zone,
                interesting_price=interesting,
                pct_to_buy_zone=pct_to_zone,
                conviction=conviction,
                action=action_val,
                status=status,
                notes=notes,
            ))

        # Sort by proximity to buy zone
        self.watchlist.sort(key=lambda w: w.pct_to_buy_zone)

    def get_near_buy(self, threshold_pct: float = 15.0) -> List[WatchlistItem]:
        """Get stocks within threshold% of buy zone."""
        return [w for w in self.watchlist
                if w.pct_to_buy_zone <= threshold_pct
                and w.action not in ("STRONG_AVOID", "AVOID")]

    def get_in_zone(self) -> List[WatchlistItem]:
        """Get stocks currently in their buy zone."""
        return [w for w in self.watchlist
                if w.status == "IN_ZONE"
                and w.action not in ("STRONG_AVOID", "AVOID")]

    def print_watchlist(self):
        if not self.watchlist:
            print(f"\n  WATCHLIST: Empty")
            return

        print(f"\n  INTELLIGENT WATCHLIST:")
        print(f"  {'─' * 90}")
        print(f"  {'#':<3} {'Symbol':<16} {'Price':<10} {'Buy Zone':<10} "
              f"{'Gap%':<8} {'Conv':<6} {'Action':<14} {'Status':<12} {'Notes'}")
        print(f"  {'─' * 90}")

        for i, w in enumerate(self.watchlist, 1):
            # Status icons
            status_icon = {
                "IN_ZONE": "🟢",
                "APPROACHING": "🟡",
                "PASSED": "⚪",
                "MONITORING": "🔵",
            }.get(w.status, "⚪")

            gap_str = f"{w.pct_to_buy_zone:+.1f}%" if w.pct_to_buy_zone > 0 else "AT ZONE"
            if w.pct_to_buy_zone <= 0:
                gap_str = "IN ZONE"

            print(f"  {i:<3} {w.symbol:<16} ₹{w.current_price:<9,.0f} "
                  f"₹{w.buy_zone:<9,.0f} {gap_str:<8} {w.conviction:<6.1f} "
                  f"{w.action:<14} {status_icon} {w.status:<10} {w.notes}")

        # Summary
        in_zone = self.get_in_zone()
        near_buy = self.get_near_buy()

        print(f"\n  SUMMARY:")
        print(f"    Total watched:     {len(self.watchlist)}")
        print(f"    In buy zone:       {len(in_zone)}")
        print(f"    Approaching (<15%):{len(near_buy)}")

        if in_zone:
            print(f"\n  🟢 IN BUY ZONE (consider with consensus):")
            for w in in_zone:
                print(f"    {w.symbol}: ₹{w.current_price:,.0f} "
                      f"(conviction: {w.conviction}/10)")

        if near_buy and not in_zone:
            closest = near_buy[0]
            print(f"\n  🟡 CLOSEST TO BUY ZONE:")
            print(f"    {closest.symbol}: ₹{closest.current_price:,.0f} → "
                  f"₹{closest.buy_zone:,.0f} ({closest.pct_to_buy_zone:.1f}% away)")


# ═══════════════════════════════════════════════════════════════════════
#  6. COMBINED INTELLIGENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════

class MarketIntelligence:
    """
    Combines all intelligence components into a single decision engine.
    """

    def __init__(self):
        self.regime_manager = RegimeAdaptiveManager()
        self.vix_gate = IndiaVIXRiskGate()
        self.breadth = MarketBreadth()
        self.consensus = ConsensusFilter()
        self.watchlist = WatchlistManager()

    def analyze(self, regime: str, vix_value: float,
                stock_data: Dict[str, pd.DataFrame],
                signals_df: pd.DataFrame,
                verdicts: list,
                strategies: list) -> dict:
        """
        Run full intelligence analysis.
        Returns dict with all results + recommendations.
        """
        results = {}

        # 1. Strategy weights
        weights = self.regime_manager.get_weights(regime, strategies)
        results["strategy_weights"] = weights
        self.regime_manager.print_weights(regime, strategies)

        # 2. VIX risk gate
        risk_profile = self.vix_gate.get_risk_profile(vix_value)
        results["risk_profile"] = risk_profile
        self.vix_gate.print_risk_profile(vix_value)

        # 3. Market breadth
        breadth_result = self.breadth.calculate(stock_data)
        results["breadth"] = breadth_result
        self.breadth.print_breadth(breadth_result)

        # 4. Consensus signals
        consensus_signals = self.consensus.find_consensus(signals_df, weights)
        results["consensus"] = consensus_signals
        self.consensus.print_consensus(consensus_signals)

        # 5. Watchlist
        self.watchlist.build_from_verdicts(verdicts)
        results["watchlist"] = self.watchlist
        self.watchlist.print_watchlist()

        # 6. Final recommendation
        recommendation = self._generate_recommendation(
            regime, risk_profile, breadth_result, consensus_signals
        )
        results["recommendation"] = recommendation
        self._print_recommendation(recommendation)

        return results

    def _generate_recommendation(self, regime, risk_profile,
                                  breadth, consensus) -> dict:
        """Generate actionable recommendation from all inputs."""
        rec = {
            "action": "WAIT",
            "position_size": risk_profile.position_pct,
            "max_positions": risk_profile.max_positions,
            "reasons": [],
        }

        # Count strong signals
        strong_buys = [c for c in consensus
                       if c.signal_type == "BUY"
                       and c.strength in ("STRONG", "CONVICTION")]
        strong_sells = [c for c in consensus
                        if c.signal_type == "SELL"
                        and c.strength in ("STRONG", "CONVICTION")]

        # Decision logic
        if regime == REGIME_BEAR and breadth.breadth_regime in ("WEAK", "DIRE"):
            if strong_buys:
                rec["action"] = "SELECTIVE BUY"
                rec["reasons"].append(
                    f"Bear market but {len(strong_buys)} strong BUY consensus"
                )
                rec["reasons"].append("Use minimum position size")
                rec["position_size"] = min(risk_profile.position_pct, 0.07)
            else:
                rec["action"] = "WAIT / DEFENSIVE"
                rec["reasons"].append(f"Bear market + weak breadth "
                                      f"({breadth.pct_above_20sma:.0f}%)")
                rec["reasons"].append("No strong BUY consensus")
                rec["reasons"].append("Wait for regime change")

        elif regime == REGIME_BULL and breadth.breadth_regime in ("STRONG", "MODERATE"):
            if strong_buys:
                rec["action"] = "BUY"
                rec["reasons"].append(f"Bull market + strong breadth")
                rec["reasons"].append(f"{len(strong_buys)} strong BUY consensus")
            else:
                rec["action"] = "HOLD / MONITOR"
                rec["reasons"].append("Bull market but no strong consensus")
                rec["reasons"].append("Wait for clearer signals")

        elif regime == REGIME_RANGE:
            rec["action"] = "SELECTIVE / MEAN REVERSION"
            rec["reasons"].append("Range-bound market")
            rec["reasons"].append("Use mean reversion at range edges")
            rec["position_size"] = min(risk_profile.position_pct, 0.10)

        # Override: strong sells always reduce
        if strong_sells:
            rec["reasons"].append(
                f"⚠️ {len(strong_sells)} STRONG SELL signals — reduce exposure"
            )

        # VIX override
        if risk_profile.vix_regime in ("HIGH", "EXTREME"):
            rec["reasons"].append(
                f"VIX {risk_profile.vix_level:.1f} ({risk_profile.vix_regime}) "
                f"— minimum exposure"
            )
            rec["position_size"] = min(rec["position_size"], 0.07)

        return rec

    def _print_recommendation(self, rec: dict):
        """Print the final recommendation."""
        action = rec["action"]

        if "BUY" in action:
            icon = "🟢"
        elif "SELL" in action or "DEFENSIVE" in action or "WAIT" in action:
            icon = "🔴"
        else:
            icon = "🟡"

        print(f"\n  ╔{'═' * 58}╗")
        print(f"  ║{'INTELLIGENT RECOMMENDATION':^58}║")
        print(f"  ╠{'═' * 58}╣")
        print(f"  ║  {icon} ACTION: {action:<46}║")
        print(f"  ║  Position Size: {rec['position_size']*100:.0f}% "
              f"per trade{'':<32}║")
        print(f"  ║  Max Positions: {rec['max_positions']}"
              f"{'':<39}║")
        print(f"  ╠{'═' * 58}╣")
        for reason in rec["reasons"]:
            # Wrap long reasons
            while len(reason) > 54:
                print(f"  ║  {reason[:54]:<56}║")
                reason = "    " + reason[54:]
            print(f"  ║  {reason:<56}║")
        print(f"  ╚{'═' * 58}╝")


# ═══════════════════════════════════════════════════════════════════════
#  HELPER: Fetch India VIX
# ═══════════════════════════════════════════════════════════════════════

def fetch_india_vix(fetcher) -> float:
    """Fetch current India VIX. Returns fallback if unavailable."""
    try:
        vix_data = fetcher.get_historical_data("^INDIAVIX", period="5d")
        if not vix_data.empty:
            return float(vix_data["Close"].iloc[-1])
    except Exception:
        pass

    # Fallback: estimate from Nifty ATR
    try:
        import warnings
        warnings.filterwarnings("ignore")
        nifty = fetcher.get_nifty_index(period="1m")
        if not nifty.empty:
            atr_pct = (nifty["High"] - nifty["Low"]).tail(20).mean() / \
                       nifty["Close"].tail(20).mean() * 100
            # Rough approximation: daily ATR% × 16 ≈ annualized VIX
            return atr_pct * 16
    except Exception:
        pass

    return 18.0  # conservative fallback