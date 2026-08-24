"""
Indian Market Trading System — Daily Runner v3.1
Full pipeline execution: Regime → VIX → Research → Strategies →
Signals → Intelligence → Paper Trading → Charts → Summary.

Now supports all stocks in the balanced sector universe.
"""

import schedule
import time
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from scanner import SignalScanner
from strategies import (
    get_all_strategies, get_regime_strategies,
    get_all_strategies_with_shorts, get_strategy_info, Signal
)
from config import (
    QUICK_TEST_STOCKS, NIFTY_50_STOCKS, STOCK_UNIVERSE,
    get_core_stocks, get_active_stocks, get_all_stocks,
    get_sector_stocks, get_sector_info, print_sector_distribution,
    RiskConfig, Sector
)
from data_fetcher import YFinanceDataFetcher
from paper_trader import PaperTrader
from regime_detector import RegimeDetector
from risk_manager import RiskManager
from visualizer import Visualizer
from database import Database
from verdict import VerdictEngine
from backtester import Backtester


class DailyRunner:
    """
    Runs the full trading system pipeline.

    Modes:
        "core"     — Priority 1 stocks only (fast, ~30 stocks)
        "active"   — Priority 1+2 stocks (standard, ~40 stocks)
        "all"      — All stocks in universe (~45 stocks)
        "quick"    — QUICK_TEST_STOCKS (testing, 10 stocks)
        "custom"   — Pass your own list
    """

    SCAN_MODES = {
        "core":   get_core_stocks,
        "active": get_active_stocks,
        "all":    get_all_stocks,
        "quick":  lambda: QUICK_TEST_STOCKS,
    }

    def __init__(self, mode: str = "active", symbols: Optional[List[str]] = None,
                 enable_paper_trading: bool = True,
                 enable_backtesting: bool = False):
        """
        Args:
            mode: "core", "active", "all", or "quick"
            symbols: Custom symbol list (overrides mode)
            enable_paper_trading: Run paper trading simulation
            enable_backtesting: Run strategy backtesting
        """
        # Stock universe
        if symbols:
            self.symbols = symbols
            self.mode = "custom"
        elif mode in self.SCAN_MODES:
            self.symbols = self.SCAN_MODES[mode]()
            self.mode = mode
        else:
            print(f"  [WARNING] Unknown mode '{mode}', falling back to 'core'")
            self.symbols = get_core_stocks()
            self.mode = "core"

        # Core components
        self.fetcher = YFinanceDataFetcher()
        self.db = Database()
        self.risk_manager = RiskManager(risk_config=RiskConfig())

        # State
        self.regime: Optional[str] = None
        self.vix_value: Optional[float] = None
        self.nifty_data: Optional[pd.DataFrame] = None
        self.stock_data_cache: Dict[str, pd.DataFrame] = {}
        self.verdicts: List[Dict] = []
        self.signals_df: Optional[pd.DataFrame] = None
        self.consensus_df: Optional[pd.DataFrame] = None
        self.market_breadth: Dict = {}
        self.watchlist: List[Dict] = []
        self.start_time: Optional[datetime] = None

        # Config
        self.enable_paper_trading = enable_paper_trading
        self.enable_backtesting = enable_backtesting
        self.log_dir = "daily_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    # ============================================================
    #  MAIN PIPELINE
    # ============================================================

    def run_full_pipeline(self):
        """Execute the complete trading system pipeline."""
        self.start_time = datetime.now()
        today = self.start_time.strftime("%Y-%m-%d")

        print(f"\n{'█'*70}")
        print(f"  INDIAN MARKET TRADING SYSTEM — FULL PIPELINE")
        print(f"  Generated: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Mode: {self.mode} ({len(self.symbols)} stocks)")
        print(f"{'█'*70}")

        # Section 1: Market Regime
        self._run_section("MARKET REGIME", self._detect_regime)

        # Section 2: VIX Risk Gate
        self._run_section("INDIA VIX RISK GATE", self._check_vix)

        # Section 3: Pre-load data for all stocks
        self._run_section("DATA LOADING", self._load_all_data)

        # Section 4: Equity Research (all stocks)
        self._run_section("EQUITY RESEARCH", self._run_equity_research)

        # Section 5: Strategy Comparison
        if self.enable_backtesting:
            self._run_section("STRATEGY COMPARISON", self._compare_strategies)

        # Section 6: Daily Signal Scan + Consensus
        self._run_section("DAILY SIGNAL SCAN + CONSENSUS", self._run_signal_scan)

        # Section 7: Market Intelligence
        self._run_section("MARKET INTELLIGENCE", self._run_market_intelligence)

        # Section 8: Paper Trading
        if self.enable_paper_trading:
            self._run_section("PAPER TRADING", self._run_paper_trading)

        # Section 9: Charts
        self._run_section("CHARTS", self._generate_charts)

        # Section 10: Final Summary
        self._run_section("FINAL SUMMARY", self._print_final_summary)

        # Save report
        self._save_report(today)

    def _run_section(self, name, func):
        """Run a section with timing and error handling."""
        print(f"\n{'█'*70}")
        print(f"  {name}")
        print(f"{'█'*70}")
        try:
            func()
            print(f"\n  ✅ {name} — DONE")
        except Exception as e:
            print(f"\n  ❌ {name} — ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    #  SECTION 1: MARKET REGIME
    # ============================================================

    def _detect_regime(self):
        """Detect market regime using Nifty data."""
        print(f"\n  {'─'*50}")
        print(f"  Market Regime Detection")
        print(f"  {'─'*50}")

        self.nifty_data = self.fetcher.get_nifty_index(period="1y")
        if self.nifty_data is None or self.nifty_data.empty:
            print("  [ERROR] Could not fetch Nifty data")
            self.regime = "UNKNOWN"
            return

        detector = RegimeDetector()
        result = detector.detect(self.nifty_data)
        self.regime = result.get("regime", "UNKNOWN")

        # Print regime details
        print(f"\n  {'='*50}")
        print(f"  MARKET REGIME: {self._regime_icon()} {self.regime}")
        print(f"  {'='*50}")
        for key, value in result.items():
            if key != "regime":
                print(f"  {key}: {value}")

        # Update risk manager
        self.risk_manager.regime = self.regime

    def _regime_icon(self):
        icons = {"BULL": "↑", "BEAR": "↓", "SIDEWAYS": "↔", "UNKNOWN": "?"}
        return icons.get(self.regime, "?")

    # ============================================================
    #  SECTION 2: VIX RISK GATE
    # ============================================================

    def _check_vix(self):
        """Check India VIX and set risk parameters."""
        print(f"\n  {'─'*50}")
        print(f"  India VIX Risk Gate")
        print(f"  {'─'*50}")

        try:
            vix_data = self.fetcher.get_historical_data("^INDIAVIX", period="5d")
            if not vix_data.empty:
                self.vix_value = round(vix_data["Close"].iloc[-1], 1)
            else:
                self.vix_value = 15.0
                print("  [WARNING] VIX data unavailable, defaulting to 15.0")
        except Exception as e:
            self.vix_value = 15.0
            print(f"  [WARNING] VIX fetch failed ({e}), defaulting to 15.0")

        # Update risk manager
        self.risk_manager.update_vix(self.vix_value)

        # Print VIX dashboard
        print(f"\n  INDIA VIX RISK GATE:")
        print(f"  {'─'*50}")
        print(f"  VIX Level:    {self.vix_value}")
        print(f"  Regime:       {self.risk_manager.vix_regime}")
        print(f"  Position:     {self.risk_manager.vix_position_pct*100:.0f}% per trade")
        print(f"  Max Positions:{self.risk_manager.vix_max_positions}")
        print(f"  Message:      {self._vix_message()}")

        # Print thresholds table
        print(f"\n  VIX THRESHOLDS:")
        for threshold, regime, pos_pct, max_pos in RiskManager.VIX_THRESHOLDS:
            marker = " ◄" if regime == self.risk_manager.vix_regime else ""
            print(f"    < {threshold:<4} → {regime:<12} Position: {pos_pct*100:.0f}%  "
                  f"Max: {max_pos}{marker}")

    def _vix_message(self):
        messages = {
            "AGGRESSIVE": "Very low volatility — aggressive sizing allowed",
            "NORMAL": "Normal volatility — standard sizing",
            "CAUTIOUS": "Elevated volatility — reduce positions",
            "DEFENSIVE": "High volatility — defensive mode",
            "CASH_PRES": "Extreme volatility — preserve capital",
        }
        return messages.get(
            self.risk_manager.vix_regime,
            self.risk_manager.vix_message,  # fall back to the gate's real message
        )

    # ============================================================
    #  SECTION 3: DATA LOADING
    # ============================================================

    def _load_all_data(self, max_workers: int = 8):
        """Pre-load data for all stocks in parallel (network-bound)."""
        print(f"\n  Loading data for {len(self.symbols)} stocks "
              f"(parallel, {max_workers} workers)...")

        def _fetch(symbol):
            try:
                data = self.fetcher.get_historical_data(symbol, period="1y")
                if data is not None and not data.empty and len(data) >= 50:
                    return symbol, data, None
                return symbol, None, "insufficient data"
            except Exception as e:
                return symbol, None, str(e)

        fetched = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_fetch, s): s for s in self.symbols}
            for fut in as_completed(futures):
                symbol, data, err = fut.result()
                fetched[symbol] = (data, err)

        loaded = 0
        failed = 0
        for symbol in self.symbols:
            data, err = fetched[symbol]
            if data is not None:
                self.stock_data_cache[symbol] = data
                loaded += 1
                print(f"  {symbol} ✓ ({len(data)} bars)")
            else:
                failed += 1
                print(f"  {symbol} ✗ ({err})")

        print(f"\n  Data loaded: {loaded}/{len(self.symbols)} stocks ({failed} failed)")

    # ============================================================
    #  SECTION 4: EQUITY RESEARCH
    # ============================================================

    def _run_equity_research(self):
        """Run full equity research on all loaded stocks."""
        engine = VerdictEngine()
        self.verdicts = []

        total = len(self.stock_data_cache)
        print(f"\n  Running research on {total} stocks...")
        print(f"  (Each stock: Fundamentals → Management → Valuation → Technicals → Risks)\n")

        for i, symbol in enumerate(self.stock_data_cache.keys(), 1):
            print(f"\n  Research: {symbol} ({i}/{total})")
            print(f"  {'─'*50}")
            try:
                verdict = engine.generate_verdict(symbol)
                engine.print_verdict(verdict)

                # Store verdict as dict for ranking
                self.verdicts.append({
                    "symbol": verdict.symbol,
                    "company_name": verdict.company_name,
                    "sector": verdict.sector,
                    "action": verdict.action.value,
                    "conviction_score": verdict.conviction_score,
                    "current_price": verdict.current_price,
                    "interesting_price": verdict.interesting_price,
                    "stop_loss": verdict.stop_loss_price,
                    "target": verdict.target_price,
                    "fundamental_score": verdict.fundamental_score,
                    "management_score": verdict.management_score,
                    "valuation_score": verdict.valuation_score,
                    "technical_score": verdict.technical_score,
                    "risk_score": verdict.risk_score,
                    "bull_points": verdict.key_bull_points,
                    "bear_points": verdict.key_bear_points,
                    "risk_factors": [
                        {"category": r.category, "risk": r.risk,
                         "impact": r.impact, "probability": r.probability}
                        for r in verdict.risk_factors
                    ],
                    "red_flags": [
                        p.replace("RED FLAG: ", "")
                        for p in verdict.key_bear_points
                        if "RED FLAG" in p
                    ],
                    "valuation_rating": verdict.valuation_summary,
                    "one_line": verdict.one_line_summary,
                })

            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")

        # Print rankings
        self._print_stock_rankings()

    def _print_stock_rankings(self):
        """Print ranked stock table."""
        if not self.verdicts:
            return

        # Sort by conviction score
        ranked = sorted(self.verdicts, key=lambda x: x["conviction_score"], reverse=True)

        print(f"\n{'█'*70}")
        print(f"  STOCK RANKINGS")
        print(f"{'█'*70}")
        print(f"  {'#':<4} {'Symbol':<16} {'Action':<14} {'Conv':<6} "
              f"{'Fund':<6} {'Mgmt':<6} {'Val':<6} {'Tech':<6} {'Price'}")
        print(f"  {'─'*75}")

        for i, v in enumerate(ranked, 1):
            icon = self._action_icon(v["action"])
            print(f"  {i:<4} {v['symbol']:<16} {v['action']:<14} "
                  f"{v['conviction_score']:<6.1f} "
                  f"{v['fundamental_score']:<6.1f} "
                  f"{v['management_score']:<6.1f} "
                  f"{v['valuation_score']:<6.1f} "
                  f"{v['technical_score']:<6.1f} "
                  f"₹{v['current_price']:,.0f}")

        print(f"{'█'*70}")

    # ============================================================
    #  SECTION 5: STRATEGY COMPARISON
    # ============================================================

    def _compare_strategies(self):
        """Backtest and compare all strategies."""
        # Use first 5 stocks with data for strategy comparison
        test_stocks = list(self.stock_data_cache.keys())[:5]
        if not test_stocks:
            print("  No stock data available for backtesting")
            return

        data_dict = {s: self.stock_data_cache[s] for s in test_stocks}
        bt = Backtester(initial_capital=500000, position_size_pct=0.15)

        strategies = get_regime_strategies(self.regime or "SIDEWAYS")
        comparison = []

        # Show strategy weights
        print(f"\n  STRATEGY WEIGHTS ({self.regime or 'UNKNOWN'} regime):")
        print(f"  {'Strategy':<40} {'Category':<18} {'Weight':<7} {'Bar'}")
        print(f"  {'─'*70}")
        for strat in strategies:
            weight = self._get_strategy_weight(strat)
            bar = "█" * int(weight * 10) + "░" * (20 - int(weight * 10))
            print(f"  {strat.name:<40} {strat.category:<18} {weight:<5.1f}x {bar}")

        # Backtest each strategy
        for strat in strategies:
            print(f"  Testing: {strat.name}...")
            try:
                r = bt.run_multi_stock_backtest(strat, data_dict)
                a = r["aggregate_metrics"]
                if a.get("total_trades", 0) > 0:
                    comparison.append({
                        "Type": "PINE" if "Structure" in strat.name or "RTS" in strat.name
                               or "Bear" in strat.name else "ORIGINAL",
                        "Category": strat.category,
                        "Strategy": strat.name,
                        "Trades": a["total_trades"],
                        "WinRate%": a["win_rate"],
                        "Return%": a["total_return_pct"],
                        "MaxDD%": a["max_drawdown_pct"],
                        "Sharpe": a["sharpe_ratio"],
                        "ProfitFactor": a["profit_factor"]
                    })
            except Exception as e:
                print(f"  [WARNING] Backtest failed for {strat.name}: {e}")

        if comparison:
            df = pd.DataFrame(comparison).sort_values("Sharpe", ascending=False)
            print(f"\n{'='*90}")
            print(f"  ALL STRATEGIES COMPARED — Sorted by Sharpe (Regime: {self.regime})")
            print(f"{'='*90}")
            print(df.to_string(index=False))

            # Best by category
            print(f"\n  Best by category:")
            for cat in df["Category"].unique():
                best = df[df["Category"] == cat].iloc[0]
                print(f"    {cat}: {best['Strategy']} "
                      f"(Sharpe: {best['Sharpe']}, Return: {best['Return%']}%)")

    def _get_strategy_weight(self, strat):
        """Get regime-adjusted weight for a strategy."""
        regime_weights = {
            "BULL": {"trend": 1.5, "momentum": 1.2, "mean_reversion": 0.5,
                     "structure": 1.0, "adaptive": 1.0, "bear_short": 0.0},
            "BEAR": {"trend": 0.3, "momentum": 0.3, "mean_reversion": 1.2,
                     "structure": 1.0, "adaptive": 0.8, "bear_short": 1.3},
            "SIDEWAYS": {"trend": 1.0, "momentum": 1.0, "mean_reversion": 1.0,
                          "structure": 1.0, "adaptive": 1.0, "bear_short": 0.5},
        }
        weights = regime_weights.get(self.regime or "SIDEWAYS", regime_weights["SIDEWAYS"])
        return weights.get(strat.category, 1.0)

    # ============================================================
    #  SECTION 6: SIGNAL SCAN + CONSENSUS
    # ============================================================

    def _run_signal_scan(self):
        """Run daily signal scan with regime-aware strategies."""
        strategies = get_regime_strategies(self.regime or "SIDEWAYS")

        print(f"\n  Scanning {len(self.stock_data_cache)} stocks "
              f"with {len(strategies)} strategies...")

        scanner = SignalScanner(
            symbols=list(self.stock_data_cache.keys()),
            strategies=strategies,
            regime=self.regime or "SIDEWAYS"
        )

        # Use cached data
        scanner.fetcher = self.fetcher

        self.signals_df = scanner.scan(data_cache=self.stock_data_cache,
                                        max_workers=8)
        scanner.print_signals(self.signals_df)

        # Consensus
        self.consensus_df = scanner.scan_for_consensus()
        if self.consensus_df is not None and not self.consensus_df.empty:
            resolved = scanner.resolve_conflicts(self.consensus_df)
            scanner.print_consensus(resolved if not resolved.empty else self.consensus_df)
        else:
            print(f"\n  No consensus signals.")

        # Signal statistics
        if not self.signals_df.empty:
            buy_count = len(self.signals_df[self.signals_df["Signal"] == "BUY"])
            sell_count = len(self.signals_df[self.signals_df["Signal"] == "SELL"])
            short_count = len(self.signals_df[self.signals_df["Signal"] == "SHORT"])
            print(f"\n  SIGNAL STATISTICS:")
            print(f"    BUY signals:   {buy_count}")
            print(f"    SELL signals:  {sell_count}")
            print(f"    SHORT signals: {short_count}")

            # Pine strategy signals
            pine_signals = self.signals_df[
                self.signals_df["Strategy"].str.contains("Structure|RTS|Bear", case=False)
            ]
            if not pine_signals.empty:
                print(f"\n  PINE/BEAR STRATEGY SIGNALS:")
                for _, r in pine_signals.iterrows():
                    print(f"    {r['Signal']} {r['Symbol']} @ ₹{r['Price']:,.2f} "
                          f"({r['Strategy']})")

    # ============================================================
    #  SECTION 7: MARKET INTELLIGENCE
    # ============================================================

    def _run_market_intelligence(self):
        """Build market intelligence: breadth, watchlist, recommendation."""
        # Market breadth
        self._calculate_breadth()

        # Intelligent watchlist
        self._build_watchlist()

        # Final recommendation
        self._generate_recommendation()

    def _calculate_breadth(self):
        """Calculate market breadth indicators."""
        above_20sma = 0
        above_50sma = 0
        total = 0

        for symbol, data in self.stock_data_cache.items():
            if len(data) < 50:
                continue
            total += 1
            close = data["Close"]
            cur = close.iloc[-1]

            sma20 = close.rolling(20).mean().iloc[-1]
            if cur > sma20:
                above_20sma += 1

            sma50 = close.rolling(50).mean().iloc[-1]
            if cur > sma50:
                above_50sma += 1

        pct_20 = (above_20sma / total * 100) if total > 0 else 0
        pct_50 = (above_50sma / total * 100) if total > 0 else 0

        if pct_20 > 65:
            breadth_regime = "STRONG"
        elif pct_20 > 50:
            breadth_regime = "MODERATE"
        elif pct_20 > 35:
            breadth_regime = "WEAK"
        else:
            breadth_regime = "VERY_WEAK"

        self.market_breadth = {
            "stocks_analyzed": total,
            "above_20sma": above_20sma,
            "above_50sma": above_50sma,
            "pct_above_20sma": round(pct_20, 1),
            "pct_above_50sma": round(pct_50, 1),
            "breadth_regime": breadth_regime,
        }

        print(f"\n  MARKET BREADTH:")
        print(f"  {'─'*50}")
        print(f"  Stocks analyzed:   {total}")
        print(f"  Above 20 SMA:      {above_20sma}/{total} ({pct_20:.1f}%)")
        print(f"  Above 50 SMA:      {above_50sma}/{total} ({pct_50:.1f}%)")
        print(f"  Breadth Regime:    {breadth_regime}")
        print(f"  Message:           {self._breadth_message(breadth_regime)}")

        bar_filled = int(pct_20 / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"  [{bar}] {pct_20:.0f}%")

    def _breadth_message(self, regime):
        messages = {
            "STRONG": "Broad participation — trend-following works",
            "MODERATE": "Mixed participation — selective entries",
            "WEAK": "Narrow market — caution advised",
            "VERY_WEAK": "Broad weakness — defensive positioning",
        }
        return messages.get(regime, "")

    def _build_watchlist(self):
        """Build intelligent watchlist with buy zones."""
        self.watchlist = []

        for v in self.verdicts:
            symbol = v["symbol"]
            current = v["current_price"]
            interesting = v["interesting_price"]
            action = v["action"]
            conviction = v["conviction_score"]

            if current <= 0 or interesting <= 0:
                continue

            gap_pct = ((current - interesting) / interesting) * 100

            # Status based on gap
            if gap_pct <= 0:
                status = "🟢 IN BUY ZONE"
            elif gap_pct <= 5:
                status = "🟡 VERY CLOSE"
            elif gap_pct <= 15:
                status = "🟡 APPROACHING"
            elif gap_pct <= 20:
                status = "⚪ MONITORING"
            else:
                status = "🔵 FAR"

            # Override status for avoid stocks
            if action in ["STRONG_AVOID", "AVOID"]:
                if gap_pct <= 15:
                    status = "🟡 APPROACHING (AVOID)"
                else:
                    status = "⚪ PASSED (AVOID)"

            self.watchlist.append({
                "symbol": symbol,
                "price": current,
                "buy_zone": interesting,
                "gap_pct": round(gap_pct, 1),
                "conviction": conviction,
                "action": action,
                "status": status,
            })

        # Sort by gap (closest to buy zone first)
        self.watchlist.sort(key=lambda x: x["gap_pct"])

        # Print watchlist
        print(f"\n  INTELLIGENT WATCHLIST:")
        print(f"  {'─'*90}")
        print(f"  {'#':<4} {'Symbol':<16} {'Price':<10} {'Buy Zone':<10} "
              f"{'Gap%':<8} {'Conv':<6} {'Action':<14} {'Status'}")
        print(f"  {'─'*90}")

        for i, w in enumerate(self.watchlist, 1):
            print(f"  {i:<4} {w['symbol']:<16} ₹{w['price']:<9,.0f} "
                  f"₹{w['buy_zone']:<9,.0f} {w['gap_pct']:>+6.1f}% "
                  f"{w['conviction']:<6.1f} {w['action']:<14} {w['status']}")

        # Summary
        in_zone = [w for w in self.watchlist if "IN BUY ZONE" in w["status"]]
        approaching = [w for w in self.watchlist if "APPROACHING" in w["status"]
                       and "AVOID" not in w["status"]]

        print(f"\n  SUMMARY:")
        print(f"    Total watched:     {len(self.watchlist)}")
        print(f"    In buy zone:       {len(in_zone)}")
        print(f"    Approaching (<15%):{len(approaching)}")

        if in_zone:
            print(f"\n  🟢 IN BUY ZONE:")
            for w in in_zone:
                print(f"    {w['symbol']}: ₹{w['price']:,.0f} "
                      f"(buy below ₹{w['buy_zone']:,.0f})")
        elif approaching:
            closest = approaching[0]
            print(f"\n  🟡 CLOSEST TO BUY ZONE:")
            print(f"    {closest['symbol']}: ₹{closest['price']:,.0f} "
                  f"→ ₹{closest['buy_zone']:,.0f} ({closest['gap_pct']:.1f}% away)")

    def _generate_recommendation(self):
        """Generate final actionable recommendation."""
        # Count signals by type
        in_zone = len([w for w in self.watchlist if "IN BUY ZONE" in w["status"]])
        conviction_signals = 0
        strong_signals = 0

        if self.consensus_df is not None and not self.consensus_df.empty:
            if "Strength" in self.consensus_df.columns:
                conviction_signals = len(
                    self.consensus_df[self.consensus_df["Strength"] == "CONVICTION"]
                )
                strong_signals = len(
                    self.consensus_df[self.consensus_df["Strength"] == "STRONG"]
                )

        # Decision logic
        if conviction_signals > 0 and in_zone > 0:
            action = "BUY"
            message = f"{conviction_signals} conviction signals + {in_zone} in buy zone"
        elif strong_signals > 0:
            action = "PREPARE"
            message = f"{strong_signals} strong signals — wait for buy zone alignment"
        elif in_zone > 0:
            action = "WATCH"
            message = f"{in_zone} stocks in buy zone — wait for signal confirmation"
        else:
            action = "WAIT"
            message = "No strong signals — waiting for alignment"

        print(f"\n  {'='*50}")
        print(f"  INTELLIGENT RECOMMENDATION")
        print(f"  {'='*50}")
        action_icon = {
            "BUY": "🟢", "PREPARE": "🟡", "WATCH": "🟡", "WAIT": "🔴"
        }
        print(f"  {action_icon.get(action, '')} ACTION: {action}")
        print(f"  {message}")
        print(f"  Position Size: {self.risk_manager.vix_position_pct*100:.0f}% per trade")
        print(f"  Max Positions: {self.risk_manager.vix_max_positions}")
        print(f"  {'='*50}")

    # ============================================================
    #  SECTION 8: PAPER TRADING
    # ============================================================

    def _run_paper_trading(self):
        """Run paper trading simulation with regime-aware strategies."""
        from strategies import get_regime_strategies

        strategies = get_regime_strategies(self.regime or "SIDEWAYS")
        test_stocks = list(self.stock_data_cache.keys())[:5]

        if not test_stocks:
            print("  No data for paper trading")
            return

        # Run for each Pine/Bear strategy
        target_strategies = [
            s for s in strategies
            if "bear" in s.category or "Structure" in s.name or "RTS" in s.name
        ]

        if not target_strategies:
            target_strategies = strategies[:2]

        for strat in target_strategies:
            print(f"\n  Simulating: {strat.name} "
                  f"(position: {self.risk_manager.vix_position_pct*100:.0f}%)")

            trader = PaperTrader(
                initial_capital=500000.0,
                risk_config=RiskConfig()
            )

            # Simulate on each stock
            for symbol in test_stocks:
                data = self.stock_data_cache.get(symbol)
                if data is None or data.empty:
                    continue

                try:
                    df = strat.generate_signals(data)
                    if df.empty:
                        continue

                    atr_series = PaperTrader.calculate_atr(data)

                    for idx in range(len(df)):
                        row = df.iloc[idx]
                        signal = int(row["Signal"])
                        if signal == 0:  # HOLD
                            continue

                        price = row["Close"]
                        date = str(df.index[idx].date())
                        atr_val = atr_series.iloc[idx] if not pd.isna(atr_series.iloc[idx]) else None

                        # Only entries (BUY=1, SHORT=-2) are risk-gated;
                        # exits (SELL=-1, COVER=2) always pass through.
                        if signal in (1, -2):
                            side = "BUY" if signal == 1 else "SHORT"
                            open_symbols = list(trader.positions.keys()) + \
                                list(getattr(trader, "short_positions", {}).keys())
                            allowed, reason, _ = self.risk_manager.check_pre_trade(
                                symbol,
                                side,
                                open_symbols,
                                price,
                                atr_val or 0
                            )
                            if not allowed:
                                continue

                        trader.process_signal(
                            symbol, signal, price, date, strat.name,
                            atr=atr_val, bar_index=idx
                        )

                except Exception as e:
                    print(f"  [WARNING] Paper trading failed for {symbol}: {e}")

            # Close remaining positions
            for symbol in list(trader.positions.keys()):
                data = self.stock_data_cache.get(symbol)
                if data is not None and not data.empty:
                    trader._close_position(
                        symbol, data["Close"].iloc[-1],
                        str(data.index[-1].date()), "END_SIMULATION"
                    )

            # Print results
            trader.print_trade_log()
            summary = trader.get_performance_summary()
            if "message" not in summary:
                print(f"\n  {strat.name} PERFORMANCE:")
                for k, v in summary.items():
                    if k not in ("exit_reasons",):
                        print(f"    {k}: {v}")

    # ============================================================
    #  SECTION 9: CHARTS
    # ============================================================

    def _generate_charts(self):
        """Generate all charts in one call."""
        try:
            viz = Visualizer()

            # Sector data from verdicts
            sector_counts = {}
            for v in self.verdicts:
                sector = v.get("sector", "Unknown")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            # Generate everything
            viz.generate_all_charts(
                verdicts=self.verdicts,
                signals_df=self.signals_df,
                watchlist=self.watchlist,
                breadth_data=self.market_breadth,
                risk_data=self.risk_manager.get_dashboard(
                    {}  # positions from paper trader if available
                ),
                sector_data=sector_counts,
                stock_data_cache=self.stock_data_cache,
            )

        except Exception as e:
            print(f"  [WARNING] Chart generation error: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    #  SECTION 10: FINAL SUMMARY
    # ============================================================

    def _print_final_summary(self):
        """Print comprehensive final summary."""
        duration = (datetime.now() - self.start_time).total_seconds()

        # Sector distribution
        sector_counts = {}
        for v in self.verdicts:
            sector = v.get("sector", "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Action distribution
        action_counts = {}
        for v in self.verdicts:
            action = v["action"]
            action_counts[action] = action_counts.get(action, 0) + 1

        # Print summary
        print(f"\n  Duration:         {duration:.1f}s")
        print(f"  Stocks Analyzed:  {len(self.verdicts)}")
        print(f"  Strategies:       {len(get_regime_strategies(self.regime or 'SIDEWAYS'))}")
        print(f"  Market Regime:    {self.regime}")
        print(f"  India VIX:        {self.vix_value}")
        print(f"  Position Size:    {self.risk_manager.vix_position_pct*100:.0f}%")
        print(f"  Risk Mode:        {self.risk_manager.vix_regime}")

        # Action distribution
        print(f"\n  ACTION DISTRIBUTION:")
        for action, count in sorted(action_counts.items()):
            print(f"    {action}: {count}")

        # Sector distribution
        print(f"\n  SECTOR COVERAGE:")
        for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
            print(f"    {sector}: {count}")

        # Final rankings
        ranked = sorted(self.verdicts, key=lambda x: x["conviction_score"],
                       reverse=True)

        print(f"\n  {'='*60}")
        print(f"  FINAL STOCK RANKINGS")
        print(f"  {'='*60}")
        for i, v in enumerate(ranked, 1):
            icon = self._action_icon(v["action"])
            print(f"  {i:>3}. {icon} {v['symbol']:<16} "
                  f"{v['conviction_score']}/10  ₹{v['current_price']:,.0f}")
        print(f"  {'='*60}")

        # Recommendation
        in_zone = len([w for w in self.watchlist if "IN BUY ZONE" in w["status"]])
        print(f"\n  NEXT STEPS:")
        if in_zone > 0:
            print(f"  1. ACTION: REVIEW {in_zone} stocks in buy zone")
        else:
            print(f"  1. ACTION: WAIT")
        print(f"  2. Position Size: {self.risk_manager.vix_position_pct*100:.0f}%")

    def _action_icon(self, action):
        icons = {
            "STRONG_BUY": "🟢🟢",
            "BUY": "🟢",
            "HOLD": "🟡",
            "AVOID": "🔴",
            "STRONG_AVOID": "🔴🔴",
        }
        return icons.get(action, "")

    # ============================================================
    #  REPORT SAVING
    # ============================================================

    def _save_report(self, today):
        """Save all results to files."""
        # Verdicts to JSON
        verdicts_file = os.path.join(self.log_dir, f"verdicts_{today}.json")
        with open(verdicts_file, "w") as f:
            json.dump(self.verdicts, f, indent=2, default=str)
        print(f"\n  Verdicts saved: {verdicts_file}")

        # Watchlist to CSV
        if self.watchlist:
            wl_file = os.path.join(self.log_dir, f"watchlist_{today}.csv")
            pd.DataFrame(self.watchlist).to_csv(wl_file, index=False)
            print(f"  Watchlist saved: {wl_file}")

        # Signals to CSV
        if self.signals_df is not None and not self.signals_df.empty:
            sig_file = os.path.join(self.log_dir, f"signals_{today}.csv")
            self.signals_df.to_csv(sig_file, index=False)
            print(f"  Signals saved: {sig_file}")

        # Log to database
        if self.signals_df is not None and not self.signals_df.empty:
            for _, row in self.signals_df.iterrows():
                try:
                    self.db.log_signal(
                        row["Date"], row["Symbol"], row["Strategy"],
                        row["Signal"], row["Price"]
                    )
                except Exception as e:
                    print(f"  [WARNING] DB log failed for {row['Symbol']}: {e}")

        print(f"  Report complete: {self.log_dir}/")

    # ============================================================
    #  SCHEDULER
    # ============================================================

    def schedule_daily(self):
        """Schedule automatic daily runs."""
        print("  Scheduler started. Scans at 09:30 and 15:15 IST.")
        schedule.every().day.at("09:30").do(self.run_full_pipeline)
        schedule.every().day.at("15:15").do(self.run_full_pipeline)
        while True:
            schedule.run_pending()
            time.sleep(60)

    # ============================================================
    #  QUICK SCAN (backward compatible)
    # ============================================================

    def run_daily_scan(self):
        """Quick scan only (backward compatible with original)."""
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"\n{'='*60}")
        print(f"  DAILY SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")

        strategies = get_regime_strategies(self.regime or "SIDEWAYS")
        scanner = SignalScanner(symbols=self.symbols, strategies=strategies)
        signals = scanner.scan()
        scanner.print_signals(signals)

        consensus = scanner.scan_for_consensus()
        if not consensus.empty:
            scanner.print_consensus(consensus)

        # Save to DB
        if not signals.empty:
            for _, row in signals.iterrows():
                self.db.log_signal(row["Date"], row["Symbol"], row["Strategy"],
                                  row["Signal"], row["Price"])
            signals.to_csv(
                os.path.join(self.log_dir, f"signals_{today}.csv"), index=False
            )
        print(f"\n  Scan complete.")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print(f"""
{'█'*70}
  INDIAN MARKET TRADING SYSTEM — DAILY RUNNER v3.1
  Sector-Balanced Universe | Regime-Adaptive | Risk-Managed
{'█'*70}

  SCAN MODES:
    1. Full Pipeline — Core stocks (priority 1 only)
    2. Full Pipeline — Active stocks (priority 1+2)
    3. Full Pipeline — All stocks (full universe)
    4. Full Pipeline — Quick test (10 stocks)
    5. Quick Scan only (signals + consensus)
    6. Sector Distribution Report
    """)

    choice = input("Enter 1-6: ").strip()

    if choice == "1":
        runner = DailyRunner(mode="core")
        runner.run_full_pipeline()
    elif choice == "2":
        runner = DailyRunner(mode="active")
        runner.run_full_pipeline()
    elif choice == "3":
        runner = DailyRunner(mode="all")
        runner.run_full_pipeline()
    elif choice == "4":
        runner = DailyRunner(mode="quick")
        runner.run_full_pipeline()
    elif choice == "5":
        runner = DailyRunner(mode="active")
        runner.run_daily_scan()
    elif choice == "6":
        print_sector_distribution()
        info = get_sector_info()
        print(f"\n  Core stocks: {len(get_core_stocks())}")
        print(f"  Active stocks: {len(get_active_stocks())}")
        print(f"  All stocks: {len(get_all_stocks())}")
    else:
        runner = DailyRunner(mode="active")
        runner.run_full_pipeline()