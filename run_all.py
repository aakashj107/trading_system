#!/usr/bin/env python3
"""
Indian Market Trading System — Run Everything v3 (Option C)
Intelligence Layer: Regime-Adaptive + VIX Risk Gate + Consensus + Watchlist
"""

import sys
import os
import traceback
from datetime import datetime


# ── VALIDATED LIVE CONFIG (see live_run.py / trading-system-edge-diagnosis) ──
# 10y walk-forward on the REAL PaperTrader showed the system is only tradable
# with: (a) consensus min_agree=2 (trade only when >=2 of 17 strategies agree,
# kills noise-churn that bled -92% at min_agree=1) and (b) long-only (the short
# leg lost money under Indian STT + bull grinds). Result: +54.9% / 10y, PF 1.27.
CONSENSUS_MIN_AGREE = 2
ALLOW_SHORT = False
PAPER_CAPITAL = 1_000_000.0
PAPER_SIZING = "volatility"  # ATR-stop risk-based; lifts ann. return ~6.9%->10.4%
PAPER_CB = False            # CB tested: cuts return & worsens DD on this portfolio; off by default
# ── Output Capture ──────────────────────────────────────────
class TeeOutput:
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def section(title):
    print(f"\n{'█' * 70}")
    print(f"  {title}")
    print(f"{'█' * 70}\n")


def sub(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


def safe(desc, func, *args, **kwargs):
    sub(desc)
    try:
        result = func(*args, **kwargs)
        print(f"\n  ✅ {desc} — DONE")
        return result
    except Exception as e:
        print(f"\n  ❌ {desc} — ERROR: {e}")
        traceback.print_exc()
        return None


def fetch_with_retry(fetcher, symbol, period="2y", retries=2):
    """Fetch data with retry and symbol alias support."""
    from expanded_config import SYMBOL_ALIASES
    aliases = SYMBOL_ALIASES.get(symbol, [symbol])
    for alt_symbol in aliases:
        for attempt in range(retries):
            df = fetcher.get_historical_data(alt_symbol, period=period)
            if not df.empty:
                return df
    print(f"  [WARNING] No data for {symbol}")
    import pandas as pd
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
def main():
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"reports/full_report_{timestamp}.txt"

    tee = TeeOutput(report_file)
    sys.stdout = tee

    start_time = datetime.now()

    print(f"{'=' * 70}")
    print(f"  INDIAN MARKET TRADING SYSTEM — FULL REPORT v3 (Option C)")
    print(f"  Generated: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Report: {report_file}")
    print(f"  Features: Regime-Adaptive + VIX Risk + Consensus + Watchlist")
    print(f"{'=' * 70}")

    # ── Import ───────────────────────────────────────────────
    section("IMPORTING MODULES")
    try:
        from data_fetcher import YFinanceDataFetcher
        from strategies import get_all_strategies
        from backtester import Backtester
        from paper_trader import PaperTrader
        from scanner import SignalScanner
        from visualizer import Visualizer
        from regime_detector import RegimeDetector
        from fundamentals import FundamentalAnalyzer
        from management_analyzer import ManagementAnalyzer
        from valuation import ValuationAnalyzer
        from verdict import VerdictEngine
        from expanded_config import EXPANDED_UNIVERSE
        from strategies_pine import get_pine_strategies, get_all_strategies_with_pines
        from market_intelligence import (
            MarketIntelligence, RegimeAdaptiveManager,
            IndiaVIXRiskGate, MarketBreadth, ConsensusFilter,
            WatchlistManager, fetch_india_vix,
        )

        print("  All modules imported ✅")
        print(f"  Original strategies: {len(get_all_strategies())}")
        print(f"  Pine strategies:     {len(get_pine_strategies())}")
        print(f"  Total strategies:    {len(get_all_strategies_with_pines())}")
        print(f"  Stock universe:      {len(EXPANDED_UNIVERSE)}")
    except Exception as e:
        print(f"  IMPORT ERROR: {e}")
        traceback.print_exc()
        sys.stdout = tee.terminal
        tee.close()
        print(f"\nReport: {report_file}")
        return

    fetcher = YFinanceDataFetcher()
    all_strategies = get_all_strategies_with_pines()
    engine = VerdictEngine()

    # ════════════════════════════════════════════════════════
    # SECTION 1: MARKET REGIME
    # ════════════════════════════════════════════════════════
    section("SECTION 1: MARKET REGIME")

    def run_regime():
        nifty = fetcher.get_nifty_index(period="1y")
        if nifty.empty:
            print("  No Nifty data")
            return "UNKNOWN", {}
        rd = RegimeDetector()
        rd.print_regime_report(nifty)
        regime, details = rd.detect_regime(nifty)
        regime_str = details.get("regime", "UNKNOWN")
        return regime_str, details

    regime_result = safe("Market Regime Detection", run_regime)
    regime_str = regime_result[0] if regime_result else "UNKNOWN"

    # ════════════════════════════════════════════════════════
    # SECTION 2: INDIA VIX RISK GATE
    # ════════════════════════════════════════════════════════
    section("SECTION 2: INDIA VIX RISK GATE")

    def run_vix():
        vix_value = fetch_india_vix(fetcher)
        print(f"  India VIX: {vix_value:.1f}")
        vix_gate = IndiaVIXRiskGate()
        risk_profile = vix_gate.print_risk_profile(vix_value)
        return vix_value, risk_profile

    vix_result = safe("India VIX Risk Gate", run_vix)
    vix_value = vix_result[0] if vix_result else 18.0
    risk_profile = vix_result[1] if vix_result else IndiaVIXRiskGate().get_risk_profile(18.0)

    # ════════════════════════════════════════════════════════
    # SECTION 3: EQUITY RESEARCH
    # ════════════════════════════════════════════════════════
    section("SECTION 3: EQUITY RESEARCH")

    research_stocks = EXPANDED_UNIVERSE[:15]
    all_verdicts = []

    for symbol in research_stocks:
        def run_research(sym=symbol):
            verdict = engine.generate_verdict(sym)
            engine.print_verdict(verdict)
            return verdict
        result = safe(f"Research: {symbol}", run_research)
        if result:
            all_verdicts.append(result)

    # Rankings
    if all_verdicts:
        sub("STOCK RANKINGS")
        sorted_v = sorted(all_verdicts,
                          key=lambda v: v.conviction_score, reverse=True)
        print(f"  {'#':<4} {'Symbol':<16} {'Action':<14} {'Conv':<6} "
              f"{'Fund':<5} {'Mgmt':<5} {'Val':<5} {'Tech':<5} {'Price':<10}")
        print(f"  {'─' * 75}")
        for i, v in enumerate(sorted_v, 1):
            print(f"  {i:<4} {v.symbol:<16} {v.action.value:<14} "
                  f"{v.conviction_score:<6.1f} "
                  f"{v.fundamental_score:<5.1f} "
                  f"{v.management_score:<5.1f} "
                  f"{v.valuation_score:<5.1f} "
                  f"{v.technical_score:<5.1f} "
                  f"₹{v.current_price:<9,.0f}")

    # ════════════════════════════════════════════════════════
    # SECTION 4: ALL STRATEGIES COMPARED
    # ════════════════════════════════════════════════════════
    section("SECTION 4: ALL STRATEGIES COMPARED (12 strategies)")

    import pandas as pd

    def run_comparison():
        test_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
                       "INFY.NS", "ICICIBANK.NS"]
        data_dict = {}
        for sym in test_stocks:
            df = fetch_with_retry(fetcher, sym, period="2y")
            if not df.empty:
                data_dict[sym] = df

        if not data_dict:
            print("  No data")
            return None

        # Show regime-adaptive weights
        rm = RegimeAdaptiveManager()
        rm.print_weights(regime_str, all_strategies)

        bt = Backtester(initial_capital=500000,
                        position_size_pct=risk_profile.position_pct)
        comparison = []

        for strategy in all_strategies:
            print(f"  Testing: {strategy.name}...")
            result = bt.run_multi_stock_backtest(strategy, data_dict)
            agg = result["aggregate_metrics"]
            if agg.get("total_trades", 0) > 0:
                is_pine = "PINE" if any(k in strategy.name for k in
                                        ["RTS", "Structure", "Combined"]) \
                    else "ORIGINAL"
                cat = rm.get_category(strategy.name)
                comparison.append({
                    "Type": is_pine,
                    "Category": cat,
                    "Strategy": strategy.name,
                    "Trades": agg["total_trades"],
                    "WinRate%": agg["win_rate"],
                    "Return%": agg["total_return_pct"],
                    "MaxDD%": agg["max_drawdown_pct"],
                    "Sharpe": agg["sharpe_ratio"],
                    "ProfitFactor": agg["profit_factor"],
                })

        if comparison:
            df_comp = pd.DataFrame(comparison).sort_values("Sharpe",
                                                            ascending=False)
            print(f"\n{'=' * 100}")
            print(f"  ALL STRATEGIES COMPARED — Sorted by Sharpe "
                  f"(Regime: {regime_str})")
            print(f"{'=' * 100}")
            print(df_comp.to_string(index=False))

            # Best by category
            for cat in ["mean_reversion", "structure", "adaptive",
                        "trend", "momentum"]:
                cat_df = df_comp[df_comp["Category"] == cat]
                if not cat_df.empty:
                    best = cat_df.iloc[0]
                    print(f"\n  Best {cat}: {best['Strategy']} "
                          f"(Sharpe: {best['Sharpe']}, "
                          f"Return: {best['Return%']}%)")

            # Pine vs Original
            pine_df = df_comp[df_comp["Type"] == "PINE"]
            orig_df = df_comp[df_comp["Type"] == "ORIGINAL"]
            if not pine_df.empty and not orig_df.empty:
                print(f"\n  📊 PINE vs ORIGINAL:")
                print(f"     Best Pine:     {pine_df.iloc[0]['Strategy']} "
                      f"(Sharpe: {pine_df.iloc[0]['Sharpe']})")
                print(f"     Best Original: {orig_df.iloc[0]['Strategy']} "
                      f"(Sharpe: {orig_df.iloc[0]['Sharpe']})")

            return df_comp
        return None

    safe("Full Strategy Comparison", run_comparison)

    # ════════════════════════════════════════════════════════
    # SECTION 5: DAILY SIGNAL SCAN + CONSENSUS
    # ════════════════════════════════════════════════════════
    section("SECTION 5: DAILY SIGNAL SCAN + CONSENSUS")

    def run_scan():
        scan_stocks = EXPANDED_UNIVERSE[:20]
        scanner = SignalScanner(symbols=scan_stocks,
                                strategies=all_strategies)
        signals = scanner.scan()
        scanner.print_signals(signals)

        if not signals.empty:
            # Consensus analysis
            rm = RegimeAdaptiveManager()
            weights = rm.get_weights(regime_str, all_strategies)
            cf = ConsensusFilter()
            consensus = cf.find_consensus(signals, weights)
            cf.print_consensus(consensus)

            # Stats
            print(f"\n  SIGNAL STATISTICS:")
            print(f"    BUY signals:  {len(signals[signals['Signal'] == 'BUY'])}")
            print(f"    SELL signals: {len(signals[signals['Signal'] == 'SELL'])}")

            # Pine strategy signals
            pine_sigs = signals[signals["Strategy"].str.contains(
                "RTS|Structure|Combined", na=False)]
            if not pine_sigs.empty:
                print(f"\n  PINE STRATEGY SIGNALS:")
                for _, r in pine_sigs.iterrows():
                    print(f"    {r['Signal']} {r['Symbol']} "
                          f"@ ₹{r['Price']:,.2f} ({r['Strategy']})")

        return signals

    signals_df = safe("Daily Signal Scan + Consensus", run_scan)

    # ════════════════════════════════════════════════════════
    # SECTION 6: MARKET BREADTH + INTELLIGENCE
    # ════════════════════════════════════════════════════════
    section("SECTION 6: MARKET INTELLIGENCE ENGINE")

    def run_intelligence():
        # Fetch data for breadth
        breadth_stocks = EXPANDED_UNIVERSE[:20]
        stock_data = {}
        for sym in breadth_stocks:
            df = fetch_with_retry(fetcher, sym, period="6mo")
            if not df.empty:
                stock_data[sym] = df

        # Run full intelligence
        mi = MarketIntelligence()
        results = mi.analyze(
            regime=regime_str,
            vix_value=vix_value,
            stock_data=stock_data,
            signals_df=signals_df if signals_df is not None else pd.DataFrame(),
            verdicts=all_verdicts,
            strategies=all_strategies,
        )

        return results

    intel_result = safe("Market Intelligence Engine", run_intelligence)

    # ════════════════════════════════════════════════════════
    # SECTION 7: PAPER TRADING (Regime-Adaptive)
    # ════════════════════════════════════════════════════════
    section("SECTION 7: PAPER TRADING (Regime-Adaptive)")

    def run_paper():
        paper_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
                        "INFY.NS", "ICICIBANK.NS"]
        all_data = {}
        for sym in paper_stocks:
            df = fetch_with_retry(fetcher, sym, period="1y")
            if not df.empty:
                all_data[sym] = df

        if not all_data:
            print("  No data")
            return None

        # Use regime-adaptive position sizing
        pos_size = risk_profile.position_pct if risk_profile else 0.10
        # NOTE: validated live config (10y real PaperTrader) uses
        # CONSENSUS_MIN_AGREE=2 + ALLOW_SHORT=False -> +54.9%/10y, PF 1.27.
        # This section runs a single-strategy demo; the full consensus path
        # lives in live_run.py with the same winning params as defaults.
        trader = PaperTrader(initial_capital=PAPER_CAPITAL)
        if PAPER_SIZING in ("volatility", "kelly"):
            trader.sizing_method = PAPER_SIZING
        if not PAPER_CB:
            trader.cb_daily_limit = trader.cb_weekly_limit = trader.cb_monthly_limit = 0.0
            trader.cb_losing_streak = 10**9

        # Pick top 2 Pine strategies for simulation
        pine_strats = get_pine_strategies()
        for strategy in pine_strats[:2]:
            print(f"\n  Simulating: {strategy.name} "
                  f"(position: {pos_size * 100:.0f}%)")

            signals_dict = {}
            for symbol, data in all_data.items():
                signals_dict[symbol] = strategy.generate_signals(data)

            all_dates = sorted(set(
                d for df in signals_dict.values() for d in df.index
            ))

            for date in all_dates:
                date_str = str(date.date())
                for symbol, df in signals_dict.items():
                    if date not in df.index:
                        continue
                    sval = int(df.loc[date, "Signal"])
                    # Respect ALLOW_SHORT: skip short entries in the validated config
                    if not ALLOW_SHORT and sval in (-2,):
                        continue
                    action = trader.process_signal(
                        symbol=symbol, signal=sval,
                        price=df.loc[date, "Close"], date=date_str,
                        strategy_name=strategy.name
                    )
                trader.take_snapshot(date_str)

            summary = trader.get_performance_summary()
            print(f"\n  {strategy.name} PERFORMANCE:")
            for k, v in summary.items():
                if k not in ["positions"]:
                    print(f"    {k}: {v}")

        trader.print_trade_log()
        return True

    safe("Paper Trading (Regime-Adaptive)", run_paper)

    # ════════════════════════════════════════════════════════
    # SECTION 8: CHARTS
    # ════════════════════════════════════════════════════════
    section("SECTION 8: CHARTS")

    def run_charts():
        viz = Visualizer()
        data = fetch_with_retry(fetcher, "RELIANCE.NS", period="2y")
        if not data.empty:
            from strategies import SMACrossover
            bt = Backtester(initial_capital=500000)
            result = bt.run_backtest(SMACrossover(20, 50), data,
                                     "RELIANCE.NS")
            if not result["equity_curve"].empty:
                viz.plot_equity_curve(result["equity_curve"],
                                      "RELIANCE — SMA 20/50",
                                      "reliance_equity.png")
        print(f"  Charts saved to: ./charts/")
        for f in os.listdir("charts"):
            print(f"    📊 {f}")

    safe("Generate Charts", run_charts)

    # ════════════════════════════════════════════════════════
    # SECTION 9: FINAL SUMMARY (Intelligence-Enhanced)
    # ════════════════════════════════════════════════════════
    section("SECTION 9: FINAL SUMMARY (Intelligence-Enhanced)")

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"  Duration:         {duration}")
    print(f"  Stocks Analyzed:  {len(all_verdicts)}")
    print(f"  Strategies:       {len(all_strategies)} "
          f"(including {len(get_pine_strategies())} Pine)")
    print(f"  Market Regime:    {regime_str}")
    print(f"  India VIX:        {vix_value:.1f}")
    print(f"  Position Size:    {risk_profile.position_pct * 100:.0f}%")
    print(f"  Risk Mode:        {risk_profile.risk_label}")

    if all_verdicts:
        sorted_v = sorted(all_verdicts,
                          key=lambda v: v.conviction_score, reverse=True)

        print(f"\n  ╔{'═' * 58}╗")
        print(f"  ║{'FINAL STOCK RANKINGS':^58}║")
        print(f"  ╠{'═' * 58}╣")
        for i, v in enumerate(sorted_v, 1):
            icon = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡",
                    "AVOID": "🔴", "STRONG_AVOID": "🔴🔴"}.get(
                v.action.value, "")
            line = (f"  {i:>2}. {icon} {v.symbol:<16} "
                    f"{v.conviction_score}/10  ₹{v.current_price:,.0f}")
            print(f"  ║{line:<58}║")
        print(f"  ╚{'═' * 58}╝")

    # Intelligent recommendation summary
    if intel_result and "recommendation" in intel_result:
        rec = intel_result["recommendation"]
        print(f"\n  NEXT STEPS:")
        print(f"  1. ACTION: {rec['action']}")
        print(f"  2. Position Size: {rec['position_size'] * 100:.0f}%")
        for reason in rec.get("reasons", []):
            print(f"     • {reason}")
    else:
        print(f"\n  NEXT STEPS:")
        print(f"  1. Share this report for detailed analysis")
        print(f"  2. Set up Telegram for daily alerts")
        print(f"  3. Run daily at 3:15 PM IST")

    print(f"\n{'=' * 70}")
    print(f"  REPORT COMPLETE — {duration}")
    print(f"  File: {report_file}")
    print(f"{'=' * 70}")

    sys.stdout = tee.terminal
    tee.close()
    print(f"\n✅ Report saved: {report_file}")


if __name__ == "__main__":
    main()