"""
Indian Market Trading System v3 — Main Runner
v3.1 — Added ranking with red flag penalization
"""

from data_fetcher import YFinanceDataFetcher
from strategies import get_all_strategies, SMACrossover, RSIStrategy
from backtester import Backtester
from paper_trader import PaperTrader
from scanner import SignalScanner
from visualizer import Visualizer
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager
from regime_detector import RegimeDetector
from database import Database
from alerts import AlertSystem
from fundamentals import FundamentalAnalyzer
from management_analyzer import ManagementAnalyzer
from valuation import ValuationAnalyzer
from verdict import VerdictEngine
from config import QUICK_TEST_STOCKS, NIFTY_50_STOCKS

import warnings
warnings.filterwarnings("ignore")


# ============================================================
#  NEW: RANKING ENGINE WITH RED FLAG PENALIZATION
# ============================================================

# Red flag severity penalties
RED_FLAG_PENALTIES = {
    "Negative OCF":              -0.8,   # Cash flow problems
    "NI not backed by OCF":      -0.6,   # Earnings quality concern
    "Dangerous leverage":        -1.0,   # High debt risk
    "Poor liquidity":            -0.5,   # Liquidity risk
    "High leverage":             -0.8,   # Elevated debt
    "Promoter pledge":           -0.7,   # Promoter risk
    "Governance concern":        -1.2,   # Management risk
}

# Verdict penalties
VERDICT_PENALTIES = {
    "STRONG AVOID":  -2.0,
    "AVOID":         -1.0,
    "HOLD":          0.0,
    "BUY":           0.5,
    "STRONG BUY":    1.0,
}

# Valuation bonuses
VALUATION_BONUS = {
    "UNDERVALUED":         1.0,
    "FAIRLY_VALUED":       0.3,
    "SLIGHTLY_EXPENSIVE": -0.3,
    "EXPENSIVE":          -0.7,
    "EXTREMELY_EXPENSIVE": -1.5,
}


def rank_stock(verdict):
    """
    Calculate a fair ranking score that penalizes red flags.

    Returns:
        dict with original_score, penalties, adjusted_score, rank_rationale
    """
    raw_score = verdict.get("conviction_score", 5.0)
    red_flags = verdict.get("red_flags", [])
    bear_flags = verdict.get("bear_flags", [])
    bull_points = verdict.get("bull_points", [])
    verdict_text = verdict.get("verdict", "HOLD")

    penalties = []
    total_penalty = 0.0

    # 1. Penalize red flags
    for flag in red_flags + bear_flags:
        flag_text = flag if isinstance(flag, str) else str(flag)
        for key, penalty in RED_FLAG_PENALTIES.items():
            if key.lower() in flag_text.lower():
                penalties.append({"flag": flag_text, "penalty": penalty})
                total_penalty += penalty
                break
        else:
            # Unknown flag — apply small default penalty
            penalties.append({"flag": flag_text, "penalty": -0.3})
            total_penalty -= 0.3

    # 2. Penalize verdict
    verdict_penalty = VERDICT_PENALTIES.get(verdict_text, 0.0)
    if verdict_penalty != 0:
        penalties.append({"flag": f"Verdict: {verdict_text}", "penalty": verdict_penalty})
        total_penalty += verdict_penalty

    # 3. Bonus for valuation
    valuation = verdict.get("valuation_rating", "FAIRLY_VALUED")
    val_bonus = VALUATION_BONUS.get(valuation, 0.0)
    if val_bonus != 0:
        penalties.append({"flag": f"Valuation: {valuation}", "penalty": val_bonus})
        total_penalty += val_bonus

    # 4. Small bonus for bull points (capped)
    bull_bonus = min(len(bull_points) * 0.2, 0.6)
    if bull_bonus > 0:
        penalties.append({"flag": f"Bull points ({len(bull_points)})", "penalty": bull_bonus})
        total_penalty += bull_bonus

    # Calculate adjusted score
    adjusted_score = max(0.0, min(10.0, raw_score + total_penalty))

    # Build rationale
    rationale_parts = []
    if adjusted_score < raw_score:
        rationale_parts.append(f"Penalized {adjusted_score - raw_score:.1f} from {raw_score:.1f}")
    elif adjusted_score > raw_score:
        rationale_parts.append(f"Boosted +{adjusted_score - raw_score:.1f} from {raw_score:.1f}")

    if any(p["penalty"] < -0.5 for p in penalties):
        rationale_parts.append("Significant red flags")

    if verdict_text in ["STRONG AVOID", "AVOID"]:
        rationale_parts.append(f"Verdict: {verdict_text}")

    return {
        "symbol": verdict.get("symbol", ""),
        "original_score": raw_score,
        "penalties": penalties,
        "total_penalty": round(total_penalty, 2),
        "adjusted_score": round(adjusted_score, 2),
        "verdict": verdict_text,
        "rationale": " | ".join(rationale_parts) if rationale_parts else "Clean",
    }


def rank_all_stocks(verdicts):
    """
    Rank all stocks with red flag penalization.

    Args:
        verdicts: List of verdict dicts from VerdictEngine

    Returns:
        Sorted list of ranked stocks
    """
    rankings = []
    for v in verdicts:
        ranking = rank_stock(v)
        rankings.append(ranking)

    # Sort by adjusted score descending
    rankings.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return rankings


def print_rankings(rankings):
    """Print the adjusted rankings table"""
    print(f"\n{'='*90}")
    print(f"  ADJUSTED STOCK RANKINGS (Red Flag Penalized)")
    print(f"{'='*90}")
    print(f"  {'#':<4} {'Symbol':<16} {'Verdict':<14} {'Raw':<6} "
          f"{'Penalty':<9} {'Adjusted':<9} {'Rationale'}")
    print(f"  {'─'*85}")

    for i, r in enumerate(rankings, 1):
        # Visual indicator
        if r["adjusted_score"] >= 6.0:
            icon = "🟢"
        elif r["adjusted_score"] >= 4.5:
            icon = "🟡"
        elif r["adjusted_score"] >= 3.0:
            icon = "🟠"
        else:
            icon = "🔴"

        penalty_str = f"{r['total_penalty']:+.1f}"
        print(f"  {icon} {i:<3} {r['symbol']:<16} {r['verdict']:<14} "
              f"{r['original_score']:<6.1f} {penalty_str:<9} "
              f"{r['adjusted_score']:<9.1f} {r['rationale']}")

        # Show significant penalties
        for p in r["penalties"]:
            if p["penalty"] < -0.3:
                print(f"       └─ {p['flag']}: {p['penalty']:+.1f}")

    print(f"\n{'='*90}")

    # Summary
    avoid = [r for r in rankings if r["verdict"] in ["STRONG AVOID", "AVOID"]]
    buy = [r for r in rankings if r["verdict"] in ["BUY", "STRONG BUY"]]
    print(f"  BUY/STRONG BUY: {len(buy)} | HOLD: {len(rankings) - len(avoid) - len(buy)} | "
          f"AVOID/STRONG AVOID: {len(avoid)}")
    print(f"{'='*90}")


# ============================================================
#  EXISTING FUNCTIONS (unchanged)
# ============================================================

def full_equity_research(symbol):
    engine = VerdictEngine()
    verdict = engine.generate_verdict(symbol)
    engine.print_verdict(verdict)
    return verdict


def run_single_backtest():
    fetcher = YFinanceDataFetcher()
    data = fetcher.get_historical_data("RELIANCE.NS", period="2y")
    if data.empty:
        print("No data.")
        return
    bt = Backtester(initial_capital=500000)
    result = bt.run_backtest(SMACrossover(20, 50), data, "RELIANCE.NS")
    print(f"\n  Strategy: {result['strategy']}")
    for k, v in result["metrics"].items():
        if k != "exit_reasons":
            print(f"    {k}: {v}")


def run_multi_backtest():
    import pandas as pd
    fetcher = YFinanceDataFetcher()
    data_dict = fetcher.get_multiple_stocks(QUICK_TEST_STOCKS[:5], period="2y")
    if not data_dict:
        return
    bt = Backtester(initial_capital=500000, position_size_pct=0.15)
    comparison = []
    for strat in get_all_strategies():
        print(f"  Testing {strat.name}...")
        r = bt.run_multi_stock_backtest(strat, data_dict)
        a = r["aggregate_metrics"]
        if a.get("total_trades", 0) > 0:
            comparison.append({
                "Strategy": strat.name, "Trades": a["total_trades"],
                "Win_Rate%": a["win_rate"], "Return%": a["total_return_pct"],
                "Max_DD%": a["max_drawdown_pct"], "Sharpe": a["sharpe_ratio"],
                "Profit_Factor": a["profit_factor"]
            })
    if comparison:
        df = pd.DataFrame(comparison).sort_values("Sharpe", ascending=False)
        print(f"\n{'='*70}")
        print("STRATEGY COMPARISON")
        print(f"{'='*70}")
        print(df.to_string(index=False))


def run_signal_scan(regime="BEAR"):
    """Uses regime-aware strategies including bear shorts"""
    from strategies import get_regime_strategies

    # Get strategies appropriate for current regime
    strategies = get_regime_strategies(regime)
    print(f"  Loaded {len(strategies)} strategies for {regime} regime")

    scanner = SignalScanner(
        symbols=QUICK_TEST_STOCKS,
        strategies=strategies,  # Pass regime-aware strategies
        regime=regime
    )
    signals = scanner.scan()
    scanner.print_signals(signals)
    consensus = scanner.scan_for_consensus()
    if not consensus.empty:
        resolved = scanner.resolve_conflicts(consensus)
        scanner.print_consensus(resolved if not resolved.empty else consensus)

def run_regime():
    fetcher = YFinanceDataFetcher()
    nifty = fetcher.get_nifty_index(period="1y")
    RegimeDetector().print_regime_report(nifty)


def run_fundamentals(symbol):
    FundamentalAnalyzer().print_report(FundamentalAnalyzer().analyze(symbol))


def run_valuation(symbol):
    ValuationAnalyzer().print_report(ValuationAnalyzer().analyze(symbol))


def run_management(symbol):
    ManagementAnalyzer().print_report(ManagementAnalyzer().analyze(symbol))


def research_multiple():
    engine = VerdictEngine()
    for sym in QUICK_TEST_STOCKS[:5]:
        try:
            v = engine.generate_verdict(sym)
            engine.print_verdict(v)
        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")


def research_multiple_ranked():
    """NEW: Research multiple stocks AND rank them with penalty"""
    engine = VerdictEngine()
    verdicts = []
    for sym in QUICK_TEST_STOCKS[:5]:
        try:
            v = engine.generate_verdict(sym)
            engine.print_verdict(v)
            verdicts.append(v)
        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")

    if verdicts:
        rankings = rank_all_stocks(verdicts)
        print_rankings(rankings)


if __name__ == "__main__":
    print(f"""
{'█'*60}
  INDIAN MARKET TRADING SYSTEM v3.1
  20 Modules | Equity Research + Trading + Rankings
{'█'*60}

  EQUITY RESEARCH:
    1. Full Research (1 stock, all 6 dimensions)
    2. Multi-Stock Comparison (5 stocks)
    3. Multi-Stock Comparison + RANKINGS (with red flag penalty)
    4. Fundamentals Only
    5. Valuation Only
    6. Management DNA Only

  TRADING:
    7. Single Stock Backtest
    8. Multi-Stock Strategy Comparison
    9. Daily Signal Scan
    10. Market Regime Detection
    """)

    choice = input("Enter 1-10: ").strip()

    if choice == "1":
        sym = input("Symbol (e.g. RELIANCE.NS): ").strip()
        if not sym.endswith(".NS"):
            sym += ".NS"
        full_equity_research(sym)
    elif choice == "2":
        research_multiple()
    elif choice == "3":
        research_multiple_ranked()  # NEW
    elif choice == "4":
        sym = input("Symbol: ").strip()
        if not sym.endswith(".NS"):
            sym += ".NS"
        run_fundamentals(sym)
    elif choice == "5":
        sym = input("Symbol: ").strip()
        if not sym.endswith(".NS"):
            sym += ".NS"
        run_valuation(sym)
    elif choice == "6":
        sym = input("Symbol: ").strip()
        if not sym.endswith(".NS"):
            sym += ".NS"
        run_management(sym)
    elif choice == "7":
        run_single_backtest()
    elif choice == "8":
        run_multi_backtest()
    elif choice == "9":
        run_signal_scan()
    elif choice == "10":
        run_regime()
    else:
        full_equity_research("RELIANCE.NS")