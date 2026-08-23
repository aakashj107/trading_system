"""
Live Runner — Single scan cycle: fetch → scan → consensus → execute → alert.

Called by live_scheduler.py every 15 minutes during market hours.
Can also be run standalone for a one-shot scan.

Usage:
    python live_runner.py              # one-shot scan in paper_api mode
    python live_runner.py --live       # one-shot scan in live mode (dangerous)
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

import pytz

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

from live_data import LiveDataBridge
from scanner import SignalScanner
from market_intelligence import ConsensusFilter
from portfolio_manager import PortfolioManager
from order_manager import OrderManager, OrderSide, OrderStatus
from angel_one_live import LiveTrader
from alerts import AlertSystem
from config import (
    get_active_stocks,
    get_core_stocks,
    get_all_stocks,
    STOCK_UNIVERSE,
    RiskConfig,
)

IST = pytz.timezone("Asia/Kolkata")

# Logging setup
LOG_DIR = os.path.join(os.path.dirname(__file__), "daily_logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("live_runner")


# ============================================================
#  STATE TRACKER — persists across cycles within a session
# ============================================================

class CycleState:
    """Tracks state across scan cycles within a single session."""

    def __init__(self):
        self.cycle_count = 0
        self.last_scan_time: Optional[datetime] = None
        self.last_signals: List[Dict] = []
        self.total_buy_orders = 0
        self.total_sell_orders = 0
        self.total_rejected = 0
        self.symbols_scanned = 0

    def update(self, signals_df: pd.DataFrame, buy: int, sell: int, rejected: int):
        self.cycle_count += 1
        self.last_scan_time = datetime.now(IST)
        self.total_buy_orders += buy
        self.total_sell_orders += sell
        self.total_rejected += rejected
        if not signals_df.empty:
            self.symbols_scanned = signals_df["Symbol"].nunique()
            # Store latest consensus signals (not individual strategy rows)
            self.last_signals = []

    def summary(self) -> str:
        return (
            f"Cycle #{self.cycle_count} | "
            f"Scanned: {self.symbols_scanned} symbols | "
            f"Buys: {self.total_buy_orders} | "
            f"Sells: {self.total_sell_orders} | "
            f"Rejected: {self.total_rejected}"
        )


# ============================================================
#  CORE RUN CYCLE
# ============================================================

def run_cycle(
    bridge: LiveDataBridge,
    state: CycleState,
    regime: str = "BULL",
    min_agree: int = 3,
    capital: float = 500_000.0,
    live_mode: str = "paper_api",
    universe: str = "core",
    alert_system: Optional[AlertSystem] = None,
    portfolio: Optional[PortfolioManager] = None,
    order_manager: Optional[OrderManager] = None,
) -> Dict:
    """
    Execute one complete scan cycle.

    1. Build live data cache (historical + live LTP)
    2. Run SignalScanner on all stocks
    3. Extract consensus signals (min_agree)
    4. Execute through OrderManager → LiveTrader
    5. Send Telegram alerts
    6. Log everything

    Returns:
        Dict with cycle results for logging/alerting.
    """
    cycle_start = datetime.now(IST)
    logger.info(f"=== Cycle #{state.cycle_count + 1} starting at {cycle_start.strftime('%H:%M:%S')} ===")

    # ── 1. Build live data cache ──────────────────────────────
    if universe == "core":
        symbols = get_core_stocks()
    elif universe == "active":
        symbols = get_active_stocks()
    elif universe == "all":
        symbols = get_all_stocks()
    else:
        symbols = get_core_stocks()

    cache = bridge.build_live_cache(symbols, period="6mo", inject_ltp=True)
    if not cache:
        logger.warning("No data available — aborting cycle")
        return {"status": "NO_DATA", "cycle": state.cycle_count + 1}

    # ── 2. Run scanner ────────────────────────────────────────
    scanner = SignalScanner(
        regime=regime,
        period="6mo",
        symbols=list(cache.keys()),
    )
    signals_df = scanner.scan(data_cache=cache)

    if signals_df.empty:
        logger.info("No signals generated")
        state.update(pd.DataFrame(), 0, 0, 0)
        return {"status": "NO_SIGNALS", "cycle": state.cycle_count + 1}

    # ── 3. Extract consensus ──────────────────────────────────
    cf = ConsensusFilter()

    # Build the signals DataFrame for consensus (Symbol, Signal, Strategy)
    consensus_input = signals_df[["Symbol", "Signal", "Strategy"]].copy()
    consensus_list = cf.find_consensus(consensus_input)

    # Filter by min_agree
    consensus = [c for c in consensus_list if c.count >= min_agree]

    if not consensus:
        logger.info(f"Signals exist but no consensus >= {min_agree}")
        state.update(signals_df, 0, 0, 0)
        return {
            "status": "NO_CONSENSUS",
            "cycle": state.cycle_count + 1,
            "raw_signals": len(signals_df),
            "symbols_with_signals": signals_df["Symbol"].nunique(),
        }

    # ── 4. Execute consensus signals ──────────────────────────
    buy_orders = 0
    sell_orders = 0
    rejected = 0
    executed = []

    for c in consensus:
        sym = c.symbol
        signal_type = c.signal_type
        # Get live price from cache or LTP
        if sym in cache and not cache[sym].empty:
            price = float(cache[sym].iloc[-1]["Close"])
        else:
            ltp = bridge.get_ltp(sym)
            if ltp is None:
                rejected += 1
                continue
            price = ltp

        # Execute through OrderManager
        if order_manager is not None:
            order = order_manager.process_signal(
                sym, signal_type, price,
                "consensus_live",
                stop_loss=price * 0.95,
            )
        else:
            order = None

        # Mirror to LiveTrader (paper_api)
        live_trader = bridge.trader
        if order is not None:
            live_trader.place_order(sym, signal_type, order.quantity, price)
            executed.append({
                "symbol": sym,
                "signal": signal_type,
                "price": price,
                "quantity": order.quantity,
                "strength": c.strength,
                "count": c.count,
                "strategies": c.agreeing_strategies[:5],
            })
            if signal_type == "BUY":
                buy_orders += 1
            else:
                sell_orders += 1
        else:
            rejected += 1

    # ── 5. Send alerts ────────────────────────────────────────
    if executed and alert_system is not None:
        alert_system.send_live_trade_alert(executed, regime)

    # ── 6. Log cycle ──────────────────────────────────────────
    cycle_end = datetime.now(IST)
    duration = (cycle_end - cycle_start).total_seconds()

    cycle_result = {
        "status": "OK",
        "cycle": state.cycle_count + 1,
        "timestamp": cycle_start.isoformat(),
        "duration_sec": round(duration, 1),
        "regime": regime,
        "universe": universe,
        "symbols_scanned": len(cache),
        "raw_signals": len(signals_df),
        "symbols_with_signals": signals_df["Symbol"].nunique(),
        "consensus_signals": len(consensus),
        "buy_orders": buy_orders,
        "sell_orders": sell_orders,
        "rejected": rejected,
        "executed": executed,
    }

    state.update(signals_df, buy_orders, sell_orders, rejected)
    _log_cycle(cycle_result)

    logger.info(
        f"Cycle #{state.cycle_count} complete in {duration:.1f}s — "
        f"{len(consensus)} consensus, {buy_orders} buys, {sell_orders} sells"
    )

    return cycle_result


def _log_cycle(result: Dict):
    """Write cycle result to daily log file."""
    today = datetime.now(IST).strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"live_cycles_{today}.jsonl")
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")
    except Exception as e:
        logger.error(f"Failed to write cycle log: {e}")


# ============================================================
#  DAILY SUMMARY
# ============================================================

def send_daily_summary(
    bridge: LiveDataBridge,
    state: CycleState,
    alert_system: Optional[AlertSystem] = None,
):
    """Send end-of-day summary via Telegram."""
    summary = (
        f"*Live Trading Summary*\n"
        f"Date: {datetime.now(IST).strftime('%Y-%m-%d')}\n"
        f"Cycles: {state.cycle_count}\n"
        f"Buy orders: {state.total_buy_orders}\n"
        f"Sell orders: {state.total_sell_orders}\n"
        f"Rejected: {state.total_rejected}\n"
        f"Market status: {bridge.get_market_status()}"
    )
    if alert_system is not None:
        alert_system.send_message(summary)
    print(summary)


# ============================================================
#  STANDALONE MODE
# ============================================================

def main():
    """One-shot scan for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Live Runner — one-shot scan")
    parser.add_argument("--live", action="store_true", help="Live mode (real orders)")
    parser.add_argument("--universe", default="core", choices=["core", "active", "all"])
    parser.add_argument("--regime", default="BULL", choices=["BULL", "BEAR", "SIDEWAYS"])
    parser.add_argument("--min-agree", type=int, default=3)
    parser.add_argument("--capital", type=float, default=500_000.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    mode = "live" if args.live else "paper_api"
    print(f"\n{'='*60}")
    print(f"  LIVE RUNNER — One-shot scan ({mode})")
    print(f"  Universe: {args.universe} | Regime: {args.regime}")
    print(f"  Min agree: {args.min_agree} | Capital: ₹{args.capital:,.0f}")
    print(f"{'='*60}\n")

    bridge = LiveDataBridge(mode=mode)
    if not bridge.authenticate():
        print("[ERROR] Authentication failed — check credentials in .env")
        sys.exit(1)

    state = CycleState()
    portfolio = PortfolioManager(total_capital=args.capital, max_positions=10)
    om = OrderManager(portfolio=portfolio)
    alerts = AlertSystem()

    result = run_cycle(
        bridge=bridge,
        state=state,
        regime=args.regime,
        min_agree=args.min_agree,
        capital=args.capital,
        live_mode=mode,
        universe=args.universe,
        alert_system=alerts,
        portfolio=portfolio,
        order_manager=om,
    )

    print(f"\nCycle result:")
    for k, v in result.items():
        if k == "executed":
            print(f"  {k}:")
            for e in v:
                print(f"    {e['signal']} {e['symbol']} @ ₹{e['price']:,.2f} "
                      f"x {e['quantity']} ({e['strength']}, {e['count']} agree)")
        else:
            print(f"  {k}: {v}")

    bridge.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
