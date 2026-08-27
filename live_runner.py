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
from risk_manager import RiskManager
from angel_one_live import LiveTrader
from alerts import AlertSystem
from trade_logger import (
    log_signal,
    log_trade,
    load_open_positions,
    save_open_positions,
    record_entry,
    pop_entry,
)
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

    def update(self, signals_df: pd.DataFrame, buy: int, sell: int, rejected: int,
               consensus: Optional[List] = None):
        self.cycle_count += 1
        self.last_scan_time = datetime.now(IST)
        self.total_buy_orders += buy
        self.total_sell_orders += sell
        self.total_rejected += rejected
        if not signals_df.empty:
            self.symbols_scanned = signals_df["Symbol"].nunique()
        # Store latest consensus signals (symbol/type/strength) for the summary
        if consensus:
            self.last_signals = [
                {"symbol": c.symbol, "signal": c.signal_type,
                 "strength": c.strength, "count": c.count}
                for c in consensus
            ]

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

def fetch_vix() -> Optional[float]:
    """Fetch current India VIX level via Yahoo Finance."""
    try:
        import yfinance as yf
        df = yf.download("^INDIAVIX", period="5d", progress=False)
        if not df.empty:
            return round(float(df["Close"].iloc[-1].item()), 1)
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")
    return None


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
    risk_manager: Optional[RiskManager] = None,
    max_hold_days: int = 7,
    min_profit_time_exit: float = 0.01,
) -> Dict:
    """
    Execute one complete scan cycle.

    1. Build live data cache (historical + live LTP)
    2. Update VIX sizing; enforce stop-loss / target / time exits
    3. Run SignalScanner on all stocks
    4. Extract consensus signals (min_agree)
    5. Execute through OrderManager (risk-gated) → LiveTrader
    6. Send Telegram alerts, log everything, snapshot cycle health

    Args:
        max_hold_days: exit positions held this many days (time exit)
        min_profit_time_exit: min profit fraction required to avoid time exit

    Returns:
        Dict with cycle results for logging/alerting.
    """
    cycle_start = datetime.now(IST)

    # ── 0a. New trading day → reset daily-loss counter ────────
    if risk_manager is not None:
        today = cycle_start.date()
        if getattr(state, "_last_day", None) != today:
            risk_manager.reset_day()
            state._last_day = today

    # ── 0b. VIX risk gate — update position sizing ────────────
    vix_level = None
    if risk_manager is not None:
        vix_level = fetch_vix()
        if vix_level is not None:
            risk_manager.update_vix(vix_level)
            logger.info(f"India VIX: {vix_level} → {risk_manager.vix_risk_mode} "
                        f"({risk_manager.vix_position_pct*100:.0f}% per trade, "
                        f"max {risk_manager.vix_max_positions} positions)")

    if risk_manager is not None and risk_manager.circuit_breaker_active:
        for a in risk_manager.alerts[-3:]:
            logger.warning(f"CIRCUIT BREAKER: {a}")

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

    # ── 2. Enforce exits on open positions ────────────────────
    # (stop losses, profit targets, and time-based exits)
    exits_executed = []
    if order_manager is not None and portfolio is not None and portfolio.positions:
        prices = {sym: float(df.iloc[-1]["Close"])
                  for sym, df in cache.items() if not df.empty}

        # 2a. Stop loss / profit target
        pending = [(sym, price, reason) for sym, price, reason
                   in order_manager.check_exits(prices)]
        open_state = load_open_positions()

        # 2b. Time-based exit: held too long with negligible profit
        for sym, pos in portfolio.positions.items():
            if any(s[0] == sym for s in pending):
                continue
            cur = prices.get(sym)
            entry_info = open_state.get(sym)
            if cur is None or not entry_info:
                continue
            if "entry_date" not in entry_info or "entry_price" not in entry_info:
                continue
            try:
                hold_days = (cycle_start.date()
                             - datetime.strptime(
                                 str(entry_info["entry_date"]), "%Y-%m-%d").date()).days
            except (ValueError, KeyError):
                continue
            pnl_pct = (cur / float(entry_info["entry_price"]) - 1) * 100
            if hold_days >= max_hold_days and pnl_pct < min_profit_time_exit * 100:
                pending.append((sym, cur, "TIME_EXIT"))
                logger.info(f"TIME EXIT candidate: {sym} held {hold_days}d, "
                            f"P&L {pnl_pct:+.1f}%")

        for sym, price, reason in pending:
            order = order_manager.process_signal(sym, "SELL", price, reason)
            if order is not None:
                bridge.trader.place_order(sym, "SELL", order.quantity, price)
                exits_executed.append({
                    "symbol": sym, "price": price,
                    "quantity": order.quantity, "reason": reason,
                })
                logger.warning(f"{reason} hit: SELL {order.quantity}x {sym} @ ₹{price:,.2f}")

    if alert_system is not None and exits_executed:
        alert_system.send_message(
            "*Position Exits Executed*\n" +
            "\n".join(f"SELL {s['symbol']} @ ₹{s['price']:,.2f} ({s['reason']})"
                      for s in exits_executed)
        )

        # Close out tracked entries for rule-based exits with full P&L
        for s in exits_executed:
            entry_info = pop_entry(s["symbol"])
            if entry_info and "entry_date" in entry_info and "entry_price" in entry_info:
                pnl = (s["price"] - entry_info["entry_price"]) * s["quantity"]
                pnl_pct = ((s["price"] / entry_info["entry_price"]) - 1) * 100
                log_trade(
                    symbol=s["symbol"],
                    action="SELL",
                    entry_date=str(entry_info["entry_date"]),
                    entry_price=entry_info["entry_price"],
                    quantity=s["quantity"],
                    strategy=entry_info.get("strategy", "consensus"),
                    regime=regime,
                    cycle=state.cycle_count + 1,
                    exit_date=str(cycle_start.date()),
                    exit_price=s["price"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=s["reason"],
                )

    # ── 3. Run scanner ────────────────────────────────────────
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

    # ── 4. Extract consensus ──────────────────────────────────
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

        # Execute through OrderManager — risk gating (circuit breaker,
        # correlation limits) is applied inside OrderManager when a
        # risk_manager is attached to it.
        order = (
            order_manager.process_signal(
                sym, signal_type, price,
                "consensus_live",
                stop_loss=price * 0.95,
                target=price * 1.08,
            )
            if order_manager is not None else None
        )

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

    # ── 5b. Log signals and trades to CSV ─────────────────────
    for c in consensus:
        sym = c.symbol
        if sym in cache and not cache[sym].empty:
            price = float(cache[sym].iloc[-1]["Close"])
        else:
            price = bridge.get_ltp(sym) or 0

        # Log every consensus signal
        log_signal(
            symbol=sym,
            signal=c.signal_type,
            strength=c.strength,
            consensus_count=c.count,
            price=price,
            strategies=c.agreeing_strategies[:5],
            regime=regime,
            universe=universe,
            cycle=state.cycle_count + 1,
        )

    # Log executed trades (BUY = open entry row; SELL = close out with P&L)
    for e in executed:
        if e["signal"] == "BUY":
            record_entry(
                symbol=e["symbol"],
                entry_price=e["price"],
                quantity=e["quantity"],
                entry_date=str(cycle_start.date()),
                stop_loss=e["price"] * 0.95,
                target=e["price"] * 1.08,
                strategy="consensus",
            )
            log_trade(
                symbol=e["symbol"],
                action="BUY",
                entry_date=str(cycle_start.date()),
                entry_price=e["price"],
                quantity=e["quantity"],
                strategy="consensus",
                regime=regime,
                cycle=state.cycle_count + 1,
            )
        else:
            entry_info = pop_entry(e["symbol"])
            matched_exit = next((x for x in exits_executed
                                 if x["symbol"] == e["symbol"]), None)
            exit_reason = matched_exit["reason"] if matched_exit else "CONSENSUS_SELL"
            if entry_info and "entry_date" in entry_info and "entry_price" in entry_info:
                pnl = (e["price"] - entry_info["entry_price"]) * e["quantity"]
                pnl_pct = ((e["price"] / entry_info["entry_price"]) - 1) * 100
                try:
                    hold_days = max(0, (
                        datetime.strptime(str(cycle_start.date()), "%Y-%m-%d")
                        - datetime.strptime(str(entry_info["entry_date"]), "%Y-%m-%d")
                    ).days)
                except (ValueError, KeyError):
                    hold_days = 0
                log_trade(
                    symbol=e["symbol"],
                    action="SELL",
                    entry_date=str(entry_info["entry_date"]),
                    entry_price=entry_info["entry_price"],
                    quantity=e["quantity"],
                    strategy=entry_info.get("strategy", "consensus"),
                    regime=regime,
                    cycle=state.cycle_count + 1,
                    exit_date=str(cycle_start.date()),
                    exit_price=e["price"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    hold_days=hold_days,
                    exit_reason=exit_reason,
                )
            else:
                # No tracked entry (e.g. state was lost) — log what we know
                log_trade(
                    symbol=e["symbol"],
                    action="SELL",
                    entry_date="",
                    entry_price=0.0,
                    quantity=e["quantity"],
                    strategy="consensus",
                    regime=regime,
                    cycle=state.cycle_count + 1,
                    exit_date=str(cycle_start.date()),
                    exit_price=e["price"],
                    exit_reason=exit_reason,
                )

    # Persist portfolio positions so restarts/new runs remember holdings
    if portfolio is not None:
        # Build proper state with entry_date/entry_price keys that
        # load_open_positions() expects.  portfolio.positions uses
        # different key names (price/qty) so we must translate.
        state_to_save = {}
        for sym, pos in portfolio.positions.items():
            prev = open_state.get(sym, {})
            state_to_save[sym] = {
                "entry_price": prev.get("entry_price", pos.get("price", 0.0)),
                "entry_date": prev.get("entry_date", str(cycle_start.date())),
                "quantity": prev.get("quantity", pos.get("qty", 0)),
                "stop_loss": prev.get("stop_loss", pos.get("stop_loss", 0.0)),
                "target": prev.get("target", pos.get("target", 0.0)),
                "strategy": prev.get("strategy", "consensus"),
            }
        save_open_positions(state_to_save)

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
        "rule_exits": len(exits_executed),
        "vix": vix_level,
        "risk_mode": risk_manager.vix_risk_mode if risk_manager else "",
        "circuit_breaker": risk_manager.circuit_breaker_active if risk_manager else False,
        "rejected": rejected,
        "executed": executed,
    }

    state.update(signals_df, buy_orders, sell_orders, rejected,
                 consensus=consensus)
    _log_cycle(cycle_result)

    # Cycle-health snapshot for the 7-day evaluation
    try:
        from trade_logger import log_snapshot
        invested = sum(p["price"] * p["qty"]
                       for p in portfolio.positions.values()) if portfolio else 0.0
        log_snapshot(
            cycle=cycle_result["cycle"], regime=regime, vix=vix_level,
            risk_mode=risk_manager.vix_risk_mode if risk_manager else "-",
            symbols_scanned=len(cache), raw_signals=len(signals_df),
            consensus_signals=len(consensus), buy_orders=buy_orders,
            sell_orders=sell_orders, exits=len(exits_executed),
            rejected=rejected, open_positions=len(portfolio.positions) if portfolio else 0,
            invested_value=invested,
            circuit_breaker=bool(risk_manager.circuit_breaker_active) if risk_manager else False,
        )
    except Exception as e:
        logger.warning(f"Snapshot log failed: {e}")

    logger.info(
        f"Cycle #{state.cycle_count} complete in {duration:.1f}s — "
        f"{len(consensus)} consensus, {buy_orders} buys, {sell_orders} sells"
    )

    return cycle_result


def restore_portfolio(portfolio: PortfolioManager):
    """Restore open positions from persistent state (survives restarts and
    fresh GitHub Actions runs)."""
    saved = load_open_positions()
    for sym, info in saved.items():
        if sym not in portfolio.positions:
            portfolio.positions[sym] = {
                "symbol": sym,
                "price": info.get("entry_price", 0.0),
                "qty": info.get("quantity", 0),
                "stop_loss": info.get("stop_loss", 0.0),
                "target": info.get("target", 0.0),
            }
    if saved:
        logger.info(f"Restored {len(saved)} open position(s) from state: "
                    f"{', '.join(saved.keys())}")
    return portfolio


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
    portfolio = restore_portfolio(
        PortfolioManager(total_capital=args.capital, max_positions=10)
    )
    risk = RiskManager(total_capital=args.capital)
    om = OrderManager(portfolio=portfolio, risk_manager=risk)
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
        risk_manager=risk,
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
