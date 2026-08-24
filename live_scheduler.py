"""
Live Trading Scheduler — Runs scanner every 15 min during market hours.

Safety:
    Default mode is "paper_api" — fetches live LTP, logs signals,
    but NEVER places real orders. Set --live flag to enable real trading
    (requires live_confirmed=True in angel_one_live.py).

Usage:
    python live_scheduler.py                    # paper_api mode (safe)
    python live_scheduler.py --live             # live mode (real orders!)
    python live_scheduler.py --universe active  # scan all active stocks
    python live_scheduler.py --interval 30      # scan every 30 min

Schedule:
    Market hours: 9:15 AM - 3:30 PM IST, Mon-Fri
    Scan interval: Every 15 minutes (default)
    Daily summary: At 3:30 PM IST (market close)
"""

import os
import sys
import signal
import time
import logging
import schedule
import pytz
from datetime import datetime, timedelta
from typing import Optional

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

from live_data import LiveDataBridge
from live_runner import CycleState, run_cycle, send_daily_summary, restore_portfolio
from portfolio_manager import PortfolioManager
from order_manager import OrderManager
from risk_manager import RiskManager
from alerts import AlertSystem
from config import get_core_stocks, get_active_stocks, get_all_stocks

IST = pytz.timezone("Asia/Kolkata")

# Logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "daily_logs")
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
#  NSE HOLIDAYS (optional)
# ============================================================

def _load_holidays() -> set:
    """
    Load NSE trading holidays from nse_holidays.txt (one YYYY-MM-DD per line,
    next to this script). If the file is absent, only weekends are excluded.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nse_holidays.txt")
    holidays = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    holidays.add(line)
    return holidays


NSE_HOLIDAYS = _load_holidays()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "live_scheduler.log")),
    ],
)
logger = logging.getLogger("live_scheduler")


# ============================================================
#  MARKET HOURS CHECK
# ============================================================

def is_market_hours() -> bool:
    """Check if current time is within Indian market hours (9:15-15:30 IST, Mon-Fri,
    excluding NSE holidays listed in nse_holidays.txt)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    if now.strftime("%Y-%m-%d") in NSE_HOLIDAYS:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def is_market_close() -> bool:
    """Check if it's within the market-close window (15:25-15:40 IST) —
    for daily summary. Uses a window so a delayed loop tick still catches it."""
    now = datetime.now(IST)
    if now.weekday() >= 5 or now.strftime("%Y-%m-%d") in NSE_HOLIDAYS:
        return False
    close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return abs((now - close).total_seconds()) <= 5 * 60


# ============================================================
#  SCHEDULER
# ============================================================

class LiveScheduler:
    """
    Schedules and runs live trading cycles during market hours.

    Lifecycle:
        1. Initialize components (bridge, portfolio, order manager, alerts)
        2. Schedule scan cycles every N minutes
        3. Schedule daily summary at market close
        4. Run until market closes or Ctrl+C
    """

    def __init__(
        self,
        live_mode: str = "paper_api",
        universe: str = "core",
        regime: str = "BULL",
        min_agree: int = 3,
        capital: float = 500_000.0,
        interval_minutes: int = 15,
    ):
        self.live_mode = live_mode
        self.universe = universe
        self.regime = regime
        self.min_agree = min_agree
        self.capital = capital
        self.interval_minutes = interval_minutes

        # Components
        self.bridge = LiveDataBridge(mode=live_mode)
        self.state = CycleState()
        self.portfolio = restore_portfolio(
            PortfolioManager(total_capital=capital, max_positions=10)
        )
        self.risk_manager = RiskManager(total_capital=capital)
        self.order_manager = OrderManager(
            portfolio=self.portfolio, risk_manager=self.risk_manager
        )
        self.alert_system = AlertSystem()

        # Control
        self._running = False
        self._shutdown_requested = False
        self._breaker_alerted = False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle Ctrl+C or kill signal gracefully."""
        logger.info("Shutdown signal received — finishing current cycle...")
        self._shutdown_requested = True

    def _scan_job(self):
        """Single scan cycle — called by scheduler."""
        if self._shutdown_requested:
            return

        if not is_market_hours():
            logger.info("Outside market hours — skipping scan")
            return

        logger.info(f"Starting scan cycle (market: {self.bridge.get_market_status()})")

        try:
            result = run_cycle(
                bridge=self.bridge,
                state=self.state,
                regime=self.regime,
                min_agree=self.min_agree,
                capital=self.capital,
                live_mode=self.live_mode,
                universe=self.universe,
                alert_system=self.alert_system,
                portfolio=self.portfolio,
                order_manager=self.order_manager,
                risk_manager=self.risk_manager,
            )
            logger.info(f"Cycle result: {result['status']}")

            if self.risk_manager.circuit_breaker_active:
                logger.warning(
                    "CIRCUIT BREAKER ACTIVE — new entries blocked. "
                    f"Alerts: {self.risk_manager.alerts[-3:]}"
                )
                if self.alert_system is not None and not getattr(
                        self, "_breaker_alerted", False):
                    self.alert_system.send_message(
                        "*CIRCUIT BREAKER TRIPPED*\n"
                        + "\n".join(self.risk_manager.alerts[-3:])
                        + "\nNew entries blocked until manual review."
                    )
                    self._breaker_alerted = True

        except Exception as e:
            logger.error(f"Cycle failed: {type(e).__name__}: {e}", exc_info=True)

    def _daily_summary_job(self):
        """Send daily summary — called at market close."""
        if self._shutdown_requested:
            return

        logger.info("Sending daily summary...")
        try:
            send_daily_summary(self.bridge, self.state, self.alert_system)
        except Exception as e:
            logger.error(f"Daily summary failed: {e}")

        # Reset state for next day (daily-loss counter resets; the drawdown
        # breaker stays latched until RiskManager.clear_circuit_breaker())
        self.state = CycleState()
        self.risk_manager.reset_day()

    def start(self):
        """Start the scheduler. Blocks until shutdown."""
        self._running = True

        print(f"\n{'='*60}")
        print(f"  LIVE TRADING SCHEDULER")
        print(f"{'='*60}")
        print(f"  Mode:       {self.live_mode}")
        print(f"  Universe:   {self.universe}")
        print(f"  Regime:     {self.regime}")
        print(f"  Min agree:  {self.min_agree}")
        print(f"  Capital:    ₹{self.capital:,.0f}")
        print(f"  Interval:   Every {self.interval_minutes} min")
        print(f"  Market:     {self.bridge.get_market_status()}")
        print(f"{'='*60}\n")

        # Authenticate with Angel One
        if not self.bridge.authenticate():
            logger.error("Angel One authentication failed — check .env credentials")
            print("\n[ERROR] Cannot authenticate with Angel One.")
            print("  1. Check ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PASSWORD, ANGEL_TOTP_SECRET in .env")
            print("  2. Ensure smartapi-python is installed: pip install smartapi-python pyotp")
            sys.exit(1)

        # Send startup alert
        self.alert_system.send_message(
            f"*Live Scheduler Started*\n"
            f"Mode: {self.live_mode}\n"
            f"Universe: {self.universe}\n"
            f"Interval: {self.interval_minutes} min\n"
            f"Capital: ₹{self.capital:,.0f}"
        )

        # Schedule scan cycles
        schedule.every(self.interval_minutes).minutes.do(self._scan_job)

        # Schedule daily summary at 3:30 PM IST
        schedule.every().day.at("15:30").do(self._daily_summary_job)

        # Run first scan immediately if market is open
        if is_market_hours():
            logger.info("Market is open — running initial scan now")
            self._scan_job()

        # Main loop
        logger.info("Scheduler started — press Ctrl+C to stop")
        try:
            while self._running and not self._shutdown_requested:
                schedule.run_pending()
                time.sleep(10)  # Check every 10 seconds
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        # Cleanup
        self._cleanup()

    def _cleanup(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._running = False

        # Send shutdown alert
        summary = self.state.summary()
        self.alert_system.send_message(
            f"*Live Scheduler Stopped*\n{summary}"
        )

        # Disconnect from Angel One
        self.bridge.disconnect()
        logger.info("Disconnected from Angel One")
        logger.info(f"Session summary: {summary}")


# ============================================================
#  CLI ENTRY POINT
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Live Trading Scheduler — runs scanner every N min during market hours",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_scheduler.py                         # paper_api, core stocks, 15 min
  python live_scheduler.py --universe active       # scan all active stocks
  python live_scheduler.py --interval 30           # scan every 30 min
  python live_scheduler.py --regime BEAR --min-agree 2  # bear mode, lower threshold
  python live_scheduler.py --live                  # LIVE mode (real orders!)
        """,
    )
    parser.add_argument("--live", action="store_true",
                        help="Enable live order placement (default: paper_api)")
    parser.add_argument("--universe", default="core",
                        choices=["core", "active", "all"],
                        help="Stock universe to scan (default: core)")
    parser.add_argument("--regime", default="BULL",
                        choices=["BULL", "BEAR", "SIDEWAYS"],
                        help="Market regime (default: BULL)")
    parser.add_argument("--min-agree", type=int, default=3,
                        help="Min strategies for consensus (default: 3)")
    parser.add_argument("--capital", type=float, default=500_000.0,
                        help="Trading capital in INR (default: 500000)")
    parser.add_argument("--interval", type=int, default=15,
                        help="Scan interval in minutes (default: 15)")
    args = parser.parse_args()

    mode = "live" if args.live else "paper_api"

    if args.live:
        print("\n*** WARNING: LIVE MODE ENABLED — REAL ORDERS WILL BE PLACED ***")
        print("Press Ctrl+C within 5 seconds to abort...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    scheduler = LiveScheduler(
        live_mode=mode,
        universe=args.universe,
        regime=args.regime,
        min_agree=args.min_agree,
        capital=args.capital,
        interval_minutes=args.interval,
    )
    scheduler.start()


if __name__ == "__main__":
    main()
