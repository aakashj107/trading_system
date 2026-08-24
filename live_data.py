"""
Live Data Bridge — Fetches live LTP from Angel One and merges with Yahoo historical data.

Purpose:
    The scanner needs full OHLCV bars (SMA, RSI, MACD etc. require history).
    This module fetches Yahoo historical data and injects the live LTP into
    the latest bar so strategies see the most current price.

Usage:
    bridge = LiveDataBridge(mode="paper_api")
    cache = bridge.build_live_cache(symbols, period="6mo")
    # cache[symbol] = DataFrame with live LTP in the latest bar's Close
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

from angel_one_live import LiveTrader, ns_to_angel_symbol
from data_fetcher import YFinanceDataFetcher
from config import STOCK_UNIVERSE


IST = pytz.timezone("Asia/Kolkata")


class LiveDataBridge:
    """
    Bridges historical Yahoo data with live Angel One LTP.

    Flow:
        1. Fetch 6mo historical OHLCV from Yahoo Finance (cached)
        2. Authenticate with Angel One (paper_api mode)
        3. Fetch live LTP for each symbol
        4. Inject LTP into the latest bar's Close/High/Low
        5. Return enriched data_cache ready for the scanner
    """

    def __init__(self, mode: str = "paper_api"):
        """
        Args:
            mode: "paper" (no network), "paper_api" (auth + read-only), "live" (real orders)
        """
        self.trader = LiveTrader(mode=mode)
        self.fetcher = YFinanceDataFetcher()
        self._authenticated = False
        self._ltp_cache: Dict[str, float] = {}
        self._ltp_timestamp: Optional[datetime] = None

    def authenticate(self) -> bool:
        """Authenticate with Angel One. Safe — no orders placed."""
        if self._authenticated:
            return True
        self._authenticated = self.trader.authenticate()
        return self._authenticated

    def fetch_live_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch live LTP for a list of symbols from Angel One.

        Includes a 0.3s delay between requests to avoid rate limits.

        Returns:
            {symbol: ltp} — only includes symbols where LTP was fetched.
        """
        if not self.authenticate():
            print("[LIVE_DATA] Auth failed — cannot fetch live prices")
            return {}

        prices = {}
        for i, sym in enumerate(symbols):
            try:
                ltp = self.trader.fetch_ltp(sym)
                if ltp is not None and ltp > 0:
                    prices[sym] = ltp
            except Exception as e:
                print(f"[LIVE_DATA] LTP fetch failed for {sym}: {e}")

            # Rate-limit: pause between requests (Angel One allows ~10 req/s)
            if i < len(symbols) - 1:
                time.sleep(0.3)

        self._ltp_cache = prices
        self._ltp_timestamp = datetime.now(IST)
        print(f"[LIVE_DATA] Fetched live LTP for {len(prices)}/{len(symbols)} symbols")
        return prices

    def _get_yahoo_period(self, days: int) -> str:
        """Convert days to Yahoo period string."""
        if days <= 30:
            return "1mo"
        elif days <= 90:
            return "3mo"
        elif days <= 180:
            return "6mo"
        elif days <= 365:
            return "1y"
        else:
            return "2y"

    def build_live_cache(
        self,
        symbols: List[str],
        period: str = "6mo",
        inject_ltp: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Build a data cache with historical OHLCV + live LTP injection.

        Args:
            symbols: Stock symbols to fetch
            period: Yahoo Finance period (e.g. "6mo", "1y")
            inject_ltp: If True, replace latest bar's Close with live LTP

        Returns:
            {symbol: DataFrame} with OHLCV columns
        """
        cache = {}

        # 1. Fetch historical data from Yahoo (parallel via ThreadPool)
        print(f"[LIVE_DATA] Fetching {period} historical data for {len(symbols)} symbols...")
        for sym in symbols:
            try:
                df = self.fetcher.get_historical_data(sym, period=period)
                if df is not None and not df.empty and len(df) > 20:
                    cache[sym] = df
            except Exception as e:
                print(f"[LIVE_DATA] History fetch failed for {sym}: {e}")

        print(f"[LIVE_DATA] Got historical data for {len(cache)}/{len(symbols)} symbols")

        # 2. Fetch live LTP and inject into latest bar
        # Only inject during market hours — outside hours, Yahoo data is more reliable
        market_open = self.is_market_open()
        if inject_ltp and market_open:
            live_prices = self.fetch_live_prices(list(cache.keys()))
            injected = 0
            rejected = 0
            for sym, ltp in live_prices.items():
                if sym in cache:
                    df = cache[sym]
                    last_idx = df.index[-1]
                    yahoo_close = float(df.at[last_idx, "Close"])

                    # Sanity check: LTP should be within 15% of Yahoo close
                    if yahoo_close > 0:
                        pct_diff = abs(ltp - yahoo_close) / yahoo_close
                        if pct_diff > 0.15:
                            print(f"[LIVE_DATA] REJECTED {sym}: LTP ₹{ltp} vs Yahoo ₹{yahoo_close:.2f} "
                                  f"({pct_diff*100:.0f}% diff) — using Yahoo price")
                            rejected += 1
                            continue

                    # Update Close with live price
                    df.at[last_idx, "Close"] = ltp

                    # Update High if live price is higher
                    if ltp > df.at[last_idx, "High"]:
                        df.at[last_idx, "High"] = ltp

                    # Update Low if live price is lower
                    if ltp < df.at[last_idx, "Low"]:
                        df.at[last_idx, "Low"] = ltp

                    injected += 1

            print(f"[LIVE_DATA] Injected live LTP into {injected} symbols ({rejected} rejected)")
        elif inject_ltp:
            print(f"[LIVE_DATA] Market closed — using Yahoo prices only")

        return cache

    def get_ltp(self, symbol: str) -> Optional[float]:
        """Get cached LTP for a symbol (from last fetch_live_prices call)."""
        return self._ltp_cache.get(symbol)

    def get_ltp_age_seconds(self) -> Optional[float]:
        """How many seconds since LTP was last fetched."""
        if self._ltp_timestamp is None:
            return None
        return (datetime.now(IST) - self._ltp_timestamp).total_seconds()

    def is_market_open(self) -> bool:
        """Check if Indian stock market is currently open (9:15 AM - 3:30 PM IST, Mon-Fri)."""
        now = datetime.now(IST)
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close

    def get_market_status(self) -> str:
        """Human-readable market status."""
        now = datetime.now(IST)
        if now.weekday() >= 5:
            return "CLOSED (Weekend)"
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        pre_open = now.replace(hour=9, minute=0, second=0, microsecond=0)

        if now < pre_open:
            return "PRE-MARKET"
        elif now < market_open:
            return "PRE-OPEN (9:00-9:15)"
        elif now <= market_close:
            mins_to_close = (market_close - now).total_seconds() / 60
            return f"OPEN ({mins_to_close:.0f} min to close)"
        else:
            return "CLOSED"

    def disconnect(self):
        """Clean up Angel One session."""
        self.trader.disconnect()
        self._authenticated = False
