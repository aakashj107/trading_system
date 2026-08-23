"""
Indian Market Trading System — Data Fetcher
Supports: yfinance (free), Angel One SmartAPI (live)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import os


class YFinanceDataFetcher:
    """Fetch Indian stock data using yfinance."""

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_historical_data(
        self, symbol: str, period: str = "2y",
        interval: str = "1d", start_date: Optional[str] = None,
        end_date: Optional[str] = None, use_cache: bool = True
    ) -> pd.DataFrame:
        cache_key = f"{symbol}_{period}_{interval}"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.csv")

        if use_cache and os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cache_age = datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(cache_path)
            )
            if cache_age < timedelta(hours=6):
                print(f"  [CACHE] {symbol} loaded from cache")
                return cached

        try:
            ticker = yf.Ticker(symbol)
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"  [WARNING] No data for {symbol}")
                return pd.DataFrame()

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            df = df.dropna()

            if use_cache:
                df.to_csv(cache_path)
            print(f"  [FETCHED] {symbol}: {len(df)} candles")
            return df
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            return pd.DataFrame()

    def get_multiple_stocks(
        self, symbols: List[str], period: str = "2y", interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        data = {}
        total = len(symbols)
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{total}] Fetching {symbol}...")
            df = self.get_historical_data(symbol, period=period, interval=interval)
            if not df.empty:
                data[symbol] = df
        print(f"\nFetched {len(data)}/{total} stocks")
        return data

    def get_stock_info(self, symbol: str) -> Dict:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
            }
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            return {}

    def get_nifty_index(self, period: str = "2y") -> pd.DataFrame:
        return self.get_historical_data("^NSEI", period=period, use_cache=True)


class AngelOneDataFetcher:
    """Angel One SmartAPI integration."""

    def __init__(self, api_key: str, client_code: str, password: str, totp_secret: str):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret
        self.session = None
        self.authenticated = False

    def authenticate(self) -> bool:
        try:
            from SmartApi import SmartConnect
            import pyotp
            obj = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()
            data = obj.generateSession(self.client_code, self.password, totp)
            if data.get("status"):
                self.session = obj
                self.authenticated = True
                print("[AUTH] Angel One connected")
                return True
            print(f"[AUTH FAILED] {data.get('message', '')}")
            return False
        except ImportError:
            print("[ERROR] pip install smartapi-python pyotp")
            return False
        except Exception as e:
            print(f"[AUTH ERROR] {e}")
            return False

    def get_historical_data(self, symbol_token, exchange="NSE",
                            interval="ONE_DAY", from_date="", to_date="") -> pd.DataFrame:
        if not self.authenticated:
            return pd.DataFrame()
        try:
            params = {"exchange": exchange, "symboltoken": symbol_token,
                      "interval": interval, "fromdate": from_date, "todate": to_date}
            data = self.session.getCandleData(params)
            if data.get("status"):
                df = pd.DataFrame(data["data"],
                                  columns=["Datetime","Open","High","Low","Close","Volume"])
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                df.set_index("Datetime", inplace=True)
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"[ERROR] {e}")
            return pd.DataFrame()

    def get_live_quote(self, symbol_token, exchange="NSE") -> Dict:
        if not self.authenticated:
            return {}
        try:
            data = self.session.ltpData(exchange, symbol_token, symbol_token)
            return data.get("data", {})
        except Exception as e:
            print(f"[ERROR] {e}")
            return {}
