"""
Angel One Live Trading Integration — v3 (dynamic token lookup, SAFE by default).

SAFETY MODEL:
  - Default mode is "paper": place_order() is a print stub. NO network, NO orders.
  - mode="paper_api": authenticated READ-ONLY dry-run. NEVER places an order.
  - mode="live": requires live_confirmed=True AND successful auth.

TOKEN RESOLUTION (v3 — the key fix):
  NO hardcoded token map. Every symbol is resolved at runtime via searchScrip:
    1. Check token_cache.json (persisted across sessions)
    2. If missing, call searchScrip("NSE", symbol) → pick -EQ variant
    3. Cache the result to disk
    4. Use cached token + tradingsymbol for ltpData / placeOrder
  This eliminates wrong-price risk from stale hardcoded tokens.
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, Optional

# Angel SmartAPI constants
ANGEL_API_BASE = "https://apiconnect.angelone.in"
VARIETY_REGULAR = "NORMAL"
EXCHANGE_NSE = "NSE"
PRODUCT_CNC = "CNC"
PRODUCT_MIS = "MIS"
ORDER_TYPE_MARKET = "MARKET"
ORDER_TYPE_LIMIT = "LIMIT"
VALIDITY_DAY = "DAY"

# Token cache file — persists resolved tokens across sessions
_TOKEN_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token_cache.json")


def ns_to_angel_symbol(ticker: str) -> str:
    """'RELIANCE.NS' -> 'RELIANCE' (Angel tradingsymbol on NSE)."""
    return ticker.replace(".NS", "").replace(".BO", "")


def _load_token_cache() -> Dict[str, Dict]:
    """Load cached tokens from disk. Returns {symbol: {token, tradingsymbol}}."""
    if os.path.exists(_TOKEN_CACHE_PATH):
        try:
            with open(_TOKEN_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_token_cache(cache: Dict[str, Dict]):
    """Persist token cache to disk."""
    try:
        with open(_TOKEN_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


class LiveTrader:
    def __init__(self, api_key: str = "", client_code: str = "", password: str = "",
                 totp_secret: str = "", mode: str = "paper", live_confirmed: bool = False):
        self.mode = mode
        self.api_key = api_key or os.getenv("ANGEL_API_KEY", "")
        self.client_code = client_code or os.getenv("ANGEL_CLIENT_CODE", "")
        self.password = password or os.getenv("ANGEL_PASSWORD", "")
        self.totp_secret = totp_secret or os.getenv("ANGEL_TOTP_SECRET", "")
        self.live_confirmed = live_confirmed
        self.session: Optional["SmartConnect"] = None  # type: ignore
        self.feed_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.connected = False
        self._token_cache = _load_token_cache()

    # ── Authentication ──────────────────────────────────────────────────────
    def authenticate(self) -> bool:
        if self.mode == "paper":
            self.connected = True
            print("[AUTH] paper mode — no real connection attempted")
            return True
        if not (self.api_key and self.client_code and self.password and self.totp_secret):
            print("[AUTH] missing credentials")
            return False
        try:
            from SmartApi import SmartConnect
            import pyotp
            obj = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()
            data = obj.generateSession(self.client_code, self.password, totp)
            if data.get("status") is not True:
                print(f"[AUTH] Angel rejected session: {data.get('message')}")
                return False
            self.session = obj
            self.refresh_token = data.get("data", {}).get("refreshToken")
            self.feed_token = obj.getfeedToken() if hasattr(obj, "getfeedToken") else None
            prof = obj.getProfile(self.refresh_token or "")
            if prof.get("status") is not True:
                print(f"[AUTH] profile check failed: {prof.get('message')}")
                return False
            self.connected = True
            print(f"[AUTH] Angel One connected as {self.client_code}")
            return True
        except ImportError:
            print("[ERROR] pip install smartapi-python pyotp")
            return False
        except Exception as e:
            print(f"[AUTH] exception: {type(e).__name__}: {e}")
            return False

    # ── Token Resolution (dynamic, with cache) ──────────────────────────────
    def _resolve_token(self, symbol: str) -> Optional[Dict]:
        """
        Resolve Angel One token for a symbol via searchScrip.
        Returns {token, tradingsymbol} or None.
        Always picks the -EQ (regular equity) variant.
        """
        # Check cache first
        if symbol in self._token_cache:
            return self._token_cache[symbol]

        if not self.connected:
            if not self.authenticate():
                return None

        try:
            r = self.session.searchScrip(EXCHANGE_NSE, symbol)  # type: ignore
            if r.get("status") is True and r.get("data"):
                # Prefer -EQ variant
                match = r["data"][0]
                for item in r["data"]:
                    ts = item.get("tradingsymbol", "")
                    if ts.endswith("-EQ"):
                        match = item
                        break

                result = {
                    "token": match["symboltoken"],
                    "tradingsymbol": match.get("tradingsymbol", symbol),
                }

                # Cache to disk
                self._token_cache[symbol] = result
                _save_token_cache(self._token_cache)

                return result
        except Exception as e:
            print(f"[TOKEN] searchScrip failed for {symbol}: {e}")

        return None

    # ── Read-only market data ───────────────────────────────────────────────
    def fetch_ltp(self, ticker: str) -> Optional[float]:
        """Fetch last traded price (READ-ONLY). Returns float or None."""
        if not self.connected:
            if not self.authenticate():
                return None

        sym = ns_to_angel_symbol(ticker)

        if self.mode == "paper":
            return None

        # Resolve token dynamically
        info = self._resolve_token(sym)
        if info is None:
            print(f"[LTP] {ticker}: could not resolve token")
            return None

        try:
            resp = self.session.ltpData(  # type: ignore
                EXCHANGE_NSE, info["tradingsymbol"], info["token"]
            )
            if resp.get("status") is True:
                return float(resp["data"]["ltp"])
            print(f"[LTP] {ticker}: ltpData failed: {resp.get('message')}")
        except Exception as e:
            print(f"[LTP] {ticker}: {type(e).__name__}: {e}")

        return None

    # ── Order placement ─────────────────────────────────────────────────────
    def place_order(self, symbol, transaction_type, quantity, price,
                    product=PRODUCT_CNC, order_type=ORDER_TYPE_MARKET) -> Dict:
        if self.mode == "paper":
            print(f"  [PAPER] {transaction_type} {quantity}x {symbol} @ ₹{price:,.2f}")
            return {"status": "PAPER_FILLED"}

        if self.mode == "paper_api":
            ltp = self.fetch_ltp(symbol)
            print(f"  [DRYRUN] {transaction_type} {quantity}x {symbol} "
                  f"@ ₹{price:,.2f} | live LTP={ltp} (NO ORDER PLACED)")
            return {"status": "DRYRUN", "ltp": ltp}

        # mode == "live"
        if not self.live_confirmed:
            print("  [LIVE MODE LOCKED] live_confirmed=False — refusing order.")
            return {"status": "LOCKED"}
        if not self.connected and not self.authenticate():
            return {"status": "AUTH_FAILED"}

        sym = ns_to_angel_symbol(symbol)
        info = self._resolve_token(sym)
        if info is None:
            return {"status": "ERROR", "message": f"Token not found for {symbol}"}

        orderparams = {
            "variety": VARIETY_REGULAR,
            "tradingsymbol": info["tradingsymbol"],
            "symboltoken": info["token"],
            "transactiontype": transaction_type,
            "exchange": EXCHANGE_NSE,
            "ordertype": order_type,
            "producttype": product,
            "duration": VALIDITY_DAY,
            "quantity": str(quantity),
            "price": str(price) if order_type == ORDER_TYPE_LIMIT else "0",
            "triggerprice": "0",
            "squareoff": "0",
            "stoploss": "0",
            "disclosedquantity": "0",
        }
        try:
            resp = self.session.placeOrder(orderparams)  # type: ignore
            print(f"  [LIVE] {transaction_type} {quantity}x {info['tradingsymbol']} -> {resp}")
            return resp
        except Exception as e:
            print(f"  [LIVE ERROR] {type(e).__name__}: {e}")
            return {"status": "ERROR", "message": str(e)}

    def disconnect(self):
        if self.session:
            try:
                self.session.terminateSession(self.client_code)
            except Exception:
                pass
        self.connected = False
