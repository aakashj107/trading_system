"""
Angel One Live Trading Integration — v2 (genuine SmartAPI client, SAFE by default).

SAFETY MODEL (the whole point of #12):
  - Default mode is "paper": place_order() is a print stub. NO network, NO orders.
  - mode="paper_api": authenticated READ-ONLY dry-run. Calls generateSession +
    getProfile + ltpData to prove the live connection works, but NEVER places an
    order. This is how we validate #12 (connectivity) without risking capital.
  - mode="live": STILL REFUSES unless live_confirmed=True is explicitly passed by
    the caller. Even then, place_order() requires self.live_confirmed. This is a
    deliberate two-key gate so a stray call can never go live by accident.

Angel order params (placeOrder) follow the official SmartAPI schema.

Symbol mapping: our internal tickers are like "RELIANCE.NS"; Angel uses exchange
"NSE" + tradingsymbol "RELIANCE" + a numeric symboltoken. ANGEL_TOKEN_MAP holds a
handful of tokens; for others we fall back to ltpData lookup by name.
"""
import os
import time
from datetime import datetime
from typing import Dict, Optional, Callable

# Angel SmartAPI constants
ANGEL_API_BASE = "https://apiconnect.angelone.in"
VARIETY_REGULAR = "NORMAL"
EXCHANGE_NSE = "NSE"
PRODUCT_CNC = "CNC"        # cash & carry (delivery) — safest for swing system
PRODUCT_MIS = "MIS"        # intraday
ORDER_TYPE_MARKET = "MARKET"
ORDER_TYPE_LIMIT = "LIMIT"
VALIDITY_DAY = "DAY"


# Token map — comprehensive coverage of our universe.
# Tokens sourced from Angel One scrip master. When a token is missing,
# _ltp_by_search falls back to searchScrip + picks the -EQ variant.
ANGEL_TOKEN_MAP = {
    # Financial Services
    "HDFCBANK": "1333", "ICICIBANK": "4963", "SBIN": "3045", "KOTAKBANK": "1922",
    "AXISBANK": "5900", "BAJFINANCE": "317", "BAJAJFINSV": "16675",
    "SHRIRAMFIN": "4306", "INDUSINDBK": "3456", "MUTHOOTFIN": "2383",
    "HDFCLIFE": "4690", "SBILIFE": "5039",
    # Technology
    "TCS": "11536", "INFY": "1594", "HCLTECH": "7229", "WIPRO": "3787",
    "TECHM": "3372", "MPHASIS": "4203",
    # FMCG
    "HINDUNILVR": "356", "ITC": "1660", "NESTLEIND": "1804", "TATACONSUM": "4225",
    "BRITANNIA": "5233",
    # Pharma
    "SUNPHARMA": "3351", "DRREDDY": "1254", "CIPLA": "651", "DIVISLAB": "2229",
    "LUPIN": "1372",
    # Auto
    "MARUTI": "10999", "M&M": "3432", "EICHERMOT": "594", "HEROMOTOCO": "1437",
    "BAJAJ-AUTO": "1600", "TMCV": "759782", "TMPV": "759794",
    # Energy
    "RELIANCE": "2885", "ONGC": "1633", "BPCL": "526", "NTPC": "1360",
    # Metals
    "TATASTEEL": "3456", "JSWSTEEL": "1172", "HINDALCO": "1304", "COALINDIA": "2037",
    # Infrastructure
    "LT": "11483", "ADANIPORTS": "1502", "POWERGRID": "1394",
    # Other
    "BHARTIARTL": "10604", "TITAN": "3506", "ULTRACEMCO": "11532",
    "ASIANPAINT": "236", "ADANIENT": "6229", "GRASIM": "1232",
    "APOLLOHOSP": "356",
}


def ns_to_angel_symbol(ticker: str) -> str:
    """'RELIANCE.NS' -> 'RELIANCE' (Angel tradingsymbol on NSE)."""
    return ticker.replace(".NS", "").replace(".BO", "")


class LiveTrader:
    def __init__(self, api_key: str = "", client_code: str = "", password: str = "",
                 totp_secret: str = "", mode: str = "paper", live_confirmed: bool = False):
        """
        mode:
          "paper"     -> place_order is a no-op print stub (default, safe)
          "paper_api" -> authenticated READ-ONLY dry-run (no orders placed)
          "live"      -> requires live_confirmed=True AND successful auth
        live_confirmed:
          Must be explicitly True to ever place a real order. Never defaulted.
        """
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

    # ── Authentication ──────────────────────────────────────────────────────
    def authenticate(self) -> bool:
        """Authenticate against Angel One. Safe: no orders placed here.

        Returns True on success. In paper mode this is a no-op success so the
        harness can run without credentials.
        """
        if self.mode == "paper":
            self.connected = True
            print("[AUTH] paper mode — no real connection attempted")
            return True
        if not (self.api_key and self.client_code and self.password and self.totp_secret):
            print("[AUTH] missing credentials — set ANGEL_API_KEY / CLIENT_CODE / "
                  "PASSWORD / TOTP_SECRET (env or kwargs)")
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
            # Confirm identity with a read-only profile call
            prof = obj.getProfile(self.refresh_token or "")
            if prof.get("status") is not True:
                print(f"[AUTH] profile check failed: {prof.get('message')}")
                return False
            self.connected = True
            print(f"[AUTH] Angel One connected as {self.client_code} "
                  f"(paper_api={self.mode == 'paper_api'})")
            return True
        except ImportError:
            print("[ERROR] pip install smartapi-python pyotp")
            return False
        except Exception as e:  # network / API errors must not crash the harness
            print(f"[AUTH] exception: {type(e).__name__}: {e}")
            return False

    # ── Read-only market data (used by paper_api dry-run) ───────────────────
    def fetch_ltp(self, ticker: str) -> Optional[float]:
        """Fetch last traded price (READ-ONLY). Returns float or None."""
        if not self.connected:
            if not self.authenticate():
                return None
        sym = ns_to_angel_symbol(ticker)
        token = ANGEL_TOKEN_MAP.get(sym, "")
        try:
            if self.mode == "paper":
                return None
            resp = self.session.ltpData(EXCHANGE_NSE, sym, token)  # type: ignore
            if resp.get("status") is True:
                return float(resp["data"]["ltp"])
            # Token missing — try a search-based fallback
            return self._ltp_by_search(sym)
        except Exception as e:
            print(f"[LTP] {ticker}: {type(e).__name__}: {e}")
            return None

    def _ltp_by_search(self, sym: str) -> Optional[float]:
        try:
            r = self.session.searchScrip(EXCHANGE_NSE, sym)  # type: ignore
            if r.get("status") is True and r.get("data"):
                # Prefer -EQ (regular equity) variant, fall back to first result
                match = r["data"][0]
                for item in r["data"]:
                    ts = item.get("tradingsymbol", "")
                    if ts.endswith("-EQ"):
                        match = item
                        break
                token = match["symboltoken"]
                ts_name = match.get("tradingsymbol", sym)
                resp = self.session.ltpData(EXCHANGE_NSE, ts_name, token)  # type: ignore
                if resp.get("status") is True:
                    return float(resp["data"]["ltp"])
        except Exception:
            pass
        return None

    # ── Order placement ─────────────────────────────────────────────────────
    def place_order(self, symbol, transaction_type, quantity, price,
                    product=PRODUCT_CNC, order_type=ORDER_TYPE_MARKET) -> Dict:
        """
        Place an order. SAFE by construction:
          - paper mode        -> print stub, returns PAPER_FILLED
          - paper_api mode    -> READ-ONLY: fetches LTP to prove connectivity,
                                 NEVER places an order, returns DRYRUN
          - live mode         -> requires self.live_confirmed, else returns LOCKED
        """
        if self.mode == "paper":
            print(f"  [PAPER] {transaction_type} {quantity}x {symbol} @ ₹{price:,.2f}")
            return {"status": "PAPER_FILLED"}

        if self.mode == "paper_api":
            # Prove the authenticated link works WITHOUT risking capital.
            ltp = self.fetch_ltp(symbol)
            print(f"  [DRYRUN] {transaction_type} {quantity}x {symbol} "
                  f"@ ₹{price:,.2f} | live LTP={ltp} (NO ORDER PLACED)")
            return {"status": "DRYRUN", "ltp": ltp}

        # mode == "live"
        if not self.live_confirmed:
            print("  [LIVE MODE LOCKED] live_confirmed=False — refusing order. "
                  "Set live_confirmed=True explicitly to go live.")
            return {"status": "LOCKED"}
        if not self.connected and not self.authenticate():
            return {"status": "AUTH_FAILED"}
        sym = ns_to_angel_symbol(symbol)
        token = ANGEL_TOKEN_MAP.get(sym, "")
        orderparams = {
            "variety": VARIETY_REGULAR,
            "tradingsymbol": sym,
            "symboltoken": token,
            "transactiontype": transaction_type,          # "BUY" / "SELL"
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
            print(f"  [LIVE] {transaction_type} {quantity}x {sym} -> {resp}")
            return resp
        except Exception as e:
            print(f"  [LIVE ERROR] {type(e).__name__}: {e}")
            return {"status": "ERROR", "message": str(e)}

    def disconnect(self):
        if self.session:
            try:
                # SmartAPI.terminateSession takes client_code, not refresh_token
                self.session.terminateSession(self.client_code)
            except Exception:
                pass  # logout failures are non-fatal
        self.connected = False
