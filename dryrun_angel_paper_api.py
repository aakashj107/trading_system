"""
dryrun_angel_paper_api.py — READ-ONLY connectivity check against Angel One.

WHAT IT DOES (and does NOT do):
  - Authenticates via SmartAPI (generateSession + getProfile) using credentials
    from env vars: ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PASSWORD, ANGEL_TOTP_SECRET
  - Fetches live LTP for a small watchlist (ltpData — READ ONLY)
  - NEVER places an order. This is a connectivity / smoke test only.

SAFETY:
  - Uses LiveTrader(mode="paper_api") -> place_order is a read-only DRYRUN.
  - If LiveTrader(mode="live") were used by mistake, it still refuses without
    live_confirmed=True. This script never sets that.

USAGE:
  export ANGEL_API_KEY=xxx
  export ANGEL_CLIENT_CODE=xxxxx
  export ANGEL_PASSWORD=xxxxx
  export ANGEL_TOTP_SECRET=xxxx
  python3 dryrun_angel_paper_api.py
"""
import os
import sys

from angel_one_live import LiveTrader

# Load .env file if present (so keys work in any process, not just an
# interactive shell that exported them). Does NOT override real env vars.
def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

_load_dotenv()

# Small read-only watchlist (Angel NSE symbols, no .NS needed but accepted)
WATCHLIST = ["RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "TCS"]


def main():
    missing = [v for v in ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE",
                           "ANGEL_PASSWORD", "ANGEL_TOTP_SECRET")
               if not os.getenv(v)]
    if missing:
        print(f"[EXIT] Missing env vars: {', '.join(missing)}")
        print("Set them, then re-run. Example:")
        print("  export ANGEL_API_KEY=xxx ANGEL_CLIENT_CODE=xxxx \\")
        print("         ANGEL_PASSWORD=xxxx ANGEL_TOTP_SECRET=xxxx")
        sys.exit(2)

    # paper_api = authenticated READ-ONLY. No orders, ever.
    trader = LiveTrader(mode="paper_api")
    print("[DRYRUN] mode=paper_api (read-only, no orders will be placed)\n")

    ok = trader.authenticate()
    print(f"\n[RESULT] authenticate() -> {ok}")
    if not ok:
        print("[RESULT] Connection FAILED. Check credentials / network / TOTP.")
        sys.exit(1)

    print("\n[DRYRUN] Fetching live LTP (read-only)...\n")
    success = 0
    for sym in WATCHLIST:
        ltp = trader.fetch_ltp(sym)
        if ltp is not None:
            success += 1
            print(f"  {sym:<12} LTP = ₹{ltp:,.2f}")
        else:
            print(f"  {sym:<12} LTP = (unavailable)")

    trader.disconnect()
    print(f"\n[RESULT] Live connection OK. {success}/{len(WATCHLIST)} quotes fetched.")
    print("[RESULT] NO ORDERS PLACED — this was a read-only dry-run.")


if __name__ == "__main__":
    main()
