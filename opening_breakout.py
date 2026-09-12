"""
opening_breakout.py — Opening 15-Minute Breakout paper-trading engine

Strategy:
1. Use the first 15 minutes (09:15–09:30 IST) as the opening range.
2. After 09:30, first break above opening HIGH = LONG.
3. First break below opening LOW = SHORT.
4. Stop:
LONG  -> LOW of the 5-minute breakout candle
SHORT -> HIGH of the 5-minute breakout candle
5. Maximum holding period = 3 five-minute candles.
6. No new entries after 14:00 IST.
7. One trade per stock per day.
8. Maximum 3 trades per day.
9. Position size is calculated from risk, not capital deployment.
10. Both directions are tested independently.

IMPORTANT:
This is a paper-testing/research engine.
yfinance intraday history is limited, so this should initially be
used for forward paper testing rather than a 100+ day historical study.
"""

import os
import time
import contextlib
import io
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ============================================================

# CONFIG

# ============================================================

CAPITAL = 100000.0

# 1% risk per trade

RISK_PER_TRADE = 0.01

# Maximum number of trades generated per day

MAX_TRADES_PER_DAY = 3

# Maximum holding period:

# 3 x 5-minute candles = 15 minutes

MAX_HOLD_BARS = 3

# No new entry at/after this time

ENTRY_CUTOFF_HOUR = 14
ENTRY_CUTOFF_MINUTE = 0

# Realistic assumptions

SLIPPAGE_PCT = 0.0005
COST_PCT_RT = 0.0025

# Minimum acceptable stop distance.

# Prevents absurdly tiny stops and huge quantities.

MIN_STOP_PCT = 0.001

TRADES_CSV = "./opening_breakout_trades.csv"

# Existing NIFTY 50 list from your S2 system.

UNIVERSE = [
"RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS",
"INFY.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
"HINDUNILVR.NS", "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS",
"BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
"SUNPHARMA.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
"TATASTEEL.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS",
"NESTLEIND.NS", "ONGC.NS", "COALINDIA.NS", "GRASIM.NS",
"JSWSTEEL.NS", "HINDALCO.NS", "ADANIENT.NS", "ADANIPORTS.NS",
"BAJAJFINSV.NS", "M&M.NS", "CIPLA.NS", "DRREDDY.NS",
"DIVISLAB.NS", "BRITANNIA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
"INDUSINDBK.NS", "SHRIRAMFIN.NS", "TATACONSUM.NS",
"APOLLOHOSP.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BPCL.NS",
"TRENT.NS", "BEL.NS"
]

TRADE_COLS = [
"Trade_ID",
"Date",
"Symbol",
"Direction",

```
"Prev_Close",

"Opening_High",
"Opening_Low",
"Opening_Range",
"Opening_Range_Pct",

"Breakout_Time",
"Breakout_Candle_Open",
"Breakout_Candle_High",
"Breakout_Candle_Low",
"Breakout_Candle_Close",

"Entry_Time",
"Entry_Price",
"SL",

"Qty",
"Risk_Rs",
"Capital_Used",

"Exit_Time",
"Exit_Price",
"Exit_Reason",

"Gross_PnL_Rs",
"Costs_Rs",
"PnL_Rs",
"PnL_pct",
"R_Mult",

"MFE_Rs",
"MAE_Rs",
"MFE_R",
"MAE_R",

"Bars_Held"
```

]

# ============================================================

# TELEGRAM

# ============================================================

def telegram(msg):
tok = os.environ.get("TG_TOKEN")
chat = os.environ.get("TG_CHAT")

```
if not tok or not chat:
    return

try:
    requests.get(
        f"https://api.telegram.org/bot{tok}/sendMessage",
        params={"chat_id": chat, "text": msg},
        timeout=10
    )
except Exception as e:
    print(f"telegram failed: {e}")
```

# ============================================================

# TIME HELPERS

# ============================================================

def now_ist():
return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

def clean_intraday_index(df):
"""
Convert yfinance timestamps to naive IST timestamps.

```
yfinance may return timezone-aware timestamps.
We normalize everything to Asia/Kolkata.
"""
if df is None or df.empty:
    return df

idx = pd.DatetimeIndex(df.index)

if idx.tz is None:
    # yfinance normally supplies timezone-aware timestamps,
    # but keep this defensive path.
    idx = idx.tz_localize("Asia/Kolkata")
else:
    idx = idx.tz_convert("Asia/Kolkata")

df = df.copy()
df.index = idx.tz_localize(None)

return df
```

# ============================================================

# DATA

# ============================================================

FAILED = {}

def fetch_intraday(symbol, retries=3):
"""
Download today's 5-minute candles.

```
We intentionally use a short period because this script is designed
for end-of-day forward paper testing.
"""

df = None

for attempt in range(1, retries + 1):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            df = yf.download(
                symbol,
                period="1d",
                interval="5m",
                auto_adjust=True,
                progress=False,
                prepost=False
            )

        if df is not None and not df.empty:
            break

    except Exception:
        df = None

    time.sleep(1.5 * attempt)

if df is None or df.empty:
    FAILED[symbol] = "no intraday data"
    return None

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

needed = ["Open", "High", "Low", "Close", "Volume"]

missing = [c for c in needed if c not in df.columns]

if missing:
    FAILED[symbol] = f"missing columns: {missing}"
    return None

df = df[needed].dropna()
df = clean_intraday_index(df)

# NSE regular session only.
df = df.between_time("09:15", "15:30")

if df.empty:
    FAILED[symbol] = "no regular-session candles"
    return None

return df
```

def fetch_previous_close(symbol):
"""
Fetch previous daily close for reporting.
"""
try:
with contextlib.redirect_stdout(io.StringIO()):
df = yf.download(
symbol,
period="5d",
interval="1d",
auto_adjust=True,
progress=False
)

```
    if df is None or df.empty:
        return np.nan

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    closes = df["Close"].dropna()

    if len(closes) < 2:
        return np.nan

    return float(closes.iloc[-2])

except Exception:
    return np.nan
```

# ============================================================

# POSITION SIZING

# ============================================================

def calculate_position_size(entry, sl):
"""
Risk-based position sizing.

```
Risk budget = 1% of ₹100,000 = ₹1,000.

Qty is capped by available capital as well.
"""

risk_per_share = abs(entry - sl)

if risk_per_share <= 0:
    return 0, 0.0

if risk_per_share / entry < MIN_STOP_PCT:
    return 0, 0.0

risk_budget = CAPITAL * RISK_PER_TRADE

qty_by_risk = int(risk_budget / risk_per_share)
qty_by_capital = int(CAPITAL / entry)

qty = min(qty_by_risk, qty_by_capital)

if qty <= 0:
    return 0, 0.0

risk_rs = qty * risk_per_share

return qty, risk_rs
```

# ============================================================

# SINGLE-STOCK STRATEGY

# ============================================================

def simulate_stock(symbol, df, trade_date):
"""
Find the first opening-range breakout for one stock.

```
Returns either:
    trade dictionary
or:
    None
"""

if df is None or df.empty:
    return None

day = df[df.index.date == trade_date]

if day.empty:
    return None

opening = day.between_time("09:15", "09:29")

# We expect three 5-minute candles:
# 09:15, 09:20, 09:25
if len(opening) < 3:
    return None

opening_high = float(opening["High"].max())
opening_low = float(opening["Low"].min())

opening_range = opening_high - opening_low

if opening_range <= 0:
    return None

# Opening range percentage relative to midpoint.
midpoint = (opening_high + opening_low) / 2

if midpoint <= 0:
    return None

opening_range_pct = opening_range / midpoint * 100

# Candles after 09:30 and before cutoff.
post_open = day.between_time("09:30", "13:55")

if post_open.empty:
    return None

previous_close = fetch_previous_close(symbol)

# --------------------------------------------------------
# Find first breakout
# --------------------------------------------------------

breakout = None
direction = None

for dt, row in post_open.iterrows():

    # Entry cutoff
    if dt.hour > ENTRY_CUTOFF_HOUR:
        continue

    if (
        dt.hour == ENTRY_CUTOFF_HOUR
        and dt.minute >= ENTRY_CUTOFF_MINUTE
    ):
        continue

    candle_high = float(row["High"])
    candle_low = float(row["Low"])

    above = candle_high > opening_high
    below = candle_low < opening_low

    # If a single candle crosses BOTH sides, we cannot know
    # intrabar order from OHLC data.
    #
    # Conservative rule:
    # reject the ambiguous candle rather than inventing
    # an entry direction.
    if above and below:
        continue

    if above:
        breakout = (dt, row)
        direction = "LONG"
        break

    if below:
        breakout = (dt, row)
        direction = "SHORT"
        break

if breakout is None:
    return None

breakout_time, candle = breakout

candle_open = float(candle["Open"])
candle_high = float(candle["High"])
candle_low = float(candle["Low"])
candle_close = float(candle["Close"])

# --------------------------------------------------------
# Entry
# --------------------------------------------------------

if direction == "LONG":
    raw_entry = opening_high
    entry = raw_entry * (1 + SLIPPAGE_PCT)

    sl = candle_low

else:
    raw_entry = opening_low
    entry = raw_entry * (1 - SLIPPAGE_PCT)

    sl = candle_high

entry = round(entry, 2)
sl = round(sl, 2)

if direction == "LONG" and sl >= entry:
    return None

if direction == "SHORT" and sl <= entry:
    return None

stop_distance = abs(entry - sl)

if stop_distance / entry < MIN_STOP_PCT:
    return None

qty, risk_rs = calculate_position_size(entry, sl)

if qty <= 0:
    return None

capital_used = qty * entry

# --------------------------------------------------------
# Exit simulation
# --------------------------------------------------------

# The breakout candle counts as candle #1.
breakout_position = day.index.get_loc(breakout_time)

future = day.iloc[
    breakout_position:
    breakout_position + MAX_HOLD_BARS
]

if future.empty:
    return None

exit_time = None
exit_price = None
exit_reason = None
bars_held = 0

mfe = 0.0
mae = 0.0

for n, (dt, row) in enumerate(future.iterrows(), start=1):

    high = float(row["High"])
    low = float(row["Low"])

    if direction == "LONG":

        favorable = high - entry
        adverse = entry - low

        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        # Conservative assumption:
        # stop gets priority when both SL and breakout-side
        # favorable movement occur inside the same OHLC candle.
        if low <= sl:
            exit_time = dt
            exit_price = min(float(row["Open"]), sl)
            exit_reason = "STOPLOSS"
            bars_held = n
            break

    else:

        favorable = entry - low
        adverse = high - entry

        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        if high >= sl:
            exit_time = dt
            exit_price = max(float(row["Open"]), sl)
            exit_reason = "STOPLOSS"
            bars_held = n
            break

    if n == MAX_HOLD_BARS:

        exit_time = dt
        exit_price = float(row["Close"])
        exit_reason = "TIME_EXIT"
        bars_held = n
        break

if exit_time is None:
    return None

# --------------------------------------------------------
# P&L
# --------------------------------------------------------

if direction == "LONG":
    gross = (exit_price - entry) * qty
else:
    gross = (entry - exit_price) * qty

# Approximate round-trip transaction cost.
costs = (entry + exit_price) * qty / 2 * COST_PCT_RT

pnl = gross - costs

pnl_pct = pnl / capital_used * 100 if capital_used else 0.0

r_mult = pnl / risk_rs if risk_rs else 0.0

mfe_rs = mfe * qty
mae_rs = mae * qty

mfe_r = mfe_rs / risk_rs if risk_rs else 0.0
mae_r = mae_rs / risk_rs if risk_rs else 0.0

trade_id = (
    f"OB-{symbol.replace('.NS', '')}-"
    f"{trade_date:%Y%m%d}-{direction}"
)

return {
    "Trade_ID": trade_id,
    "Date": trade_date,

    "Symbol": symbol,
    "Direction": direction,

    "Prev_Close": round(previous_close, 2)
    if pd.notna(previous_close) else np.nan,

    "Opening_High": round(opening_high, 2),
    "Opening_Low": round(opening_low, 2),
    "Opening_Range": round(opening_range, 2),
    "Opening_Range_Pct": round(opening_range_pct, 3),

    "Breakout_Time": breakout_time,
    "Breakout_Candle_Open": round(candle_open, 2),
    "Breakout_Candle_High": round(candle_high, 2),
    "Breakout_Candle_Low": round(candle_low, 2),
    "Breakout_Candle_Close": round(candle_close, 2),

    "Entry_Time": breakout_time,
    "Entry_Price": round(entry, 2),
    "SL": round(sl, 2),

    "Qty": qty,
    "Risk_Rs": round(risk_rs, 2),
    "Capital_Used": round(capital_used, 2),

    "Exit_Time": exit_time,
    "Exit_Price": round(exit_price, 2),
    "Exit_Reason": exit_reason,

    "Gross_PnL_Rs": round(gross, 2),
    "Costs_Rs": round(costs, 2),
    "PnL_Rs": round(pnl, 2),
    "PnL_pct": round(pnl_pct, 3),
    "R_Mult": round(r_mult, 3),

    "MFE_Rs": round(mfe_rs, 2),
    "MAE_Rs": round(mae_rs, 2),
    "MFE_R": round(mfe_r, 3),
    "MAE_R": round(mae_r, 3),

    "Bars_Held": bars_held
}
```

# ============================================================

# CSV

# ============================================================

def load_trades():

```
if os.path.exists(TRADES_CSV) and os.path.getsize(TRADES_CSV) > 0:

    df = pd.read_csv(TRADES_CSV)

    for c in [
        "Date",
        "Breakout_Time",
        "Entry_Time",
        "Exit_Time"
    ]:
        if c in df.columns:
            df[c] = pd.to_datetime(
                df[c],
                errors="coerce"
            )

    return df

return pd.DataFrame(columns=TRADE_COLS)
```

def save_trades(df):
df.to_csv(TRADES_CSV, index=False)

# ============================================================

# DAILY RUN

# ============================================================

def daily_run():

```
ist = now_ist()

print(
    f"=============== "
    f"OPENING BREAKOUT @ {ist:%Y-%m-%d %H:%M} IST "
    f"==============="
)

trades = load_trades()

# We normally run after market close.
# Use the current IST date.
trade_date = ist.date()

# Weekend protection.
if trade_date.weekday() >= 5:
    print("Weekend — nothing to do.")
    return trades

# Prevent duplicate daily execution.
existing_dates = set()

if len(trades):
    existing_dates = set(
        pd.to_datetime(trades["Date"], errors="coerce")
        .dt.date
        .dropna()
    )

if trade_date in existing_dates:
    print(
        f"Trades for {trade_date} already exist. "
        f"Skipping duplicate run."
    )
    return trades

print(
    f"\nScanning {len(UNIVERSE)} NIFTY 50 stocks..."
)

candidates = []

for n, symbol in enumerate(UNIVERSE, start=1):

    print(
        f"[{n:02d}/{len(UNIVERSE)}] {symbol}",
        end=" ... "
    )

    df = fetch_intraday(symbol)

    if df is None:
        print("NO DATA")
        continue

    result = simulate_stock(
        symbol,
        df,
        trade_date
    )

    if result is None:
        print("NO BREAKOUT")
        continue

    print(
        f"{result['Direction']} "
        f"entry={result['Entry_Price']:.2f} "
        f"SL={result['SL']:.2f} "
        f"PnL=₹{result['PnL_Rs']:.2f}"
    )

    candidates.append(result)

    time.sleep(0.2)

# --------------------------------------------------------
# Daily trade limit
# --------------------------------------------------------

if candidates:

    # For v1, rank by earliest breakout.
    candidates.sort(
        key=lambda x: x["Breakout_Time"]
    )

    selected = candidates[:MAX_TRADES_PER_DAY]

else:
    selected = []

if selected:

    new_df = pd.DataFrame(selected)

    trades = pd.concat(
        [trades, new_df],
        ignore_index=True
    )

save_trades(trades)

# --------------------------------------------------------
# Dashboard
# --------------------------------------------------------

print("\n================ DAILY RESULT ================")

print(
    f"Candidates: {len(candidates)}"
)

print(
    f"Selected trades: {len(selected)}"
)

if selected:

    today = pd.DataFrame(selected)

    print(
        today[
            [
                "Symbol",
                "Direction",
                "Entry_Price",
                "SL",
                "Exit_Price",
                "Exit_Reason",
                "PnL_Rs",
                "R_Mult"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nToday's P&L: "
        f"₹{today.PnL_Rs.sum():,.2f}"
    )

    print(
        f"Today's R: "
        f"{today.R_Mult.sum():.2f}R"
    )

# --------------------------------------------------------
# Historical dashboard
# --------------------------------------------------------

closed = trades.copy()

if len(closed):

    wins = (closed["PnL_Rs"] > 0).sum()

    total = len(closed)

    win_rate = (
        wins / total * 100
        if total else 0
    )

    avg_r = closed["R_Mult"].mean()

    gross_profit = closed.loc[
        closed.PnL_Rs > 0,
        "PnL_Rs"
    ].sum()

    gross_loss = abs(
        closed.loc[
            closed.PnL_Rs < 0,
            "PnL_Rs"
        ].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    print("\n================ ALL-TIME PAPER RESULTS ================")

    print(f"Trades:          {total}")
    print(f"Win rate:        {win_rate:.1f}%")
    print(f"Net P&L:         ₹{closed.PnL_Rs.sum():,.2f}")
    print(f"Average R:       {avg_r:.3f}")
    print(f"Profit Factor:   {profit_factor:.2f}")

    print(
        "\nBy direction:"
    )

    print(
        closed.groupby("Direction").agg(
            Trades=("PnL_Rs", "count"),
            Win_Rate=("PnL_Rs", lambda x: (x > 0).mean() * 100),
            Net_PnL=("PnL_Rs", "sum"),
            Avg_R=("R_Mult", "mean")
        ).round(3).to_string()
    )

    print(
        "\nBy exit reason:"
    )

    print(
        closed.groupby("Exit_Reason").agg(
            Trades=("PnL_Rs", "count"),
            Net_PnL=("PnL_Rs", "sum"),
            Avg_R=("R_Mult", "mean")
        ).round(3).to_string()
    )

print(
    f"\nSaved → {TRADES_CSV}"
)

# --------------------------------------------------------
# Telegram
# --------------------------------------------------------

msg = (
    f"📊 Opening Breakout | {trade_date:%d-%b-%Y}\n"
    f"Candidates: {len(candidates)}\n"
    f"Trades: {len(selected)}"
)

if selected:

    today = pd.DataFrame(selected)

    msg += (
        f"\nP&L: ₹{today.PnL_Rs.sum():,.0f}"
        f"\nR: {today.R_Mult.sum():.2f}"
    )

if len(trades):

    msg += (
        f"\nTotal trades: {len(trades)}"
        f"\nTotal P&L: ₹{trades.PnL_Rs.sum():,.0f}"
    )

telegram(msg)

return trades
```

if **name** == "**main**":

```
try:
    daily_run()

except Exception as e:

    telegram(
        f"🚨 Opening Breakout FAILED: {e}"
    )

    raise
```
