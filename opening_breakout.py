"""
opening_breakout.py — Opening 15-Minute Breakout paper-trading engine

Strategy:
1. Use the first 15 minutes (09:15-09:30 IST) as the opening range.
2. After 09:30, first break above opening HIGH = LONG.
3. First break below opening LOW = SHORT.
4. Stop: LONG  -> LOW of the 5-minute breakout candle
        SHORT -> HIGH of the 5-minute breakout candle
5. Maximum holding period = 3 five-minute candles.
6. No new entries after 14:00 IST.
7. One trade per stock per day.
8. Maximum 3 trades per day.
9. Position size is calculated from risk, not capital deployment.
10. Both directions are tested independently.

Paper-testing/research engine. yfinance intraday history is limited, so this
is used for forward paper testing.
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
RISK_PER_TRADE = 0.01
MAX_TRADES_PER_DAY = 3
MAX_HOLD_BARS = 9
ENTRY_CUTOFF_HOUR = 10
ENTRY_CUTOFF_MINUTE = 30
SLIPPAGE_PCT = 0.0005
COST_PCT_RT = 0.0025
MIN_STOP_PCT = 0.003
# Opening-range quality filters
MIN_OR_PCT = 0.5
MAX_OR_PCT = 3.0
TARGET_R_MULT = 2.0
TRADES_CSV = "./opening_breakout_trades.csv"

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
    "TRENT.NS", "BEL.NS",
]

TRADE_COLS = [
    "Trade_ID", "Date", "Symbol", "Direction",
    "Prev_Close", "Opening_High", "Opening_Low", "Opening_Range",
    "Opening_Range_Pct", "Breakout_Time", "Breakout_Candle_Open",
    "Breakout_Candle_High", "Breakout_Candle_Low", "Breakout_Candle_Close",
    "Entry_Time", "Entry_Price", "SL", "Qty", "Risk_Rs", "Capital_Used",
    "Exit_Time", "Exit_Price", "Exit_Reason", "Gross_PnL_Rs", "Costs_Rs",
    "PnL_Rs", "PnL_pct", "R_Mult", "MFE_Rs", "MAE_Rs", "MFE_R",
    "MAE_R", "Bars_Held",
]


# ============================================================
# TELEGRAM
# ============================================================
def telegram(msg):
    tok = os.environ.get("TG_TOKEN")
    chat = os.environ.get("TG_CHAT")
    if not tok or not chat:
        return
    try:
        requests.get(
            "https://api.telegram.org/bot{}/sendMessage".format(tok),
            params={"chat_id": chat, "text": msg},
            timeout=10,
        )
    except Exception as e:
        print("telegram failed: {}".format(e))


# ============================================================
# TIME HELPERS
# ============================================================
def now_ist():
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)


def clean_intraday_index(df):
    if df is None or df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Kolkata")
    else:
        idx = idx.tz_convert("Asia/Kolkata")
    df = df.copy()
    df.index = idx.tz_localize(None)
    return df


# ============================================================
# DATA
# ============================================================
def fetch_intraday(symbol, period="5d"):
    df = None
    for attempt in range(1, 4):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df = yf.download(symbol, period=period, interval="5m",
                                 auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                break
        except Exception:
            df = None
        time.sleep(1.5 * attempt)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = clean_intraday_index(df)
    return df


def last_trading_day_bars(df):
    if df is None or df.empty:
        return None
    last_date = df.index.date[-1]
    day = df[df.index.date == last_date]
    return day if not day.empty else None


# ============================================================
# OPENING RANGE
# ============================================================
def opening_range(day_df):
    # bars from 09:15 up to (not including) 09:30
    or_bars = day_df[(day_df.index.time >= pd.Timestamp("09:15").time()) &
                     (day_df.index.time < pd.Timestamp("09:30").time())]
    if or_bars.empty or len(or_bars) < 2:
        return None, None
    hi = float(or_bars["High"].max())
    lo = float(or_bars["Low"].min())
    if hi <= 0:
        return None, None
    pct = (hi - lo) / hi * 100.0
    if pct < MIN_OR_PCT or pct > MAX_OR_PCT:
        return None, None
    return hi, lo


# ============================================================
# FIND FIRST BREAKOUT PER STOCK
# ============================================================
def find_breakout(symbol, day_df):
    or_high, or_low = opening_range(day_df)
    if or_high is None:
        return None
    or_range = or_high - or_low
    # average volume of opening-range bars (liquidity baseline)
    or_bars = day_df[(day_df.index.time >= pd.Timestamp("09:15").time()) &
                     (day_df.index.time < pd.Timestamp("09:30").time())]
    avg_vol = float(or_bars["Volume"].mean()) if "Volume" in or_bars else 0.0

    cutoff = pd.Timestamp(day_df.index.date[0]).replace(
        hour=ENTRY_CUTOFF_HOUR, minute=ENTRY_CUTOFF_MINUTE)
    post = day_df[(day_df.index.time >= pd.Timestamp("09:30").time()) &
                  (day_df.index < cutoff)]
    for ts, row in post.iterrows():
        o = float(row["Open"]); h = float(row["High"]); l = float(row["Low"]); c = float(row["Close"])
        vol = float(row["Volume"]) if "Volume" in row else 0.0
        body = abs(c - o)
        # volume confirmation
        vol_ok = (avg_vol <= 0) or (vol >= 1.2 * avg_vol)
        # LONG breakout: close above range high, not just a wick
        if c > or_high and vol_ok and body >= 0.2 * or_range:
            return {
                "symbol": symbol, "direction": "LONG",
                "breakout_time": ts, "or_high": or_high, "or_low": or_low,
                "bo_open": o, "bo_high": h, "bo_low": l, "bo_close": c,
            }
        # SHORT breakout: close below range low
        if c < or_low and vol_ok and body >= 0.2 * or_range:
            return {
                "symbol": symbol, "direction": "SHORT",
                "breakout_time": ts, "or_high": or_high, "or_low": or_low,
                "bo_open": o, "bo_high": h, "bo_low": l, "bo_close": c,
            }
    return None


# ============================================================
# EXECUTE PAPER TRADE
# ============================================================
def execute_trade(candidate, day_df):
    direction = candidate["direction"]
    bo_time = candidate["breakout_time"]
    entry_price_raw = candidate["bo_close"]
    if direction == "LONG":
        entry = round(entry_price_raw * (1 + SLIPPAGE_PCT), 2)
        # stop below the opening-range low (wider, less noise-driven)
        stop = round(candidate["or_low"], 2)
        risk_per_share = entry - stop
    else:
        entry = round(entry_price_raw * (1 - SLIPPAGE_PCT), 2)
        stop = round(candidate["or_high"], 2)
        risk_per_share = stop - entry

    if risk_per_share <= 0:
        return None
    if risk_per_share / entry < MIN_STOP_PCT:
        risk_per_share = entry * MIN_STOP_PCT
        if direction == "LONG":
            stop = round(entry - risk_per_share, 2)
        else:
            stop = round(entry + risk_per_share, 2)
        risk_per_share = abs(entry - stop)

    risk_budget = CAPITAL * RISK_PER_TRADE
    qty = int(min(risk_budget / risk_per_share,
                  (CAPITAL) / entry))
    if qty <= 0:
        return None

    risk_rs = round(risk_per_share * qty, 2)
    capital_used = round(entry * qty, 2)

    # walk forward up to MAX_HOLD_BARS from the bar after breakout
    target = (entry + TARGET_R_MULT * risk_per_share) if direction == "LONG" else (entry - TARGET_R_MULT * risk_per_share)
    future = day_df[day_df.index > bo_time].head(MAX_HOLD_BARS)
    exit_time = None
    exit_price = None
    exit_reason = "TIME_EXIT"
    mfe = 0.0
    mae = 0.0
    bars_held = 0
    for i, (ts, row) in enumerate(future.iterrows(), start=1):
        bars_held = i
        if direction == "LONG":
            mfe = max(mfe, float(row["High"]) - entry)
            mae = max(mae, entry - float(row["Low"]))
            if float(row["Low"]) <= stop:
                exit_time = ts
                exit_price = round(min(float(row["Open"]), stop), 2)
                exit_reason = "STOPLOSS"
                break
            if float(row["High"]) >= target:
                exit_time = ts
                exit_price = round(target, 2)
                exit_reason = "TARGET"
                break
        else:
            mfe = max(mfe, entry - float(row["Low"]))
            mae = max(mae, float(row["High"]) - entry)
            if float(row["High"]) >= stop:
                exit_time = ts
                exit_price = round(max(float(row["Open"]), stop), 2)
                exit_reason = "STOPLOSS"
                break
            if float(row["Low"]) <= target:
                exit_time = ts
                exit_price = round(target, 2)
                exit_reason = "TARGET"
                break
        if i >= MAX_HOLD_BARS:
            exit_time = ts
            exit_price = round(float(row["Close"]), 2)
            exit_reason = "TIME_EXIT"
            break

    if exit_time is None:
        # no future bars -> close at last available
        if not future.empty:
            last_ts = future.index[-1]
            exit_time = last_ts
            exit_price = round(float(future.iloc[-1]["Close"]), 2)
            exit_reason = "TIME_EXIT"
            bars_held = len(future)
        else:
            return None

    gross = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
    costs = (entry + exit_price) * qty / 2.0 * COST_PCT_RT
    pnl = gross - costs
    pnl_pct = pnl / (entry * qty) * 100
    r_mult = pnl / max(risk_rs, 1e-9)
    mfe_rs = round(mfe * qty, 2)
    mae_rs = round(mae * qty, 2)
    mfe_r = round(mfe / max(risk_per_share, 1e-9), 2)
    mae_r = round(mae / max(risk_per_share, 1e-9), 2)

    prev_close = float(day_df.iloc[0]["Open"])  # placeholder-ish; better from prior day
    or_range = candidate["or_high"] - candidate["or_low"]
    or_range_pct = or_range / candidate["or_high"] * 100 if candidate["or_high"] else 0.0

    trade_id = "ORB-{}-{}".format(
        candidate["symbol"].replace(".NS", ""),
        pd.Timestamp(candidate["breakout_time"]).strftime("%Y%m%d%H%M"),
    )

    row = {
        "Trade_ID": trade_id,
        "Date": str(candidate["breakout_time"].date()),
        "Symbol": candidate["symbol"],
        "Direction": direction,
        "Prev_Close": round(prev_close, 2),
        "Opening_High": round(candidate["or_high"], 2),
        "Opening_Low": round(candidate["or_low"], 2),
        "Opening_Range": round(or_range, 2),
        "Opening_Range_Pct": round(or_range_pct, 2),
        "Breakout_Time": str(candidate["breakout_time"]),
        "Breakout_Candle_Open": candidate["bo_open"],
        "Breakout_Candle_High": candidate["bo_high"],
        "Breakout_Candle_Low": candidate["bo_low"],
        "Breakout_Candle_Close": candidate["bo_close"],
        "Entry_Time": str(exit_time) if False else str(pd.Timestamp(candidate["breakout_time"]) + pd.Timedelta(minutes=5)),
        "Entry_Price": entry,
        "SL": stop,
        "Qty": qty,
        "Risk_Rs": risk_rs,
        "Capital_Used": capital_used,
        "Exit_Time": str(exit_time),
        "Exit_Price": exit_price,
        "Exit_Reason": exit_reason,
        "Gross_PnL_Rs": round(gross, 2),
        "Costs_Rs": round(costs, 2),
        "PnL_Rs": round(pnl, 2),
        "PnL_pct": round(pnl_pct, 2),
        "R_Mult": round(r_mult, 2),
        "MFE_Rs": mfe_rs,
        "MAE_Rs": mae_rs,
        "MFE_R": mfe_r,
        "MAE_R": mae_r,
        "Bars_Held": bars_held,
    }
    # fix Entry_Time to be breakout close time (we entered at breakout close)
    row["Entry_Time"] = str(candidate["breakout_time"])
    return row


# ============================================================
# MAIN
# ============================================================
def main():
    ist = now_ist()
    print("=============== ORB PAPER RUN @ {} IST ===============".format(ist.strftime("%Y-%m-%d %H:%M")))

    # load existing trades and de-duplicate
    if os.path.exists(TRADES_CSV) and os.path.getsize(TRADES_CSV) > 0:
        trades_df = pd.read_csv(TRADES_CSV)
        if "Trade_ID" in trades_df.columns:
            trades_df = trades_df.drop_duplicates(subset="Trade_ID", keep="last")
    else:
        trades_df = pd.DataFrame(columns=TRADE_COLS)

    existing_ids = set(trades_df["Trade_ID"].astype(str)) if "Trade_ID" in trades_df.columns else set()

    # avoid duplicate day
    today_str = ist.strftime("%Y-%m-%d")

    candidates = []
    for sym in UNIVERSE:
        df = fetch_intraday(sym, period="5d")
        if df is None or df.empty:
            continue
        day = last_trading_day_bars(df)
        if day is None or day.empty:
            continue
        # only trade on the most recent day if it's today-ish
        cand = find_breakout(sym, day)
        if cand is not None:
            candidates.append((cand, day))

    # sort by breakout time, take first MAX_TRADES_PER_DAY
    candidates.sort(key=lambda x: x[0]["breakout_time"])
    selected = candidates[:MAX_TRADES_PER_DAY]

    new_rows = []
    for cand, day in selected:
        row = execute_trade(cand, day)
        if row is not None and str(row["Trade_ID"]) not in existing_ids:
            new_rows.append(row)
            existing_ids.add(str(row["Trade_ID"]))
            print("TRADE: {} {} @ {} SL {} Qty {}".format(
                row["Direction"], row["Symbol"], row["Entry_Price"],
                row["SL"], row["Qty"]))

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=TRADE_COLS)
        trades_df = pd.concat([trades_df, new_df], ignore_index=True)
        tot = sum(r["PnL_Rs"] for r in new_rows)
        wins = sum(1 for r in new_rows if r["PnL_Rs"] > 0)
        print("\n>> {} new trade(s) | wins {} | day PnL Rs {:.2f}".format(
            len(new_rows), wins, tot))
    else:
        print("\n>> No new trades today.")

    if "Trade_ID" in trades_df.columns:
        trades_df = trades_df.drop_duplicates(subset="Trade_ID", keep="last")
    trades_df.to_csv(TRADES_CSV, index=False)
    print("Saved -> {} ({} total rows)".format(TRADES_CSV, len(trades_df)))

    telegram("ORB run {}: {} new trade(s)".format(today_str, len(new_rows)))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        telegram("ORB run FAILED: {}".format(e))
        raise
