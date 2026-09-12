"""
daily_run.py — S2 paper-trading pipeline (20/50 DMA + RSI crossover)
Config: 3×ATR stop, 2R target, 1% risk, NIFTY>200SMA regime filter.
Runs anywhere:  python daily_run.py
For GitHub Actions: cron 17:00 IST Mon–Fri → updates CSVs → commits → Telegram heartbeat.
"""

import os, time, contextlib, io, logging
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta, time as dtime, timezone

logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ---------------- CONFIG (locked after 10-year validation) ----------------
CSV_DIR           = './'
SIGNALS_CSV       = CSV_DIR + 'signals_log.csv'
TRADES_CSV        = CSV_DIR + 'paper_trades.csv'

CAPITAL           = 100000
RISK_PER_TRADE    = 0.01
ATR_SL_MULT       = 3.0        # S2 V2 — validated
TARGET_RR         = 2.0
MAX_POSITIONS     = 5
MAX_PER_STRATEGY  = 5          # single strategy → portfolio cap governs
MAX_HOLD_BARS     = 60
COST_PCT_RT       = 0.0025
SLIPPAGE_PCT      = 0.0005
USE_NIFTY_FILTER  = True
USE_VOLUME_FILTER = True
USE_200SMA_FILTER = True
RSI_MIN, RSI_MAX  = 50, 75
SIGNAL_LOOKBACK   = '2y'

UNIVERSE = [
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','SBIN.NS',
    'BHARTIARTL.NS','ITC.NS','HINDUNILVR.NS','LT.NS','KOTAKBANK.NS','AXISBANK.NS',
    'BAJFINANCE.NS','ASIANPAINT.NS','MARUTI.NS','TITAN.NS','SUNPHARMA.NS',
    'ULTRACEMCO.NS','NTPC.NS','POWERGRID.NS','TATASTEEL.NS','WIPRO.NS','TECHM.NS',
    'HCLTECH.NS','NESTLEIND.NS','ONGC.NS','COALINDIA.NS','GRASIM.NS','JSWSTEEL.NS',
    'HINDALCO.NS','ADANIENT.NS','ADANIPORTS.NS','BAJAJFINSV.NS','M&M.NS','CIPLA.NS',
    'DRREDDY.NS','DIVISLAB.NS','BRITANNIA.NS','EICHERMOT.NS','HEROMOTOCO.NS',
    'INDUSINDBK.NS','SHRIRAMFIN.NS','TATACONSUM.NS','APOLLOHOSP.NS','SBILIFE.NS',
    'HDFCLIFE.NS','BPCL.NS','TRENT.NS','BEL.NS'
]   # TATAMOTORS.NS removed post-demerger — keep YOUR health-checked list here

# ---------------- INDICATORS ----------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n, min_periods=n).mean()

def rsi(close, n=14):
    d  = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))

def atr(df, n=14):
    pc = df['Close'].shift()
    tr = pd.concat([df['High']-df['Low'], (df['High']-pc).abs(),
                    (df['Low']-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

# ---------------- DATA ----------------
FAILED = {}

def fetch(symbol, period='1y', retries=3):
    """Daily OHLCV with retry/backoff; drops today's incomplete candle."""
    df = None
    for attempt in range(1, retries + 1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df = yf.download(symbol, period=period, interval='1d',
                                 auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                break
        except Exception:
            df = None
        time.sleep(1.5 * attempt)
    if df is None or df.empty:
        if symbol not in FAILED:
            FAILED[symbol] = 'no data'
            print(f"⚠ Skipping {symbol} — no data after {retries} tries")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    if len(df) < 60:
        if symbol not in FAILED:
            FAILED[symbol] = 'short history'
            print(f"⚠ Skipping {symbol} — only {len(df)} bars")
        return None
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    if (ist.weekday() < 5 and ist.time() < dtime(15, 30)
            and df.index[-1].date() == ist.date()):
        df = df.iloc[:-1]
    time.sleep(0.3)
    return df

def nifty_regime(period='1y'):
    n = fetch('^NSEI', period)
    if n is None: return pd.Series(dtype=bool)
    return n['Close'] > sma(n['Close'], 200)

# ---------------- STRATEGY (S2 only — S1 dropped after split-half FAIL) ----------------
def prep_dma(df):
    d = df.copy()
    d['SMA20'], d['SMA50'], d['SMA200'] = sma(d.Close,20), sma(d.Close,50), sma(d.Close,200)
    d['RSI14'], d['ATR'], d['VOL20'] = rsi(d.Close), atr(d), d.Volume.rolling(20).mean()
    cross_up  = (d.SMA20 > d.SMA50) & (d.SMA20.shift() <= d.SMA50.shift())
    rsi_ok    = (d.RSI14 > RSI_MIN) & (d.RSI14 < RSI_MAX)
    trend_ok  = d.Close > d.SMA200 if USE_200SMA_FILTER else True
    vol_ok    = d.Volume > d.VOL20 if USE_VOLUME_FILTER else True
    d['BUY']        = cross_up & rsi_ok & trend_ok & vol_ok
    d['EXIT_CROSS'] = (d.SMA20 < d.SMA50) & (d.SMA20.shift() >= d.SMA50.shift())
    return d

STRATEGIES = {'DMA_20_50_RSI': prep_dma}

# ---------------- TELEGRAM (silent no-op without env vars) ----------------
def telegram(msg):
    tok, chat = os.environ.get('TG_TOKEN'), os.environ.get('TG_CHAT')
    if not tok or not chat: return
    try:
        requests.get(f"https://api.telegram.org/bot{tok}/sendMessage",
                     params=dict(chat_id=chat, text=msg), timeout=10)
    except Exception as e:
        print(f"telegram failed: {e}")

# ---------------- LOG I/O ----------------
SIGNAL_COLS = ['Signal_Date','Symbol','Strategy','Signal_Close','Signal_ATR',
               'Plan_SL','Plan_Target','Expected_Ret_pct','Status',
               'Entry_Date','Entry_Price','SL','Target',
               'Last_Date','Last_Close','Open_PnL_pct',
               'Exit_Date','Exit_Price','Exit_Reason','Actual_Ret_pct','Exit_RR']
TRADE_COLS  = ['Trade_ID','Symbol','Strategy','Signal_Date','Entry_Date','Entry_Price',
               'Qty','Risk_Rs','SL','Target','Status',
               'Last_Date','Last_Close','Open_PnL_Rs',
               'Exit_Date','Exit_Price','Exit_Reason','PnL_Rs','PnL_pct','R_Mult','Days_Held']

def load_df(path, cols, date_cols):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
        for c in date_cols:
            if c in df.columns: df[c] = pd.to_datetime(df[c], errors='coerce')
        return df
    return pd.DataFrame(columns=cols)

def load_signals(): return load_df(SIGNALS_CSV, SIGNAL_COLS, ['Signal_Date','Entry_Date','Last_Date','Exit_Date'])
def load_trades():  return load_df(TRADES_CSV,  TRADE_COLS,  ['Signal_Date','Entry_Date','Last_Date','Exit_Date'])
def save_df(df, path): df.to_csv(path, index=False)

# ---------------- EXIT ENGINE ----------------
def evaluate_exit(d, entry_date, entry, sl, target):
    """Walk bars from entry; conservative: SL checked before target."""
    fwd = d[d.index >= entry_date]
    for n, (dt, row) in enumerate(fwd.iterrows(), start=1):
        if row.Low  <= sl:        return dt, round(min(row.Open, sl), 2), 'STOPLOSS'
        if row.High >= target:    return dt, target, 'TARGET'
        if bool(row.EXIT_CROSS):  return dt, round(row.Close, 2), 'CROSS_EXIT'
        if n >= MAX_HOLD_BARS:    return dt, round(row.Close, 2), 'TIME_EXIT'
    return None

# ---------------- DAILY RUN ----------------
def daily_run():
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    print(f"=============== PAPER RUN @ {ist:%Y-%m-%d %H:%M} IST ===============")
    sig, trd = load_signals(), load_trades()

    regime    = nifty_regime(SIGNAL_LOOKBACK) if USE_NIFTY_FILTER else pd.Series(dtype=bool)
    reg_dates = regime[regime].index if len(regime) else []
    APPLY_REGIME = USE_NIFTY_FILTER and len(reg_dates) > 0
    if USE_NIFTY_FILTER and not APPLY_REGIME:
        print("⚠ NIFTY regime data unavailable — filter OFF this run (flagged in heartbeat)")

    # 1) scan for fresh signals (last completed bar)
    data_cache, new_signals = {}, []
    for sym in UNIVERSE:
        df = fetch(sym, SIGNAL_LOOKBACK)
        if df is None or len(df) < 210: continue
        for strat, prep in STRATEGIES.items():
            d = prep(df)
            if APPLY_REGIME: d['BUY'] = d['BUY'] & d.index.isin(reg_dates)
            data_cache[(sym, strat)] = d
            if bool(d.BUY.iloc[-1]):
                sdate = d.index[-1]
                dup = len(sig) and ((sig.Symbol==sym)&(sig.Strategy==strat)&(sig.Signal_Date==sdate)).any()
                if not dup:
                    c, a = d.Close.iloc[-1], d.ATR.iloc[-1]
                    sl, tgt = c - ATR_SL_MULT*a, c + TARGET_RR*(c - ATR_SL_MULT*a)
                    new_signals.append(dict(Signal_Date=sdate, Symbol=sym, Strategy=strat,
                        Signal_Close=round(c,2), Signal_ATR=round(a,2),
                        Plan_SL=round(sl,2), Plan_Target=round(tgt,2),
                        Expected_Ret_pct=round((tgt/c-1)*100,2), Status='PENDING_ENTRY'))
    if new_signals:
        sig = pd.concat([sig, pd.DataFrame(new_signals)], ignore_index=True)
        print(f"\n>> {len(new_signals)} NEW SIGNAL(S):")
        print(pd.DataFrame(new_signals)[['Signal_Date','Symbol','Signal_Close','Plan_Target','Expected_Ret_pct']].to_string(index=False))
    else:
        print("\n>> No new signals today.")

    # 2) fill pending signals at next trading day's OPEN + open paper trades within risk limits
    open_tr  = trd[trd.Status=='OPEN']
    slots    = open_tr.groupby('Strategy').size().to_dict()
    open_cnt = len(open_tr)
    cap_used = float((open_tr.Entry_Price*open_tr.Qty).sum()) if len(open_tr) else 0.0
    for i, s in sig[sig.Status=='PENDING_ENTRY'].iterrows():
        d = data_cache.get((s.Symbol, s.Strategy))
        if d is None: continue
        nxt = d[d.index > s.Signal_Date]
        if len(nxt) == 0: continue
        bar   = nxt.iloc[0]
        entry = round(bar.Open*(1+SLIPPAGE_PCT), 2)
        sl    = round(entry - ATR_SL_MULT*s.Signal_ATR, 2)
        tgt   = round(entry + TARGET_RR*(entry - sl), 2)
        sig.loc[i, ['Status','Entry_Date','Entry_Price','SL','Target']] = ['ACTIVE', bar.name, entry, sl, tgt]
        if open_cnt < MAX_POSITIONS and slots.get(s.Strategy, 0) < MAX_PER_STRATEGY:
            risk_ps = entry - sl
            qty = int(min((CAPITAL*RISK_PER_TRADE)/risk_ps, (CAPITAL-cap_used)/entry))
            if qty > 0:
                tid = f"{s.Strategy[:4]}-{s.Symbol.replace('.NS','')}-{pd.Timestamp(s.Signal_Date):%Y%m%d}"
                trd = pd.concat([trd, pd.DataFrame([dict(Trade_ID=tid, Symbol=s.Symbol, Strategy=s.Strategy,
                    Signal_Date=s.Signal_Date, Entry_Date=bar.name, Entry_Price=entry, Qty=qty,
                    Risk_Rs=round(risk_ps*qty,2), SL=sl, Target=tgt, Status='OPEN')])], ignore_index=True)
                slots[s.Strategy] = slots.get(s.Strategy, 0) + 1
                open_cnt += 1; cap_used += entry*qty

    # 3) update ACTIVE signals: live P&L + exits
    for i, s in sig[sig.Status=='ACTIVE'].iterrows():
        d = data_cache.get((s.Symbol, s.Strategy))
        if d is None: continue
        if len(d[d.index >= s.Entry_Date]):
            lc = round(d.Close.iloc[-1], 2)
            sig.loc[i, ['Last_Date','Last_Close']] = [d.index[-1], lc]
            sig.loc[i, 'Open_PnL_pct'] = round((lc/s.Entry_Price-1)*100, 2)
        ex = evaluate_exit(d, s.Entry_Date, s.Entry_Price, s.SL, s.Target)
        if ex:
            dt, px, reason = ex
            sig.loc[i, ['Status','Exit_Date','Exit_Price','Exit_Reason']] = [reason, dt, px, reason]
            sig.loc[i, 'Actual_Ret_pct'] = round((px/s.Entry_Price-1)*100, 2)
            sig.loc[i, 'Exit_RR'] = round((px-s.Entry_Price)/max(s.Entry_Price-s.SL, 1e-9), 2)

    # 4) update OPEN paper trades: live P&L + exits
    for i, t in trd[trd.Status=='OPEN'].iterrows():
        d = data_cache.get((t.Symbol, t.Strategy))
        if d is None: continue
        if len(d[d.index >= t.Entry_Date]):
            lc = round(d.Close.iloc[-1], 2)
            trd.loc[i, ['Last_Date','Last_Close']] = [d.index[-1], lc]
            trd.loc[i, 'Open_PnL_Rs'] = round((lc-t.Entry_Price)*t.Qty, 2)
        ex = evaluate_exit(d, t.Entry_Date, t.Entry_Price, t.SL, t.Target)
        if ex:
            dt, px, reason = ex
            gross = (px - t.Entry_Price)*t.Qty
            cost  = (px + t.Entry_Price)*t.Qty/2*COST_PCT_RT
            pnl   = gross - cost
            trd.loc[i, ['Status','Exit_Date','Exit_Price','Exit_Reason','PnL_Rs','PnL_pct','R_Mult','Days_Held']] = [
                'CLOSED', dt, px, reason, round(pnl,2),
                round(pnl/(t.Entry_Price*t.Qty)*100, 2),
                round(pnl/max(t.Risk_Rs, 1e-9), 2),
                (pd.Timestamp(dt)-pd.Timestamp(t.Entry_Date)).days]

    # 5) dashboard + save + heartbeat
    ot, cl = trd[trd.Status=='OPEN'], trd[trd.Status=='CLOSED']
    print(f"\nOPEN PAPER POSITIONS: {len(ot)}/{MAX_POSITIONS}")
    if len(ot):
        print(ot[['Symbol','Entry_Date','Entry_Price','SL','Target','Last_Close','Open_PnL_Rs']].to_string(index=False))
    if len(cl):
        print(f"\nCLOSED: {len(cl)} | WR {(cl.PnL_Rs>0).mean()*100:.0f}% | Net ₹{cl.PnL_Rs.sum():,.0f} | AvgR {cl.R_Mult.mean():.2f}")
    save_df(sig, SIGNALS_CSV); save_df(trd, TRADES_CSV)
    print(f"\nSaved → {SIGNALS_CSV}\n        {TRADES_CSV}")

    msg = f"📡 S2 run {ist:%d-%b} | New signals: {len(new_signals)}"
    if USE_NIFTY_FILTER and not APPLY_REGIME: msg += " | ⚠ no-regime"
    if len(ot): msg += f" | Open {len(ot)} pos ₹{ot.Open_PnL_Rs.sum():,.0f}"
    if len(cl): msg += f" | Closed {len(cl)} ₹{cl.PnL_Rs.sum():,.0f}"
    telegram(msg)
    return sig, trd

if __name__ == '__main__':
    try:
        sig, trd = daily_run()
    except Exception as e:
        telegram(f"🚨 S2 run FAILED: {e}")
        raise
