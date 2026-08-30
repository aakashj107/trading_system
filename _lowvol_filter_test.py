"""
Low-volatility basket as a REGIME FILTER for a Nifty consensus signal.
Hypothesis (from web: low-vol regimes had smaller DD / faster recovery):
  only take trades when Nifty realized volatility is LOW (bottom quantile).
Tests whether a low-vol ON/OFF gate improves a simple Nifty trend signal OOS.

Signal: trend consensus = Close>SMA50 & SMA50>SMA200 (long), inverse (short).
Filter: LOW-VOL = 20d realized vol (stdev of daily returns) in bottom 33% of
        its own trailing history. Trade only when LOW-VOL is true.
Ladder: un-gated vs gated, full 3y and 60/40 OOS, cost 0.04%/side, 2-contract cap.
No faking.
"""
import numpy as np, pandas as pd
from data_fetcher import YFinanceDataFetcher

def trend_signal(df):
    c = df["Close"]; s50 = c.rolling(50).mean(); s200 = c.rolling(200).mean()
    sig = pd.Series(0, index=df.index); pos = 0
    cv, a, b = c.values, s50.values, s200.values
    for i in range(200, len(df)):
        if pos == 0:
            if cv[i] > a[i] and a[i] > b[i]: pos = 1
            elif cv[i] < a[i] and a[i] < b[i]: pos = -1
        elif pos == 1 and (cv[i] < a[i] or a[i] < b[i]): pos = 0
        elif pos == -1 and (cv[i] > a[i] or a[i] > b[i]): pos = 0
        sig.iloc[i] = pos
    return sig

def realized_vol(df, win=20):
    r = df["Close"].pct_change()
    return r.rolling(win).std() * np.sqrt(252) * 100

def low_vol_flag(df, win=20, q=0.33):
    rv = realized_vol(df, win)
    flag = pd.Series(False, index=df.index)
    # trailing quantile: flag True when rv in bottom q of prior 252 days
    for i in range(252, len(df)):
        hist = rv.iloc[i-252:i]
        if pd.notna(rv.iloc[i]) and rv.iloc[i] <= hist.quantile(q):
            flag.iloc[i] = True
    return flag

def backtest(df, use_filter):
    sig = trend_signal(df)
    flag = low_vol_flag(df) if use_filter else pd.Series(True, index=df.index)
    c = df["Close"].values; s = sig.values; fl = flag.values
    cash, pos, peak = 1.0, 0, 1.0; eq = []; holds = 0
    for i in range(len(c)):
        active = fl[i]
        if pos != 0 and i > 0:
            cash *= (1 + (c[i]-c[i-1])/c[i-1] * pos * 2)
        if pos == 0 and s[i] != 0 and active:
            pos = s[i]; cash *= (1 - 0.0004*2); holds += 1
        elif pos != 0 and (s[i] == 0 or not active):
            cash *= (1 - 0.0004*2); pos = 0
        peak = max(peak, cash); eq.append(cash/peak)
    eq = pd.Series(eq, index=df.index[-len(eq):])
    return (cash-1)*100, (1-eq.min())*100, holds

def report(name, df):
    ug = backtest(df, False); g = backtest(df, True)
    print(f"\n=== {name} ({len(df)} bars) ===")
    print(f"  UN-GATED : return={ug[0]:7.2f}%  maxDD={ug[1]:6.2f}%  holds={ug[2]}")
    print(f"  LOW-VOL  : return={g[0]:7.2f}%  maxDD={g[1]:6.2f}%  holds={g[2]}")
    better = "LOW-VOL filter HELPS" if g[0] > ug[0] else "LOW-VOL filter does NOT help"
    print(f"  -> {better}")

f = YFinanceDataFetcher(cache_dir="data_cache")
df3 = f.get_nifty_index(period="3y")
df1 = df3[df3.index >= df3.index.max() - pd.Timedelta(days=365)]
print("NIFTY rows:", len(df3))
report("FULL 3y", df3)
report("1y window", df1)

# 60/40 OOS
n = len(df3); cut = int(n*0.6)
print("\n=== 60/40 OOS split (cost x1) ===")
ug_is, g_is = backtest(df3.iloc[:cut], False), backtest(df3.iloc[:cut], True)
ug_oos, g_oos = backtest(df3.iloc[cut:], False), backtest(df3.iloc[cut:], True)
print(f"  IN-SAMPLE   UN-GATED={ug_is[0]:7.2f}%  LOW-VOL={g_is[0]:7.2f}%")
print(f"  OUT-OF-SAMPLE UN-GATED={ug_oos[0]:7.2f}%  LOW-VOL={g_oos[0]:7.2f}%")
print("  VERDICT:", "low-vol filter improves OOS -> candidate" if g_oos[0] > ug_oos[0] else "low-vol filter does NOT improve OOS -> no value")
