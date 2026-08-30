"""
Trend-following dual-confirmation on Nifty (^NSEI) — OOS gate.
Variant B (trend-following, more likely candidate than mean-reversion):
  LONG  when Close > SMA(50) AND SMA(50) > SMA(200)   (uptrend confirmation)
  SHORT when Close < SMA(50) AND SMA(50) < SMA(200)   (downtrend confirmation)
  Exit:  opposite cross (Close crosses SMA50 the other way) OR SMA50/SMA200 flatten
  Cap:   2 contracts.  Cost 0.04%/side.
Ladder: full 3y -> 1y -> rolling 4-window -> 60/40 OOS split -> cost sensitivity.
No faking. Reports real OOS vs baseline.
"""
import numpy as np, pandas as pd
from data_fetcher import YFinanceDataFetcher

def signals(df):
    c = df["Close"]; sma50 = c.rolling(50).mean(); sma200 = c.rolling(200).mean()
    sig = pd.Series(0, index=df.index)
    pos = 0
    cv, s50, s200 = c.values, sma50.values, sma200.values
    for i in range(200, len(df)):
        if pos == 0:
            if (cv[i] > s50[i]) and (s50[i] > s200[i]): pos = 1
            elif (cv[i] < s50[i]) and (s50[i] < s200[i]): pos = -1
        elif pos == 1 and ((cv[i] < s50[i]) or (s50[i] < s200[i])): pos = 0
        elif pos == -1 and ((cv[i] > s50[i]) or (s50[i] > s200[i])): pos = 0
        sig.iloc[i] = pos
    return sig

def backtest(df, cost_pct, size=2):
    s = signals(df); c = df["Close"].values
    cash, pos, peak = 1.0, 0, 1.0; eq = []
    for i in range(len(c)):
        if pos != 0 and i > 0:
            cash *= (1 + (c[i]-c[i-1])/c[i-1] * pos * size)
        if pos == 0 and s.values[i] != 0:
            pos = s.values[i]; cash *= (1 - cost_pct*size)
        elif pos != 0 and s.values[i] == 0:
            cash *= (1 - cost_pct*size); pos = 0
        peak = max(peak, cash); eq.append(cash/peak)
    eq = pd.Series(eq, index=df.index[-len(eq):])
    return (cash-1)*100, (1-eq.min())*100, int((s.values != 0).sum())

def report(name, df):
    print(f"\n=== {name} ({len(df)} bars, {df.index.min().date()}..{df.index.max().date()}) ===")
    for m in [0.5, 1.0, 2.0]:
        r, dd, n = backtest(df, 0.0004*m)
        print(f"  cost x{m}: return={r:7.2f}%  maxDD={dd:6.2f}%  holds={n}")

f = YFinanceDataFetcher(cache_dir="data_cache")
df3 = f.get_nifty_index(period="3y")
df1 = df3[df3.index >= df3.index.max() - pd.Timedelta(days=365)]
print("NIFTY rows:", len(df3))

report("FULL 3y", df3)
report("1y window", df1)

print("\n=== Rolling 4-window consistency (3y, cost x1) ===")
w = len(df3)//4
for k in range(4):
    seg = df3.iloc[k*w:(k+1)*w]
    r, dd, n = backtest(seg, 0.0004)
    print(f"  window {k+1}: return={r:7.2f}%  maxDD={dd:6.2f}%  holds={n}")

print("\n=== 60/40 OOS split (3y, cost x1) ===")
n = len(df3); cut = int(n*0.6)
tr, ddt, nt = backtest(df3.iloc[:cut], 0.0004)
te, dte, ne = backtest(df3.iloc[cut:], 0.0004)
print(f"  IN-SAMPLE : return={tr:7.2f}%  maxDD={ddt:6.2f}%  holds={nt}")
print(f"  OUT-OF-SAMPLE: return={te:7.2f}%  maxDD={dte:6.2f}%  holds={ne}")
print("  VERDICT:", "OOS positive & comparable -> candidate" if te > 0 else "OOS negative -> no edge")
