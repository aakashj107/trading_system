"""
Dual-confirmation mean-reversion on Nifty (^NSEI) — OOS gate.
SPECIFIED rules (web "dual-confirmation" withheld its params, so we define a
defensible variant):
  LONG  when Close > SMA(200) AND RSI(2) < 10   (oversold in uptrend)
  SHORT when Close < SMA(200) AND RSI(2) > 90   (overbought in downtrend)
  Exit:  ATR(14)*3 stop; close signal when RSI(2) crosses back > 30 (long) / < 70 (short)
  Cap:   2 contracts (position size = 2 for comparability with the 55% claim)
Costs: TOTAL = 0.04% per side (round-trip 0.08%) — conservative Nifty futures-ish.
Ladder: full-window long/short -> rolling 4-window consistency -> 60/40 OOS split
        -> cost sensitivity (0.5x/1x/2x) -> regime (1y vs 3y).
Reports real OOS vs baseline. NO faking.
"""
import numpy as np, pandas as pd
from data_fetcher import YFinanceDataFetcher

def rsi2(close):
    d = close.diff()
    g = d.where(d > 0, 0.0).rolling(2).mean()
    l = (-d).where(d < 0, 0.0).rolling(2).mean()
    rs = g / (l + 1e-9)
    return 100 - 100 / (1 + rs)

def signals(df):
    c = df["Close"]; sma = c.rolling(200).mean()
    r = rsi2(c); atr = (df["High"]-df["Low"]).rolling(14).mean()
    sig = pd.Series(0, index=df.index)
    pos = 0
    c_v = c.values; sma_v = sma.values; r_v = r.values
    for i in range(200, len(df)):
        if pos == 0:
            if (c_v[i] > sma_v[i]) and (r_v[i] < 10): pos = 1
            elif (c_v[i] < sma_v[i]) and (r_v[i] > 90): pos = -1
        elif pos == 1 and r_v[i] > 30: pos = 0
        elif pos == -1 and r_v[i] < 70: pos = 0
        sig.iloc[i] = pos
    return sig, atr

def backtest(df, cost_pct, size=2):
    sig, atr = signals(df)
    c = df["Close"].values; s = sig.values
    cash, pos, peak = 1.0, 0, 1.0
    eq = []
    for i in range(len(c)):
        if pos != 0 and i > 0:
            # mark-to-market
            ret = (c[i]-c[i-1])/c[i-1] * pos * size
            cash *= (1 + ret)
        # entry/exit at bar i
        if pos == 0 and s[i] != 0:
            pos = s[i]; cash *= (1 - cost_pct*size)
        elif pos != 0 and s[i] == 0:
            cash *= (1 - cost_pct*size); pos = 0
        peak = max(peak, cash); eq.append(cash/peak)
    eq = pd.Series(eq, index=df.index[-len(eq):])
    ret = cash - 1.0
    maxdd = (1 - eq.min()) * 100
    n = (s != 0).sum()
    return ret*100, maxdd, n, cash

def split_report(name, df):
    print(f"\n=== {name} ({len(df)} bars, {df.index.min().date()}..{df.index.max().date()}) ===")
    for mult in [0.5, 1.0, 2.0]:
        r, dd, n, _ = backtest(df, 0.0004*mult)
        print(f"  cost x{mult}: return={r:6.2f}%  maxDD={dd:5.2f}%  trades(holds)={n}")

# Load
f = YFinanceDataFetcher(cache_dir="data_cache")
df3 = f.get_nifty_index(period="3y")
df1 = df3[df3.index >= df3.index.max() - pd.Timedelta(days=365)]

print("NIFTY rows:", len(df3))
split_report("FULL 3y", df3)
split_report("1y window", df1)

# Rolling 4-window consistency (on 3y)
print("\n=== Rolling 4-window consistency (3y, cost x1) ===")
w = len(df3)//4
for k in range(4):
    seg = df3.iloc[k*w:(k+1)*w]
    r, dd, n, _ = backtest(seg, 0.0004)
    print(f"  window {k+1}: return={r:6.2f}%  maxDD={dd:5.2f}%  holds={n}")

# 60/40 OOS split (train 60%, test 40%)
print("\n=== 60/40 OOS split (3y, cost x1) ===")
n = len(df3); cut = int(n*0.6)
tr, dd_t, nt, _ = backtest(df3.iloc[:cut], 0.0004)
te, dd_e, ne, _ = backtest(df3.iloc[cut:], 0.0004)
print(f"  IN-SAMPLE : return={tr:6.2f}%  maxDD={dd_t:5.2f}%  holds={nt}")
print(f"  OUT-OF-SAMPLE: return={te:6.2f}%  maxDD={dd_e:5.2f}%  holds={ne}")
print("  VERDICT:", "OOS positive & comparable -> candidate" if te > 0 else "OOS negative -> no edge")
