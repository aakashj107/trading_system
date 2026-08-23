"""End-to-end test of transpile + real-data backtest (no UI)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from pine_web.pine_parser import parse, summarize
from pine_web.pine_transpiler import transpile, TranspileError
from pine_web.run_backtest import run_pine_backtest, metrics_summary
from data_fetcher import YFinanceDataFetcher

SAMPLE = '''//@version=5
strategy("RSI Pullback", overlay=true)
rsiLen = input.int(14, title="RSI Length")
rsiLevel = input.int(30, title="RSI Oversold")
maLen = input.int(50, title="MA Length")
rsiVal = ta.rsi(close, rsiLen)
ma = ta.sma(close, maLen)
longCond  = ta.crossunder(rsiVal, rsiLevel) and close > ma
shortCond = ta.cross(ma, close)
if (longCond)
    strategy.entry("Long", strategy.long)
if (shortCond)
    strategy.close("Long")
'''

parsed = parse(SAMPLE)
print(summarize(parsed))

strat, p2 = transpile(SAMPLE)
print("Transpiled OK:", type(strat).__name__, "| name:", strat.name)

df = YFinanceDataFetcher().get_historical_data("RELIANCE.NS", period="3y", use_cache=True)
if df.empty:
    print("DATA EMPTY — abort")
    raise SystemExit(1)
print("Data rows:", len(df), df.index[0].date(), "->", df.index[-1].date())

res = run_pine_backtest(strat, df, symbol="RELIANCE.NS")
m = metrics_summary(res)
print("METRICS:")
for k, v in m.items():
    print(f"  {k}: {v}")
print("Trades:", len(res["trades"]))
