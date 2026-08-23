"""Smoke test: transpile a dual-MA crossover Pine strategy and backtest it."""
import pandas as pd, numpy as np
from pine_transpiler import transpile

PINE = '''
//@version=5
strategy("DMA Cross", overlay=true)
fastLen = input.int(10, title="Fast")
slowLen = input.int(30, title="Slow")
fastMA = ta.ema(close, fastLen)
slowMA = ta.sma(close, slowLen)
longCond  = ta.cross(fastMA, slowMA)
shortCond = ta.crossunder(fastMA, slowMA)
if (longCond)
    strategy.entry("Long", strategy.long)
if (shortCond)
    strategy.close("Long")
'''

inst, parsed = transpile(PINE)
print("Parsed title:", parsed.title, "| inputs:", parsed.inputs)
print("smells/unsupported:", parsed.smells, parsed.unsupported)

# synthetic data
np.random.seed(1)
n = 300
px = 100 + np.cumsum(np.random.randn(n) * 0.5)
idx = pd.date_range("2023-01-01", periods=n, freq="D")
df = pd.DataFrame({
    "Open": px + np.random.randn(n)*0.1, "High": px + 0.5,
    "Low": px - 0.5, "Close": px, "Volume": np.random.randint(1e5,1e6,n)},
    index=idx)

out = inst.generate_signals(df)
print("Signal counts:", out["Signal"].value_counts().to_dict())

from backtester import Backtester
bt = Backtester(initial_capital=100000, position_size_pct=0.95, stop_loss_pct=0.05)
res = bt.run_backtest(inst, out, symbol="TEST")
print("Metrics:", {k: res["metrics"].get(k) for k in
      ("total_trades","win_rate","total_return_pct","sharpe_ratio","max_drawdown_pct")})
