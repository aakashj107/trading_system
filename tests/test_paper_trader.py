"""
A2 — Paper trader (long + short) functional smoke test.

paper_trader.py must be a complete, importable module that processes
BUY/SELL and SHORT/COVER signals and reports a long/short P&L breakdown.
"""
import numpy as np
import pytest

from paper_trader import PaperTrader
from strategies import BearTrendShort, Signal


def test_paper_trader_importable_and_trades(ohlc_sharp_crash):
    """PaperTrader should process SHORT/COVER from a bear strategy."""
    out = BearTrendShort().generate_signals(ohlc_sharp_crash)
    pt = PaperTrader(initial_capital=500000)
    atr = PaperTrader.calculate_atr(ohlc_sharp_crash)
    for i in range(len(out)):
        row = out.iloc[i]
        if row["Signal"] != 0:
            pt.process_signal(
                "TEST", int(row["Signal"]), row["Close"],
                str(out.index[i].date()), "BearTrendShort",
                atr=atr.iloc[i],
            )
    assert len(pt.trade_history) > 0, "paper trader made no trades"
    summary = pt.get_performance_summary()
    # short P&L must be a finite number
    assert "short_pnl" in summary
    assert np.isfinite(summary["short_pnl"])


def test_paper_trader_long_roundtrip(ohlc_uptrend):
    """A BUY then SELL should register exactly one long trade."""
    pt = PaperTrader(initial_capital=500000)
    pt.process_signal("X", Signal.BUY.value, 100.0, "2023-01-02",
                      "test", atr=2.0)
    pt.process_signal("X", Signal.SELL.value, 110.0, "2023-01-03",
                      "test", atr=2.0)
    assert len(pt.trade_history) == 1
    assert pt.trade_history[0].pnl > 0  # bought 100 sold 110
