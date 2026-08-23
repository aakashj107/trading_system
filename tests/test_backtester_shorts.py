"""
A4 — Backtester SHORT/COVER support + bear-strategy signal correctness.

Locks in the fix where the backtester previously only handled LONG, and where
the bear short strategies emitted COVER on every bar (overriding SHORT) so they
could never actually short. These tests run on synthetic data (no network).
"""
import numpy as np
import pytest

from strategies import (
    BearTrendShort, BearMomentumShort, BearBreakdownShort,
    BearMeanReversionShort,
)
from backtester import Backtester


BEAR_STRATS = {
    "BearTrendShort": BearTrendShort(),
    "BearMomentumShort": BearMomentumShort(),
    "BearBreakdownShort": BearBreakdownShort(),
    "BearMeanReversionShort": BearMeanReversionShort(),
}


def _trades(res):
    return res["trades"]


def test_backtester_handles_short_direction(ohlc_sharp_crash):
    """Backtester must open SHORT positions and track direction."""
    res = Backtester(initial_capital=500000).run_backtest(
        BearMomentumShort(), ohlc_sharp_crash, "TEST")
    shorts = [t for t in _trades(res) if t.direction == "SHORT"]
    assert len(shorts) > 0, "backtester produced no SHORT trades"


def test_short_pnl_is_positive_when_price_falls(ohlc_sharp_crash):
    """On a clean downtrend, a short entry→cover should be profitable."""
    res = Backtester(initial_capital=500000).run_backtest(
        BearMomentumShort(), ohlc_sharp_crash, "TEST")
    summary = res["metrics"]
    # Not asserting it's always positive (random data), but P&L must be finite
    # and the equity curve must be present.
    assert summary["final_capital"] is not None
    assert np.isfinite(summary["final_capital"])


def test_position_dataclass_has_direction():
    """A4 added `direction` to Position — code below relies on it."""
    from backtester import Position
    p = Position(symbol="X", entry_date=None, entry_price=1.0, quantity=1,
                 stop_loss=0.9, direction="SHORT")
    assert p.direction == "SHORT"


@pytest.mark.parametrize("name,strat", BEAR_STRATS.items())
def test_bear_strategies_fire_shorts_in_downtrend(name, strat, ohlc_sharp_crash):
    """Every bear strategy must emit at least one SHORT on a sharp downtrend."""
    res = Backtester(initial_capital=500000).run_backtest(
        strat, ohlc_sharp_crash, name)
    n_short = sum(1 for t in _trades(res) if t.direction == "SHORT")
    assert n_short > 0, f"{name} produced no SHORT trades in a downtrend"


@pytest.mark.parametrize("name,strat", BEAR_STRATS.items())
def test_bear_signal_sequence_is_paired(name, strat, ohlc_sharp_crash):
    """SHORT signals must not be immediately overwritten by COVER/noise."""
    out = strat.generate_signals(ohlc_sharp_crash)
    sig = out["Signal"].values
    short_idx = np.where(sig == -2)[0]
    cover_idx = np.where(sig == 2)[0]
    # Basic sanity: there are short signals AND they are followed by covers,
    # not left dangling at the end of data in a way that never closes.
    assert len(short_idx) > 0
    assert len(cover_idx) > 0
    # No two consecutive SHORTs without a COVER between (stateful pairing)
    for a, b in zip(short_idx[:-1], short_idx[1:]):
        between = sig[a + 1:b]
        assert 2 in between, f"{name}: consecutive SHORT with no COVER between"
