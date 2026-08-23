"""
A1 + A3 — Risk data unification and dead-code removal.

A1: risk_manager.RiskManager must use CORRELATION_GROUPS from config.py
    (single source of truth), not a stale internal duplicate.
A3: the duplicate CCIPrepReady class must be gone; CCIPrepStrategy remains.
"""
import pytest

from config import CORRELATION_GROUPS
from risk_manager import RiskManager
from strategies import CCIPrepStrategy


def test_risk_manager_uses_config_correlation_groups():
    """A1: RiskManager must see the same (correct) groups as config.py."""
    # The stale internal dict referenced 'TATAMOTORS.NS' and lacked ADANI.
    assert "TATAMOTORS.NS" not in _flatten(CORRELATION_GROUPS)
    assert "ADANI" in CORRELATION_GROUPS
    # RiskManager.check_pre_trade returns (allowed, reason, details)
    rm = RiskManager(max_correlated=2)  # max 2 correlated positions per group
    # With 2 private banks already held, a 3rd must be blocked.
    res = rm.check_pre_trade("ICICIBANK.NS", "BUY", ["HDFCBANK.NS", "KOTAKBANK.NS"])
    assert len(res) == 3
    ok, reason, _details = res
    assert ok is False, "should block 3rd private-bank position"


def _flatten(groups):
    out = []
    for members in groups.values():
        out.extend(members)
    return out


def test_cciprep_ready_dead_class_removed():
    """A3: the duplicate CCIPrepReady class must no longer exist."""
    import strategies
    assert not hasattr(strategies, "CCIPrepReady")
    assert hasattr(strategies, "CCIPrepStrategy")


def test_cciprep_strategy_registered_and_runs(ohlc_uptrend):
    """CCIPrepStrategy (the surviving one) must still generate signals."""
    out = CCIPrepStrategy().generate_signals(ohlc_uptrend)
    assert "Signal" in out.columns
    assert "CCI" in out.columns
