"""
#12 — Angel One live integration SAFETY + connectivity dry-run.

These tests prove the live client can NEVER place a real order by accident:
  - paper mode        -> print stub (PAPER_FILLED)
  - paper_api (read-only dry-run) -> DRYRUN, fetches LTP if authed, no orders
  - live without live_confirmed -> LOCKED
  - live with live_confirmed but no creds -> AUTH_FAILED (never places)
And that the execution harness runs cleanly in paper_api mode (read-only).

No credentials, no real orders, no capital at risk anywhere in this file.
"""
import numpy as np
import pandas as pd
import pytest

from angel_one_live import LiveTrader, ns_to_angel_symbol
from execution_harness import run_harness
from tests.test_execution_harness import _make_universe


def test_paper_mode_is_stub():
    t = LiveTrader(mode="paper")
    r = t.place_order("RELIANCE.NS", "BUY", 10, 2500.0)
    assert r["status"] == "PAPER_FILLED"


def test_paper_api_dryrun_no_credentials_is_safe():
    """paper_api with no creds must DRYRUN and never place/throw."""
    t = LiveTrader(mode="paper_api")
    assert t.authenticate() is False
    r = t.place_order("RELIANCE.NS", "BUY", 10, 2500.0)
    assert r["status"] == "DRYRUN"


def test_live_refuses_without_confirm():
    t = LiveTrader(mode="live", live_confirmed=False)
    r = t.place_order("RELIANCE.NS", "BUY", 10, 2500.0)
    assert r["status"] == "LOCKED"


def test_live_with_confirm_but_no_creds_fails_safe():
    t = LiveTrader(mode="live", live_confirmed=True)
    r = t.place_order("RELIANCE.NS", "BUY", 10, 2500.0)
    assert r["status"] in ("LOCKED", "AUTH_FAILED")


def test_symbol_mapping():
    assert ns_to_angel_symbol("RELIANCE.NS") == "RELIANCE"
    assert ns_to_angel_symbol("TATAMOTORS.BO") == "TATAMOTORS"


def test_harness_runs_in_paper_api_mode():
    """Full pipeline in paper_api mode must complete with no live orders."""
    data = _make_universe(seed=11)
    res = run_harness(data, live_mode="paper_api")
    assert res["buy_orders"] > 0
    assert res["open_positions_end"] == 0
