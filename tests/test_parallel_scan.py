"""
B2/B3 — Parallel data loading + parallel scanner.

Locks in:
  - scanner.scan() accepts a data_cache dict and parallelizes per-stock fetch+scan
  - the parallel path must produce IDENTICAL results to the sequential fetch path
"""
import pandas as pd
import pytest

from scanner import SignalScanner
from strategies import get_regime_strategies


class _FakeFetcher:
    """Returns cached DataFrames without network."""
    def __init__(self, cache):
        self._cache = cache

    def get_historical_data(self, symbol, period=None):
        return self._cache.get(symbol)


def _build_scanner(universe_cache, fetcher):
    sc = SignalScanner(
        symbols=list(universe_cache.keys()),
        strategies=get_regime_strategies("SIDESWAYS"),
        regime="SIDESWAYS",
    )
    sc.fetcher = fetcher
    return sc


def test_parallel_scan_matches_sequential(universe_cache):
    """The parallel (data_cache) path must equal the sequential (fetch) path."""
    sc = _build_scanner(universe_cache, _FakeFetcher(universe_cache))

    seq = sc.scan()                       # sequential, via fetcher
    par = sc.scan(data_cache=universe_cache, max_workers=8)  # parallel, cached

    seq_keys = set(zip(seq["Symbol"], seq["Strategy"], seq["Signal"])) \
        if not seq.empty else set()
    par_keys = set(zip(par["Symbol"], par["Strategy"], par["Signal"])) \
        if not par.empty else set()

    assert seq_keys == par_keys, "parallel scan diverged from sequential"
    assert len(par) > 0, "scan produced no signals on synthetic data"


def test_scan_uses_cache_no_refetch(universe_cache):
    """When data_cache is passed, scanner must not hit the network fetcher."""

    class _RecordingFetcher:
        calls = 0
        def get_historical_data(self, symbol, period=None):
            type(self).calls += 1
            return universe_cache.get(symbol)

    sc = _build_scanner(universe_cache, _RecordingFetcher())
    sc.scan(data_cache=universe_cache, max_workers=4)
    # No network calls should occur when everything is already cached.
    assert _RecordingFetcher.calls == 0
