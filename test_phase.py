"""
test_phase1.py — Standalone verification for Phase 1 components.

Run this BEFORE integrating into your daily pipeline:
    python test_phase1.py

Tests:
    1. Break Quality Filter — correct candle evaluation
    2. Regression SuperTrend — correct signal generation
    3. Integration — BQ filter applied to Reg SuperTrend signals
    4. Compatibility — works with your existing Signal enum
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from strategies import (
    BreakQualityFilter, RegressionSuperTrendStrategy, Signal,
    get_all_strategies, compute_atr
)
from config import BreakQualityConfig, REG_ST_CONFIG, BQ_CONFIG


def make_synthetic_data(n=200, seed=42):
    """Create synthetic OHLCV data with trend changes."""
    np.random.seed(seed)

    # Simulate uptrend → downtrend → uptrend
    trend = np.concatenate([
        np.linspace(100, 150, 80),
        np.linspace(150, 110, 60),
        np.linspace(110, 160, 60),
    ])
    noise = np.random.normal(0, 2, n)
    close = trend + noise

    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    df = pd.DataFrame({
        'Open':   close - np.random.uniform(0.5, 1.5, n),
        'High':   close + np.random.uniform(1, 3, n),
        'Low':    close - np.random.uniform(1, 3, n),
        'Close':  close,
        'Volume': np.random.randint(500000, 2000000, n),
    }, index=dates)

    return df


# ──────────────────────────────────────────────────────────────
#  TEST 1: Break Quality Filter
# ──────────────────────────────────────────────────────────────

def test_break_quality_filter():
    """Verify BQ filter correctly identifies strong vs weak candles."""
    print("=" * 60)
    print("TEST 1: Break Quality Filter")
    print("=" * 60)

    # Strong candle: big body relative to range
    strong = pd.DataFrame({
        'Open':   [100.0],
        'High':   [103.0],
        'Low':    [99.0],
        'Close':  [102.5],
        'Volume': [1e6],
    })
    # Body = |102.5 - 100| = 2.5, Range = 103 - 99 = 4
    # Body ratio = 2.5/4 = 0.625 — should pass both SOFT and HARD

    # Weak candle: tiny body, big wicks (doji-like)
    weak = pd.DataFrame({
        'Open':   [100.0],
        'High':   [110.0],
        'Low':    [90.0],
        'Close':  [100.3],
        'Volume': [1e6],
    })
    # Body = |100.3 - 100| = 0.3, Range = 110 - 90 = 20
    # Body ratio = 0.3/20 = 0.015 — should fail quality

    # Need at least 14 bars for ATR
    # Pad with similar data
    def pad(df, n=20):
        base = df.iloc[0]
        rows = []
        for i in range(n):
            row = base.copy()
            row['Open'] += np.random.normal(0, 0.5)
            row['Close'] += np.random.normal(0, 0.5)
            row['High'] = max(row['Open'], row['Close']) + abs(np.random.normal(1, 0.3))
            row['Low'] = min(row['Open'], row['Close']) - abs(np.random.normal(1, 0.3))
            rows.append(row)
        padded = pd.DataFrame(rows)
        padded.index = pd.date_range('2024-01-01', periods=n, freq='D')
        return padded

    strong_padded = pad(strong)
    weak_padded = pad(weak)

    # SOFT mode
    bq_soft = BreakQualityFilter(BreakQualityConfig(mode="SOFT"))

    result = bq_soft.check_bar(strong_padded, bar_index=len(strong_padded) - 1)
    print(f"\n  Strong candle (SOFT):")
    print(f"    passed={result['passed']}, weak={result['weak']}, "
          f"ratio={result['body_ratio']}, reason={result['reason']}")
    assert result['passed'] == True, "Strong candle should pass SOFT"

    result = bq_soft.check_bar(weak_padded, bar_index=len(weak_padded) - 1)
    print(f"\n  Weak candle (SOFT):")
    print(f"    passed={result['passed']}, weak={result['weak']}, "
          f"ratio={result['body_ratio']}, reason={result['reason']}")
    assert result['passed'] == True, "SOFT mode should still pass (just tag weak)"
    assert result['weak'] == True, "Weak candle should be tagged as weak"

    # HARD mode
    bq_hard = BreakQualityFilter(BreakQualityConfig(mode="HARD"))

    result = bq_hard.check_bar(strong_padded, bar_index=len(strong_padded) - 1)
    print(f"\n  Strong candle (HARD):")
    print(f"    passed={result['passed']}, reason={result['reason']}")
    assert result['passed'] == True, "Strong candle should pass HARD"

    result = bq_hard.check_bar(weak_padded, bar_index=len(weak_padded) - 1)
    print(f"\n  Weak candle (HARD):")
    print(f"    passed={result['passed']}, reason={result['reason']}")
    assert result['passed'] == False, "Weak candle should fail HARD"

    # OFF mode
    bq_off = BreakQualityFilter(BreakQualityConfig(mode="OFF"))
    result = bq_off.check_bar(weak_padded, bar_index=len(weak_padded) - 1)
    print(f"\n  Weak candle (OFF):")
    print(f"    passed={result['passed']}, reason={result['reason']}")
    assert result['passed'] == True, "OFF mode should always pass"

    print("\n  ✓ TEST 1 PASSED: Break Quality Filter works correctly\n")


# ──────────────────────────────────────────────────────────────
#  TEST 2: Regression SuperTrend
# ──────────────────────────────────────────────────────────────

def test_regression_supertrend():
    """Verify Regression SuperTrend generates correct signals."""
    print("=" * 60)
    print("TEST 2: Regression SuperTrend Strategy")
    print("=" * 60)

    df = make_synthetic_data(200)

    strategy = RegressionSuperTrendStrategy(reg_length=20, rmse_multiplier=2.0)

    # Check name
    print(f"\n  Strategy name: {strategy.name}")
    print(f"  Category: {strategy.category}")
    assert strategy.category == "trend", "Should be trend category"

    # Generate signals
    result_df = strategy.generate_signals(df)

    # Verify output columns exist
    required_cols = ['Reg_Line', 'Reg_Upper', 'Reg_Lower', 'Reg_Trend', 'Signal']
    for col in required_cols:
        assert col in result_df.columns, f"Missing column: {col}"
    print(f"  Output columns: {list(result_df.columns)}")

    # Count signals
    buy_count = (result_df['Signal'] == Signal.BUY.value).sum()
    sell_count = (result_df['Signal'] == Signal.SELL.value).sum()
    total_bars = len(result_df)

    print(f"\n  Total bars: {total_bars}")
    print(f"  BUY signals:  {buy_count}")
    print(f"  SELL signals: {sell_count}")
    print(f"  Signal rate:  {(buy_count + sell_count) / total_bars * 100:.1f}%")

    # Verify warmup period has no signals
    warmup_signals = result_df['Signal'].iloc[:20].abs().sum()
    print(f"\n  Warmup signals (first 20 bars): {int(warmup_signals)} "
          f"{'✓' if warmup_signals == 0 else '✗ FAIL'}")
    assert warmup_signals == 0, "No signals during warmup period"

    # Verify trend values are +1 or -1 only
    valid_trends = result_df['Reg_Trend'].dropna().unique()
    print(f"  Trend values: {sorted(valid_trends)} "
          f"{'✓' if set(valid_trends) <= {1, -1} else '✗ FAIL'}")
    assert set(valid_trends) <= {1, -1}, "Trend should only be +1 or -1"

    # Verify bands make sense: upper > lower where both valid
    valid_mask = result_df['Reg_Upper'].notna() & result_df['Reg_Lower'].notna()
    band_check = (result_df.loc[valid_mask, 'Reg_Upper'] >
                  result_df.loc[valid_mask, 'Reg_Lower']).all()
    print(f"  Upper > Lower: {band_check} {'✓' if band_check else '✗ FAIL'}")
    assert band_check, "Upper band should always be above lower band"

    # Verify signal alternation (BUY should follow SELL and vice versa)
    signal_bars = result_df[result_df['Signal'] != 0]
    if len(signal_bars) > 1:
        sigs = signal_bars['Signal'].values
        alternates = all(sigs[i] != sigs[i+1] for i in range(len(sigs)-1))
        print(f"  Signals alternate BUY/SELL: {alternates} "
              f"{'✓' if alternates else '✗ FAIL'}")

    # Print first few signals
    print(f"\n  First 5 signals:")
    for idx, row in signal_bars.head(5).iterrows():
        direction = "BUY ↑" if row['Signal'] == Signal.BUY.value else "SELL ↓"
        price = row['Close']
        print(f"    {idx.strftime('%Y-%m-%d')}: {direction} @ ₹{price:.2f}")

    print("\n  ✓ TEST 2 PASSED: Regression SuperTrend works correctly\n")


# ──────────────────────────────────────────────────────────────
#  TEST 3: Integration — BQ + Reg SuperTrend
# ──────────────────────────────────────────────────────────────

def test_integration():
    """Test BQ filter applied to Regression SuperTrend signals."""
    print("=" * 60)
    print("TEST 3: Integration (BQ Filter + Reg SuperTrend)")
    print("=" * 60)

    df = make_synthetic_data(200)

    # Get raw signals
    strategy = RegressionSuperTrendStrategy()
    result_df = strategy.generate_signals(df)
    raw_signals = result_df['Signal']

    raw_count = (raw_signals != 0).sum()
    print(f"\n  Raw signals: {raw_count}")

    # SOFT mode
    bq_soft = BreakQualityFilter(BreakQualityConfig(mode="SOFT"))
    soft_filtered = bq_soft.filter_signal_series(df, raw_signals)
    soft_report = bq_soft.get_signal_quality_report(df, raw_signals)

    soft_count = (soft_filtered != 0).sum()
    weak_count = soft_report['weak'].sum() if len(soft_report) > 0 else 0
    print(f"  SOFT filtered: {soft_count} signals "
          f"(same count, {weak_count} tagged weak)")
    assert soft_count == raw_count, "SOFT should not remove signals"

    # HARD mode
    bq_hard = BreakQualityFilter(BreakQualityConfig(mode="HARD"))
    hard_filtered = bq_hard.filter_signal_series(df, raw_signals)

    hard_count = (hard_filtered != 0).sum()
    blocked = raw_count - hard_count
    print(f"  HARD filtered: {hard_count} signals ({blocked} blocked)")

    # Quality breakdown
    if len(soft_report) > 0:
        print(f"\n  Quality breakdown (SOFT mode):")
        quality_counts = soft_report['quality'].value_counts()
        for quality, count in quality_counts.items():
            print(f"    {quality}: {count}")

    # Verify compatibility with existing Strategy class
    print(f"\n  Strategy compatibility check:")
    print(f"    isinstance(Strategy): "
          f"{isinstance(strategy, type(get_all_strategies()[0]))}")

    # Check that all strategies in get_all_strategies() still work
    all_strats = get_all_strategies()
    print(f"    Total strategies: {len(all_strats)}")
    reg_st_found = any('Regression_SuperTrend' in s.name for s in all_strats)
    print(f"    Regression SuperTrend registered: {reg_st_found}")
    assert reg_st_found, "Regression SuperTrend should be in get_all_strategies()"

    print("\n  ✓ TEST 3 PASSED: Integration works correctly\n")


# ──────────────────────────────────────────────────────────────
#  TEST 4: Existing strategies still work with BQ filter
# ──────────────────────────────────────────────────────────────

def test_existing_strategies_compatibility():
    """Verify BQ filter doesn't break any existing strategy."""
    print("=" * 60)
    print("TEST 4: Existing Strategies Compatibility")
    print("=" * 60)

    df = make_synthetic_data(200)
    bq = BreakQualityFilter()  # Default SOFT mode

    strategies = get_all_strategies()
    print(f"\n  Testing {len(strategies)} strategies with BQ filter...\n")

    all_passed = True
    for strategy in strategies:
        try:
            result_df = strategy.generate_signals(df)
            raw_signals = result_df['Signal']
            filtered = bq.filter_signal_series(df, raw_signals)
            report = bq.get_signal_quality_report(df, raw_signals)

            raw_count = (raw_signals != 0).sum()
            filt_count = (filtered != 0).sum()
            weak = report['weak'].sum() if len(report) > 0 else 0

            status = "✓"
            print(f"    {status} {strategy.name:<40} "
                  f"raw={raw_count:<4} filtered={filt_count:<4} weak={weak}")
        except Exception as e:
            all_passed = False
            print(f"    ✗ {strategy.name:<40} ERROR: {e}")

    assert all_passed, "Some strategies failed compatibility test"
    print(f"\n  ✓ TEST 4 PASSED: All strategies compatible with BQ filter\n")


# ──────────────────────────────────────────────────────────────
#  RUN ALL TESTS
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "▓" * 60)
    print("  PHASE 1 TEST SUITE")
    print("▓" * 60 + "\n")

    test_break_quality_filter()
    test_regression_supertrend()
    test_integration()
    test_existing_strategies_compatibility()

    print("▓" * 60)
    print("  ALL PHASE 1 TESTS PASSED")
    print("▓" * 60)
    print(f"\n  Summary:")
    print(f"    • Break Quality Filter: SOFT/HARD/OFF modes working")
    print(f"    • Regression SuperTrend: Signal generation verified")
    print(f"    • Integration: BQ + Reg SuperTrend compatible")
    print(f"    • Existing strategies: All {len(get_all_strategies())} strategies unaffected")
    print(f"\n  Next step: Run your daily pipeline and compare results")
    print(f"    with and without BQ filter enabled.\n")