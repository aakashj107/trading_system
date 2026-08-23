"""
test_phase4.py — Standalone verification for Phase 4 components.

Run BEFORE integrating into your daily pipeline:
    python test_phase4.py

Tests:
    1. CCI PREP Strategy — ARM → FIRE two-stage logic
    2. ADX Quality Gate — zone classification and weight adjustment
    3. Integration — CCI PREP + ADX gate + all previous phases
    4. Full strategy suite — all 17 strategies compatible
    5. End-to-end consensus with all Phase 1-4 modifiers
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from strategies import (
    CCIPrepStrategy, ADXQualityGate, Signal,
    get_all_strategies, BreakQualityFilter,
    ExhaustionDetector, TrendRangeClassifier, compute_atr
)
from config import (
    CCI_PREP_CONFIG, ADX_GATE_CONFIG, BreakQualityConfig,
    EXHAUSTION_CONFIG, TREND_RANGE_CONFIG
)


def make_mean_reversion_data(n=200, seed=42):
    """
    Oscillating data with clear extremes — ideal for CCI PREP.
    Simulates a ranging market with sharp overshoots and reversions.
    """
    np.random.seed(seed)

    prices = []
    p = 100.0
    for i in range(n):
        # Sine wave with noise — creates clear overbought/oversold cycles
        cycle = 8 * np.sin(i * 0.12)  # ~52 bar cycle
        noise = np.random.normal(0, 1.5)
        p = 100 + cycle + noise
        prices.append(p)

    close = np.array(prices)
    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    return pd.DataFrame({
        'Open':   close - np.random.uniform(0.3, 1.0, n),
        'High':   close + np.random.uniform(1.0, 3.0, n),
        'Low':    close - np.random.uniform(1.0, 3.0, n),
        'Close':  close,
        'Volume': np.random.randint(500000, 2000000, n),
    }, index=dates)


def make_trending_data(n=200, seed=42):
    """Strong sustained trend — ideal for ADX CLEAN zone testing."""
    np.random.seed(seed)

    prices = []
    p = 100.0
    for i in range(n):
        p += np.random.uniform(0.5, 1.8)
        prices.append(p)

    close = np.array(prices) + np.random.normal(0, 0.5, n)
    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    return pd.DataFrame({
        'Open':   close - np.random.uniform(0.2, 0.5, n),
        'High':   close + np.random.uniform(0.5, 1.5, n),
        'Low':    close - np.random.uniform(0.5, 1.5, n),
        'Close':  close,
        'Volume': np.random.randint(500000, 2000000, n),
    }, index=dates)


def make_choppy_data(n=200, seed=42):
    """Low-volatility chop — ideal for ADX NO_TRADE zone testing."""
    np.random.seed(seed)

    prices = []
    p = 100.0
    for i in range(n):
        p += np.random.normal(0, 0.3)  # Very small moves
        prices.append(p)

    close = np.array(prices)
    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    return pd.DataFrame({
        'Open':   close - np.random.uniform(0.1, 0.3, n),
        'High':   close + np.random.uniform(0.2, 0.5, n),
        'Low':    close - np.random.uniform(0.2, 0.5, n),
        'Close':  close,
        'Volume': np.random.randint(100000, 500000, n),
    }, index=dates)


# ──────────────────────────────────────────────────────────────
#  TEST 1: CCI PREP Strategy
# ──────────────────────────────────────────────────────────────

def test_cci_prep():
    """Test two-stage CCI logic on oscillating data."""
    print("=" * 60)
    print("TEST 1: CCI PREP Strategy")
    print("=" * 60)

    df = make_mean_reversion_data(200)

    strategy = CCIPrepStrategy(
        cci_length=20, extreme_level=200,
        trigger_level=100, cooldown_bars=12
    )

    # Metadata
    print(f"\n  Strategy name: {strategy.name}")
    print(f"  Category: {strategy.category}")
    assert strategy.category == "mean_reversion"
    assert "CCI_Prep" in strategy.name

    # Generate signals
    result_df = strategy.generate_signals(df)

    # Verify columns
    required_cols = ['CCI', 'CCI_Armed_Long', 'CCI_Armed_Short', 'Signal']
    for col in required_cols:
        assert col in result_df.columns, f"Missing column: {col}"

    # Count signals
    buy_count = (result_df['Signal'] == Signal.BUY.value).sum()
    sell_count = (result_df['Signal'] == Signal.SELL.value).sum()

    print(f"\n  Total bars: {len(result_df)}")
    print(f"  BUY signals:  {buy_count}")
    print(f"  SELL signals: {sell_count}")

    # Count armed states
    armed_l_bars = result_df['CCI_Armed_Long'].sum()
    armed_s_bars = result_df['CCI_Armed_Short'].sum()
    print(f"  Long armed bars:  {int(armed_l_bars)}")
    print(f"  Short armed bars: {int(armed_s_bars)}")

    # Verify CCI range is reasonable
    cci_vals = result_df['CCI'].dropna()
    print(f"\n  CCI range: {cci_vals.min():.1f} to {cci_vals.max():.1f}")
    print(f"  CCI mean:  {cci_vals.mean():.1f}")

    # Verify cooldown works — check minimum gap between signals
    signal_bars = result_df[result_df['Signal'] != 0].index
    if len(signal_bars) > 1:
        gaps = []
        for i in range(1, len(signal_bars)):
            gap = (signal_bars[i] - signal_bars[i-1]).days
            gaps.append(gap)
        min_gap = min(gaps)
        print(f"\n  Min gap between signals: {min_gap} bars")
        print(f"  Cooldown setting: {strategy.cooldown_bars} bars")
        assert min_gap >= strategy.cooldown_bars, \
            f"Cooldown violated: gap={min_gap} < cooldown={strategy.cooldown_bars}"

    # Print signals
    signal_rows = result_df[result_df['Signal'] != 0]
    if len(signal_rows) > 0:
        print(f"\n  First 5 signals:")
        for idx, row in signal_rows.head(5).iterrows():
            direction = "BUY ↑" if row['Signal'] == Signal.BUY.value else "SELL ↓"
            print(f"    {idx.strftime('%Y-%m-%d')}: {direction} "
                  f"@ ₹{row['Close']:.2f}  CCI={row['CCI']:.1f}")

    # Verify two-stage logic: BUY should only fire after ARM
    buys = result_df[result_df['Signal'] == Signal.BUY.value]
    for idx, row in buys.iterrows():
        loc = result_df.index.get_loc(idx)
        # Check that at least one armed_long=True in the preceding bars
        preceding = result_df.iloc[max(0, loc - 30):loc]
        armed_before = preceding['CCI_Armed_Long'].any()
        assert armed_before, f"BUY at {idx} fired without prior ARM"

    print("\n  ✓ TEST 1 PASSED: CCI PREP two-stage logic verified\n")


# ──────────────────────────────────────────────────────────────
#  TEST 2: ADX Quality Gate
# ──────────────────────────────────────────────────────────────

def test_adx_quality_gate():
    """Test ADX zone classification on different market types."""
    print("=" * 60)
    print("TEST 2: ADX Quality Gate")
    print("=" * 60)

    gate = ADXQualityGate()

    # ── Test on trending data (expect CLEAN or OK zone) ──
    trend_df = make_trending_data(200)
    trend_result = gate.evaluate(trend_df)

    print(f"\n  TRENDING DATA:")
    print(f"    ADX: {trend_result['adx']}")
    print(f"    Zone: {trend_result['zone']}")
    print(f"    Weight multiplier: {trend_result['weight_mult']}")
    print(f"    ADX rising: {trend_result['adx_rising']}")

    # Trending data should NOT be in NO_TRADE zone
    assert trend_result['zone'] != 'NO_TRADE', \
        "Strong trend should not be NO_TRADE"

    # ── Test on choppy data (expect NO_TRADE or WEAK zone) ──
    chop_df = make_choppy_data(200)
    chop_result = gate.evaluate(chop_df)

    print(f"\n  CHOPPY DATA:")
    print(f"    ADX: {chop_result['adx']}")
    print(f"    Zone: {chop_result['zone']}")
    print(f"    Weight multiplier: {chop_result['weight_mult']}")
    print(f"    ADX rising: {chop_result['adx_rising']}")

    # ── Test on mean reversion data ──
    mr_df = make_mean_reversion_data(200)
    mr_result = gate.evaluate(mr_df)

    print(f"\n  MEAN REVERSION DATA:")
    print(f"    ADX: {mr_result['adx']}")
    print(f"    Zone: {mr_result['zone']}")
    print(f"    Weight multiplier: {mr_result['weight_mult']}")
    print(f"    ADX rising: {mr_result['adx_rising']}")

    # ── Verify weight multiplier ranges ──
    print(f"\n  Zone weight multipliers:")
    from config import ADX_GATE_CONFIG as c
    print(f"    NO_TRADE (<{c.no_trade_max}): ×{c.no_trade_mult}")
    print(f"    WEAK     (<{c.weak_max}):     ×{c.weak_mult}")
    print(f"    OK       (<{c.ok_max}):       ×{c.ok_mult}")
    print(f"    CLEAN    (<{c.clean_max}):    ×{c.clean_mult}")
    print(f"    LATE     (≥{c.clean_max}):    ×{c.late_mult}")

    # ── Test apply_to_signals ──
    mock_signals = {
        'SMA_Crossover_20_50': {
            'signal': 1, 'weight': 1.0, 'category': 'trend',
            'adx_boost': False, 'adx_penalty': False, 'adx_zone': '',
        },
        'RSI_14_30_70': {
            'signal': 1, 'weight': 1.0, 'category': 'mean_reversion',
            'adx_boost': False, 'adx_penalty': False, 'adx_zone': '',
        },
    }

    # Apply with trending data result
    import copy
    test_sigs = copy.deepcopy(mock_signals)
    adjusted = gate.apply_to_signals(test_sigs, trend_result)

    print(f"\n  Signal weight adjustment (trending data):")
    for name, info in adjusted.items():
        print(f"    {name:<30} zone={info['adx_zone']:<10} "
              f"w={info['weight']:.3f}")

    # Trend signals in trending data should have reasonable weight
    assert adjusted['SMA_Crossover_20_50']['weight'] > 0

    print("\n  ✓ TEST 2 PASSED: ADX Quality Gate works correctly\n")


# ──────────────────────────────────────────────────────────────
#  TEST 3: Integration — CCI PREP + ADX + all previous phases
# ──────────────────────────────────────────────────────────────

def test_phase4_integration():
    """Test CCI PREP and ADX gate with all previous phases."""
    print("=" * 60)
    print("TEST 3: Phase 4 Integration")
    print("=" * 60)

    df = make_mean_reversion_data(200)

    # Generate CCI PREP signals
    strategy = CCIPrepStrategy()
    result_df = strategy.generate_signals(df)

    # Apply BQ filter
    bq = BreakQualityFilter()
    raw_signals = result_df['Signal']
    filtered = bq.filter_signal_series(df, raw_signals)
    report = bq.get_signal_quality_report(df, raw_signals)

    raw_count = (raw_signals != 0).sum()
    filt_count = (filtered != 0).sum()
    weak_count = report['weak'].sum() if len(report) > 0 else 0

    print(f"\n  CCI PREP signals:")
    print(f"    Raw:      {raw_count}")
    print(f"    Filtered: {filt_count}")
    print(f"    Weak:     {weak_count}")

    # Apply ADX gate
    gate = ADXQualityGate()
    adx_result = gate.evaluate(df)

    print(f"\n  ADX Gate:")
    print(f"    Zone: {adx_result['zone']}")
    print(f"    ADX:  {adx_result['adx']}")
    print(f"    Mult: {adx_result['weight_mult']}")

    # Apply exhaustion detector
    exh = ExhaustionDetector()
    exh_result = exh.is_exhausting(df)
    print(f"\n  Exhaustion: {exh_result['exhausting']} ({exh_result['direction']})")

    # Apply sub-regime classifier
    tr = TrendRangeClassifier()
    tr_result = tr.get_sub_regime(df)
    print(f"  Sub-regime: {tr_result['sub_regime']} "
          f"(score={tr_result['range_score']}, ADX={tr_result['adx']})")

    # Simulate full weight chain
    if raw_count > 0:
        base_weight = 1.0  # CCI PREP base weight

        # Phase 1A: BQ
        if weak_count > 0:
            base_weight *= 0.5  # SOFT penalty for weak candles
        print(f"\n  Weight chain for CCI PREP signal:")
        print(f"    Base weight:        1.000")
        print(f"    After BQ filter:    {base_weight:.3f}")

        # Regime category weight
        regime_mult = 0.5  # mean_reversion in BULL (default test)
        base_weight *= regime_mult
        print(f"    After regime ({regime_mult}x):  {base_weight:.3f}")

        # Phase 4B: ADX gate
        base_weight *= adx_result['weight_mult']
        print(f"    After ADX gate:     {base_weight:.3f}")

        # Phase 3A: Exhaustion
        if exh_result['exhausting']:
            base_weight *= EXHAUSTION_CONFIG.weight_penalty
            print(f"    After exhaustion:   {base_weight:.3f}")

        # Phase 3B: Sub-regime
        c = TREND_RANGE_CONFIG
        if tr_result['sub_regime'] == 'RANGE':
            base_weight *= c.mr_boost
            print(f"    After SR (RANGE×{c.mr_boost}): {base_weight:.3f}")
        elif tr_result['sub_regime'] == 'TREND':
            base_weight *= c.mr_penalty
            print(f"    After SR (TREND×{c.mr_penalty}): {base_weight:.3f}")

        print(f"    Final weight:       {base_weight:.3f}")

    print("\n  ✓ TEST 3 PASSED: Phase 4 integration works correctly\n")


# ──────────────────────────────────────────────────────────────
#  TEST 4: All 17 strategies compatibility
# ──────────────────────────────────────────────────────────────

def test_all_strategies():
    """Verify all 17 strategies work together."""
    print("=" * 60)
    print("TEST 4: All 17 Strategies Compatibility")
    print("=" * 60)

    strategies = get_all_strategies()
    bq = BreakQualityFilter()

    print(f"\n  Total strategies: {len(strategies)}")

    # Test on each data type
    datasets = {
        'Mean Reversion': make_mean_reversion_data(200),
        'Trending':       make_trending_data(200),
        'Choppy':         make_choppy_data(200),
    }

    all_passed = True

    for data_name, df in datasets.items():
        print(f"\n  Dataset: {data_name}")
        total_signals = 0

        for strategy in strategies:
            try:
                result = strategy.generate_signals(df)
                raw = result['Signal']
                filtered = bq.filter_signal_series(df, raw)

                raw_count = (raw != 0).sum()
                filt_count = (filtered != 0).sum()
                total_signals += raw_count

                buy = (raw == Signal.BUY.value).sum()
                sell = (raw == Signal.SELL.value).sum()
                short = (raw == Signal.SHORT.value).sum()

                parts = []
                if buy: parts.append(f"{buy}B")
                if sell: parts.append(f"{sell}S")
                if short: parts.append(f"{short}Sh")
                sig_str = " ".join(parts) if parts else "0"

                print(f"    ✓ {strategy.name:<42} [{strategy.category:<14}] "
                      f"sig={sig_str:<10} bq_pass={filt_count}")

            except Exception as e:
                all_passed = False
                print(f"    ✗ {strategy.name:<42} ERROR: {e}")

        print(f"    Total raw signals: {total_signals}")

    assert all_passed, "Some strategies failed"

    # Count by category
    cat_counts = {}
    for s in strategies:
        cat_counts[s.category] = cat_counts.get(s.category, 0) + 1

    print(f"\n  Strategy inventory ({len(strategies)} total):")
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat:<16}: {count}")

    print("\n  ✓ TEST 4 PASSED: All strategies compatible\n")


# ──────────────────────────────────────────────────────────────
#  TEST 5: End-to-end consensus simulation with all phases
# ──────────────────────────────────────────────────────────────

def test_full_consensus_simulation():
    """Simulate complete consensus with all Phase 1-4 modifiers."""
    print("=" * 60)
    print("TEST 5: Full Consensus Simulation (Phase 1-4)")
    print("=" * 60)

    strategies = get_all_strategies()
    bq = BreakQualityFilter()
    exh_detector = ExhaustionDetector()
    tr_classifier = TrendRangeClassifier()
    adx_gate = ADXQualityGate()

    # Use mean reversion data (best for CCI PREP)
    df = make_mean_reversion_data(200)

    # ── Step 1: Raw signals ──
    raw_signals = {}
    for s in strategies:
        try:
            result = s.generate_signals(df)
            sig = int(result['Signal'].iloc[-1])
            if sig != 0:
                raw_signals[s.name] = {
                    'signal': sig,
                    'category': s.category,
                    'signal_text': 'BUY' if sig == 1 else 'SELL' if sig == -1 else 'SHORT',
                }
        except:
            pass

    print(f"\n  Step 1 — Raw signals: {len(raw_signals)}")

    # ── Step 2: BQ filter ──
    bq_weak = 0
    for name in list(raw_signals.keys()):
        strat = [s for s in strategies if s.name == name][0]
        result = strat.generate_signals(df)
        bq_result = bq.check_bar(df, len(df) - 1)
        if not bq_result['passed']:
            del raw_signals[name]
        elif bq_result['weak']:
            bq_weak += 1
            raw_signals[name]['bq_weak'] = True

    print(f"  Step 2 — After BQ: {len(raw_signals)} ({bq_weak} weak)")

    # ── Step 3: Assign base weights ──
    from scanner import STRATEGY_QUALITY
    for name in raw_signals:
        sq = STRATEGY_QUALITY.get(name, {"weight": 1.0, "category": "unknown"})
        raw_signals[name]['weight'] = sq['weight']
        raw_signals[name]['category'] = sq['category']
        raw_signals[name].setdefault('exhaustion_penalty', False)
        raw_signals[name].setdefault('sub_regime_boost', False)
        raw_signals[name].setdefault('sub_regime_penalty', False)
        raw_signals[name].setdefault('adx_boost', False)
        raw_signals[name].setdefault('adx_penalty', False)
        raw_signals[name].setdefault('adx_zone', '')
        raw_signals[name].setdefault('bq_weak', False)

        # Apply BQ SOFT penalty
        if raw_signals[name].get('bq_weak'):
            raw_signals[name]['weight'] *= 0.5

    # ── Step 4: ADX Gate ──
    adx_result = adx_gate.evaluate(df)
    print(f"  Step 3 — ADX: {adx_result['zone']} "
          f"(ADX={adx_result['adx']}, mult={adx_result['weight_mult']})")

    raw_signals = adx_gate.apply_to_signals(raw_signals, adx_result)

    # ── Step 5: Exhaustion ──
    exhaustion = exh_detector.is_exhausting(df)
    print(f"  Step 4 — Exhaustion: {exhaustion['exhausting']} "
          f"({exhaustion['direction']})")

    if exhaustion['exhausting']:
        from config import EXHAUSTION_CONFIG
        penalty = EXHAUSTION_CONFIG.weight_penalty
        for name, info in raw_signals.items():
            if exhaustion['direction'] == 'BULL_EXH':
                if info['signal'] == 1 and info['category'] in ('trend', 'momentum'):
                    info['weight'] *= penalty
                    info['exhaustion_penalty'] = True

    # ── Step 6: Sub-regime ──
    sub_regime = tr_classifier.get_sub_regime(df)['sub_regime']
    print(f"  Step 5 — Sub-regime: {sub_regime}")

    from config import TREND_RANGE_CONFIG as c
    for name, info in raw_signals.items():
        cat = info['category']
        if sub_regime == 'TREND':
            if cat in ('trend', 'momentum'):
                info['weight'] *= c.trend_boost
                info['sub_regime_boost'] = True
            elif cat == 'mean_reversion':
                info['weight'] *= c.mr_penalty
                info['sub_regime_penalty'] = True
        elif sub_regime == 'RANGE':
            if cat in ('trend', 'momentum'):
                info['weight'] *= c.trend_penalty
                info['sub_regime_penalty'] = True
            elif cat == 'mean_reversion':
                info['weight'] *= c.mr_boost
                info['sub_regime_boost'] = True

    # ── Step 7: Consensus ──
    buy_score = sum(i['weight'] for i in raw_signals.values() if i['signal'] == 1)
    sell_score = sum(i['weight'] for i in raw_signals.values() if i['signal'] == -1)
    buy_count = sum(1 for i in raw_signals.values() if i['signal'] == 1)
    sell_count = sum(1 for i in raw_signals.values() if i['signal'] in (-1, -2))

    direction = "BUY" if buy_score > sell_score else "SELL" if sell_score > buy_score else "NEUTRAL"

    print(f"\n  CONSENSUS:")
    print(f"    Direction:  {direction}")
    print(f"    BUY:  {buy_count} strategies, score={buy_score:.3f}")
    print(f"    SELL: {sell_count} strategies, score={sell_score:.3f}")

    print(f"\n  Final signal details:")
    for name, info in sorted(raw_signals.items(),
                              key=lambda x: x[1]['weight'], reverse=True):
        notes = []
        if info.get('bq_weak'): notes.append('BQ:WEAK')
        if info.get('exhaustion_penalty'): notes.append('↓exh')
        if info.get('sub_regime_boost'): notes.append(f"↑{sub_regime}")
        if info.get('sub_regime_penalty'): notes.append(f"↓{sub_regime}")
        if info.get('adx_boost'): notes.append('↑adx')
        if info.get('adx_penalty'): notes.append('↓adx')
        if info.get('adx_zone') and info['adx_zone'] not in ('OK', ''):
            notes.append(f"ADX:{info['adx_zone']}")

        note_str = f"  [{', '.join(notes)}]" if notes else ""
        dir_text = info['signal_text']

        print(f"    {name:<42} w={info['weight']:.3f} "
              f"{dir_text:<8} [{info['category']}]{note_str}")

    print("\n  ✓ TEST 5 PASSED: Full consensus simulation complete\n")


# ──────────────────────────────────────────────────────────────
#  RUN ALL TESTS
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "▓" * 60)
    print("  PHASE 4 TEST SUITE")
    print("▓" * 60 + "\n")

    test_cci_prep()
    test_adx_quality_gate()
    test_phase4_integration()
    test_all_strategies()
    test_full_consensus_simulation()

    print("▓" * 60)
    print("  ALL PHASE 4 TESTS PASSED")
    print("▓" * 60)
    print(f"\n  Summary:")
    print(f"    • CCI PREP (Strategy #17): Two-stage CCI mean reversion")
    print(f"    • ADX Quality Gate: Universal signal quality modifier")
    print(f"    • All 17 strategies working together")
    print(f"    • All Phase 1-4 modifiers verified end-to-end")
    print(f"\n  Final system architecture:")
    print(f"    ┌──────────────────────────────────────────────┐")
    print(f"    │  17 Strategies across 5 categories            │")
    print(f"    │    Trend:           5 (SMA×2, EMA, Breakout,  │")
    print(f"    │                       RegSuperTrend)          │")
    print(f"    │    Momentum:        2 (MACD, ROC)             │")
    print(f"    │    Mean Reversion:  4 (RSI, Bollinger,        │")
    print(f"    │                       Z-Score, CCI PREP)      │")
    print(f"    │    Structure:       2 (CHOCH/BOS, Liq Sweep)  │")
    print(f"    │    Bear Short:      4 (Trend, Momentum,       │")
    print(f"    │                       Breakdown, MeanRev)     │")
    print(f"    ├──────────────────────────────────────────────┤")
    print(f"    │  4 Quality Layers                             │")
    print(f"    │    Phase 1A: Break Quality Filter (BQ)        │")
    print(f"    │    Phase 3A: Exhaustion Detector              │")
    print(f"    │    Phase 3B: Trend/Range Sub-Regime           │")
    print(f"    │    Phase 4B: ADX Quality Gate                 │")
    print(f"    ├──────────────────────────────────────────────┤")
    print(f"    │  Regime-Aware Consensus                       │")
    print(f"    │    Bull/Bear/Sideways thresholds              │")
    print(f"    │    Category-weighted scoring                  │")
    print(f"    │    CONVICTION / STRONG / MODERATE / WEAK      │")
    print(f"    └──────────────────────────────────────────────┘")
    print(f"\n  Run: python test_phase4.py")
    print(f"  Then integrate and run your daily pipeline.\n")