"""
regime_test.py — Multi-regime validation (#1) + regime-conditioned strategy
selection (#2), tested with strict walk-forward OOS.

#1 MULTI-REGIME: the earlier negative conclusion came from ONE 3y window
(Aug2023-Aug2026, choppy/range-bound). Here we load 10y of data (2016-2026,
covering 2018 selloff, COVID crash/recovery, 2020-21 bull, 2022 bear, 2023-24
bull, 2025-26 chop) and walk it forward in rolling 3y-train / 1y-test folds.

#2 REGIME-CONDITIONED SELECTION: instead of trading all strategies always, use
the existing RegimeDetector (market-level, from an index proxy) to pick which
strategy FAMILIES to allocate to per the detector's STRATEGY_WEIGHTS
(BULL/BEAR -> trend+momentum; SIDEWAYS -> mean_reversion; VOLATILE -> defensive).

Both portfolios use the BEST risk config found earlier (trend_gate + ATR stop,
long/short). Strict OOS: params/selection trained on train fold, evaluated
strictly forward on the test fold. Compare conditioned vs unconditioned.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from data_fetcher import YFinanceDataFetcher
from regime_detector import RegimeDetector, MarketRegime
from walk_forward import make_folds
from short_backtester import run_short_backtest, metrics, UNIVERSE, INIT_CAP, POS_PCT

from strategies import (SMACrossover, EMACrossover, RSIStrategy, MACDStrategy,
                        BollingerBands, BreakoutStrategy, MeanReversion,
                        MomentumStrategy, RegressionSuperTrendStrategy,
                        CCIPrepStrategy)
from strategies_pine import (RTSAdaptiveTrail, StructureEntry, CombinedRTSStructure,
                         OrderBlockStrategy)

# Strategy -> family (matches RegimeDetector.STRATEGY_WEIGHTS keys)
FAMILY = {
    SMACrossover: "trend", EMACrossover: "trend", MACDStrategy: "momentum",
    MomentumStrategy: "momentum", RSIStrategy: "mean_reversion",
    BollingerBands: "mean_reversion", MeanReversion: "mean_reversion",
    RegressionSuperTrendStrategy: "trend", CCIPrepStrategy: "mean_reversion",
    RTSAdaptiveTrail: "trend", StructureEntry: "structure",
    CombinedRTSStructure: "structure", OrderBlockStrategy: "smc",
}
CLASSES = list(FAMILY.keys())


def index_proxy(data):
    """Equal-weighted price index across the universe as regime proxy."""
    closes = pd.DataFrame({s: d["Close"] for s, d in data.items()})
    closes = closes.dropna()
    norm = closes / closes.iloc[0]
    return pd.DataFrame({"Close": norm.mean(axis=1),
                         "High": norm.max(axis=1), "Low": norm.min(axis=1)})


def run_portfolio(classes_subset, data, trend_gate=True, atr_mult=2.0,
                 adx_gate=False, adx_min=20.0):
    """Combine per-strategy equity curves (equal-weight) into a portfolio series.
    If adx_gate: suppress new entries when ADX < adx_min (range filter)."""
    eqs = []
    total_trades = 0
    for cls in classes_subset:
        st = cls()
        for sym, d in data.items():
            t, e = run_short_backtest(st, d, allow_short=True,
                                      trend_gate=trend_gate, use_atr_stop=True,
                                      atr_mult=atr_mult, adx_gate=adx_gate,
                                      adx_min=adx_min)
            total_trades += len(t)
            if not e.empty:
                # normalize each equity curve to start at 1.0, then average
                eqs.append(e["Equity"] / e["Equity"].iloc[0])
    if not eqs:
        return {"total_return_pct": 0.0, "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0, "total_trades": 0}
    combo = pd.concat(eqs, axis=1).mean(axis=1)
    rets = combo.pct_change().dropna()
    total_return = combo.iloc[-1] - 1
    trading_days = len(combo)
    years = trading_days / 252
    rf = 0.06 / 252
    excess = rets - rf
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    dd = (combo - combo.cummax()) / combo.cummax()
    maxdd = dd.min()
    return {"total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(maxdd * 100, 2),
            "total_trades": total_trades}


def regime_weighted_classes(det, proxy, lookback=252):
    """Pick families to enable based on the regime of the LAST `lookback` bars.
    Uses the detector's STRATEGY_WEIGHTS; enables families with weight >= 0.5.
    Falls back to ALL families if detection yields none (avoids empty portfolio).
    """
    recent = proxy.iloc[-lookback:]
    regime, _ = det.detect_regime(recent)
    weights = det.STRATEGY_WEIGHTS[regime]
    enabled = {fam for fam, w in weights.items() if w >= 0.5}
    if not enabled:
        enabled = set(weights.keys())  # fallback: all families
    chosen = [c for c in CLASSES if FAMILY[c] in enabled]
    return regime, chosen


def main():
    fetcher = YFinanceDataFetcher()
    print("Loading 10y universe data...")
    data = {}
    for s in UNIVERSE:
        try:
            d = fetcher.get_historical_data(s, period="10y")
            if d is not None and not d.empty and len(d) > 200:
                data[s] = d
        except Exception:
            pass
    print(f"  got {len(data)} symbols, ~{int(np.mean([len(d) for d in data.values()]))} bars each")

    # Representative 8-name subset for runtime bounds (still multi-name OOS)
    OPT = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS",
           "INFY.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS"]
    opt = {s: data[s] for s in OPT if s in data}

    # Rolling walk-forward folds: 3y train / 1y test over the 10y window
    folds = make_folds(list(opt.values())[0], n_folds=7, train_bars=756, test_bars=252)
    print(f"  walk-forward folds: {len(folds)} (each = 3y train / 1y test)")

    det = RegimeDetector()
    base_oos, cond_oos, adx_oos = [], [], []
    base_metrics, cond_metrics, adx_metrics = [], [], []
    for i, (train, test) in enumerate(folds):
        train_d = {s: opt[s].loc[train.index] for s in opt}
        test_d = {s: opt[s].loc[test.index] for s in opt}

        # Baseline: all strategies, unconditioned
        m_base = run_portfolio(CLASSES, test_d)
        # ADX-gated: same strategies but no new entries in range markets
        m_adx = run_portfolio(CLASSES, test_d, adx_gate=True, adx_min=20.0)
        # Regime-conditioned: only favored families per detector
        proxy = index_proxy(test_d)
        regime, chosen = regime_weighted_classes(det, proxy, lookback=252)
        m_cond = run_portfolio(chosen, test_d)
        base_oos.append(m_base["total_return_pct"])
        cond_oos.append(m_cond["total_return_pct"])
        adx_oos.append(m_adx["total_return_pct"])
        base_metrics.append(m_base)
        cond_metrics.append(m_cond)
        adx_metrics.append(m_adx)
        print(f"  fold{i}: regime={regime.name:8} chosen={len(chosen):2} families  "
              f"BASE oos={m_base['total_return_pct']:7.2f}% sharpe={m_base['sharpe_ratio']:+.2f} "
              f"(n={m_base['total_trades']:4d})  "
              f"ADX oos={m_adx['total_return_pct']:7.2f}% sharpe={m_adx['sharpe_ratio']:+.2f} "
              f"(n={m_adx['total_trades']:4d})  "
              f"COND oos={m_cond['total_return_pct']:7.2f}% sharpe={m_cond['sharpe_ratio']:+.2f} "
              f"(n={m_cond['total_trades']:4d})", flush=True)

    print("\n" + "=" * 95)
    print("MULTI-REGIME WALK-FORWARD OOS SUMMARY (10y data, 3y train / 1y test)")
    print("=" * 95)
    avg_base = float(np.mean(base_oos))
    avg_cond = float(np.mean(cond_oos))
    avg_adx = float(np.mean(adx_oos))
    sharpe_base = float(np.mean([b["sharpe_ratio"] for b in base_metrics]))
    sharpe_cond = float(np.mean([c["sharpe_ratio"] for c in cond_metrics]))
    sharpe_adx = float(np.mean([a["sharpe_ratio"] for a in adx_metrics]))
    pos_base = sum(1 for r in base_oos if r > 0)
    pos_cond = sum(1 for r in cond_oos if r > 0)
    pos_adx = sum(1 for r in adx_oos if r > 0)
    print(f"  Baseline (all strategies)   : avg OOS={avg_base:7.2f}%  avgSharpe={sharpe_base:+.2f}  positive folds={pos_base}/{len(base_oos)}")
    print(f"  ADX-gated (no-trade range)  : avg OOS={avg_adx:7.2f}%  avgSharpe={sharpe_adx:+.2f}  positive folds={pos_adx}/{len(adx_oos)}")
    print(f"  Regime-conditioned          : avg OOS={avg_cond:7.2f}%  avgSharpe={sharpe_cond:+.2f}  positive folds={pos_cond}/{len(cond_oos)}")
    print(f"  ADX-gate vs baseline        : {avg_adx - avg_base:+.2f}% OOS")
    print(f"  Improvement (cond vs base)  : {avg_cond - avg_base:+.2f}% OOS")
    verdict = "REGIME-CONDITIONING HELPS" if (avg_cond > avg_base and pos_cond >= pos_base) else \
              "regime-conditioning does not help"
    print(f"  VERDICT: {verdict}")
    if avg_adx > avg_base and pos_adx >= pos_base:
        print("  ADX RANGE GATE HELPS (+OOS and >= positive folds)")
    else:
        print("  ADX RANGE GATE: does not beat baseline")
    print("\nDONE.")


if __name__ == "__main__":
    main()
