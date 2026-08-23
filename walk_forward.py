"""
walk_forward.py — Walk-forward parameter optimization for the trading system.

Goal: find whether ANY strategy has tunable edge, and prove it with a STRICT
forward out-of-sample test (no look-ahead, no re-optimization on test).

Method (purposely conservative to avoid overfitting):
  - Load ~3y of data per symbol.
  - Walk-forward folds: train on window T, test on the NEXT non-overlapping window.
      e.g. fold1: train [0,18m) test [18m,30m); fold2: train [6m,24m) test [24m,36m)
        (expanding-or-rolling — here ROLLING of fixed length).
  - On each fold's TRAIN set: grid-search each strategy's params to maximize
      Sharpe (risk-adjusted, less overfit-prone than raw return), requiring
      a minimum trade count (>=8) so we don't pick lucky 1-trade configs.
  - Evaluate the chosen params on the fold's TEST set (never seen in selection).
  - Aggregate OOS results across folds and across symbols.

Report: does the OPTIMIZED param set beat the DEFAULT params on OOS?
If optimized OOS > 0 and optimized OOS > default OOS, the strategy has edge.

Reuses short_backtester.run_short_backtest (trend gate + ATR stop) so findings
are consistent with the earlier analysis.
"""
import warnings
warnings.filterwarnings("ignore")

import itertools
import pandas as pd
import numpy as np
from strategies import (SMACrossover, EMACrossover, RSIStrategy, MACDStrategy,
                        BollingerBands, BreakoutStrategy, MeanReversion,
                        MomentumStrategy, RegressionSuperTrendStrategy,
                        CCIPrepStrategy)
from strategies_pine import (RTSAdaptiveTrail, StructureEntry, CombinedRTSStructure,
                         OrderBlockStrategy)
from data_fetcher import YFinanceDataFetcher
from short_backtester import (run_short_backtest, metrics, UNIVERSE,
                              INIT_CAP, POS_PCT)


# ── Parameter grids per strategy class ──
# Kept small on purpose: each extra param multiplies runtime.
GRIDS = {
    SMACrossover: {"fast_period": [10, 20, 50], "slow_period": [50, 100, 200]},
    EMACrossover: {"fast_period": [8, 12, 26], "slow_period": [21, 26, 50]},
    RSIStrategy: {"period": [10, 14, 21], "oversold": [25, 30, 35], "overbought": [65, 70, 75]},
    MACDStrategy: {"fast": [8, 12], "slow": [21, 26], "signal_period": [9]},
    BollingerBands: {"period": [15, 20, 25], "num_std": [1.5, 2.0, 2.5]},
    BreakoutStrategy: {"lookback": [10, 20, 30], "volume_multiplier": [1.2, 1.5, 2.0]},
    MeanReversion: {"period": [20, 50], "entry_std": [1.5, 2.0], "exit_std": [0.5]},
    MomentumStrategy: {"roc_period": [10, 20, 30], "volume_period": [20]},
    RegressionSuperTrendStrategy: {"ema_period": [20, 30], "bb_period": [20],
                                    "sma_period": [50]},
    CCIPrepStrategy: {"cci_length": [14, 20, 30], "extreme_level": [100, 200]},
    RTSAdaptiveTrail: {"ma_length": [21], "rsi_length": [13], "t2": [2.0]},
    StructureEntry: {"reg_length": [20], "rmse_mult": [2.0]},
    CombinedRTSStructure: {},
    OrderBlockStrategy: {"atr_len": [14], "pivot_len": [5, 8], "disp_mult": [0.3, 0.5, 0.8],
                         "min_grade": [4.0, 5.0, 6.0], "use_vol": [True, False]},
}


def _instantiate(cls, params):
    if not params:
        return cls()
    return cls(**params)


def grid_search(strategy_cls, train_dict, min_trades=8):
    """Return best (params, train_sharpe) over the param grid on train_dict."""
    best = None  # (sharpe, params)
    keys = list(GRIDS[strategy_cls].keys())
    for vals in itertools.product(*[GRIDS[strategy_cls][k] for k in keys]):
        params = dict(zip(keys, vals))
        try:
            st = _instantiate(strategy_cls, params)
        except Exception:
            continue
        all_t, all_eq = [], []
        for sym, d in train_dict.items():
            t, e = run_short_backtest(st, d, allow_short=True,
                                      trend_gate=True, use_atr_stop=True, atr_mult=2.0)
            all_t.extend(t)
            if not e.empty:
                all_eq.append(e)
        if not all_t:
            continue
        m = metrics(all_t, pd.concat(all_eq))
        if m.get("total_trades", 0) < min_trades:
            continue
        sharpe = m.get("sharpe_ratio", -99)
        if best is None or sharpe > best[0]:
            best = (sharpe, params, m)
    return best


def evaluate_params(strategy_cls, params, test_dict):
    st = _instantiate(strategy_cls, params)
    all_t, all_eq = [], []
    for sym, d in test_dict.items():
        t, e = run_short_backtest(st, d, allow_short=True,
                                  trend_gate=True, use_atr_stop=True, atr_mult=2.0)
        all_t.extend(t)
        if not e.empty:
            all_eq.append(e)
    if not all_t:
        return {"total_trades": 0}
    return metrics(all_t, pd.concat(all_eq))


def make_folds(data, n_folds=3, train_bars=450, test_bars=300):
    """Rolling walk-forward folds over each symbol's series.

    Uses a single rolling window of length (train_bars + test_bars) and steps
    it forward by test_bars each fold. Clamps to available data.
    """
    total = len(data)
    win = train_bars + test_bars
    folds = []
    if win > total:
        # Not enough data for one full fold: use a single fold covering all.
        if total >= 80:
            split = int(total * 0.6)
            folds.append((data.iloc[:split], data.iloc[split:]))
        return folds
    start = 0
    while start + win <= total and len(folds) < n_folds:
        train = data.iloc[start:start + train_bars]
        test = data.iloc[start + train_bars:start + win]
        if len(train) >= 60 and len(test) >= 40:
            folds.append((train, test))
        start += test_bars
    return folds


def main():
    fetcher = YFinanceDataFetcher()
    print("Loading 3y universe data...")
    data = {}
    for s in UNIVERSE:
        try:
            d = fetcher.get_historical_data(s, period="3y")
            if d is not None and not d.empty and len(d) > 200:
                data[s] = d
        except Exception:
            pass
    print(f"  got {len(data)} symbols, ~{int(np.mean([len(d) for d in data.values()]))} bars each")

    # Representative subset used for optimization + OOS (keeps runtime bounded
    # while remaining a meaningful, multi-name out-of-sample test).
    OPT_UNIVERSE = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS",
                    "INFY.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS"]
    opt_data = {s: data[s] for s in OPT_UNIVERSE if s in data}
    print(f"  optimizing on {len(opt_data)} symbols")

    # Build per-symbol folds once (3y -> single forward 60/40 fold)
    symbol_folds = {sym: make_folds(d) for sym, d in opt_data.items()}
    n_folds = min(len(f) for f in symbol_folds.values()) if symbol_folds else 0
    print(f"  folds per symbol: {n_folds}")

    summary = []
    for cls in GRIDS:
        name = cls.__name__
        fold_oos = []   # optimized OOS return% per fold
        fold_def = []   # default OOS return% per fold
        for f in range(n_folds):
            train_dict = {sym: symbol_folds[sym][f][0] for sym in opt_data}
            test_dict = {sym: symbol_folds[sym][f][1] for sym in opt_data}
            # optimize on train
            best = grid_search(cls, train_dict)
            if best is None:
                continue
            _, params, _ = best
            # strictly forward OOS test
            oos = evaluate_params(cls, params, test_dict)
            oos_ret = oos.get("total_return_pct", 0.0)
            oos_trades = oos.get("total_trades", 0)
            # default for comparison
            default = evaluate_params(cls, {}, test_dict)
            def_ret = default.get("total_return_pct", 0.0)
            fold_oos.append(oos_ret)
            fold_def.append(def_ret)
            print(f"  {name:28} fold{f}  optRet={oos_ret:7.2f}% (n={oos_trades:3d})  "
                  f"defRet={def_ret:7.2f}%  params={params}", flush=True)

        if fold_oos:
            avg_oos = float(np.mean(fold_oos))
            avg_def = float(np.mean(fold_def))
            n_pos = sum(1 for r in fold_oos if r > 0)
            beats = sum(1 for o, d in zip(fold_oos, fold_def) if o > d)
            summary.append((name, avg_oos, avg_def, n_pos, len(fold_oos), beats))

    print("\n" + "=" * 95)
    print("WALK-FORWARD SUMMARY (avg across folds, OOS forward-tested)")
    print("=" * 95)
    print(f"{'Strategy':28}{'OptOOS%':>10}{'DefOOS%':>10}{'PosFolds':>10}{'BeatsDef':>10}")
    for name, avg_oos, avg_def, n_pos, nf, beats in sorted(summary, key=lambda r: -r[1]):
        print(f"{name:28}{avg_oos:>10.2f}{avg_def:>10.2f}{f'{n_pos}/{nf}':>10}{f'{beats}/{nf}':>10}")

    winners = [s for s in summary if s[1] > 0]
    print(f"\nStrategies with positive avg OOS (optimized): {len(winners)} / {len(summary)}")
    for s in winners:
        print(f"  {s[0]}: OptOOS={s[1]:.2f}%  DefOOS={s[2]:.2f}%  "
              f"positive in {s[3]}/{s[4]} folds, beats default in {s[5]}/{s[4]} folds")

    print("\nDONE.")


if __name__ == "__main__":
    main()
