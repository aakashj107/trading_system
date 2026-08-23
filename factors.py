"""
factors.py — Portfolio-level factors with real empirical backing.

Why this file exists:
    Every strategy in strategies.py / strategies_pine.py is a SINGLE-NAME
    time-series signal (trend / momentum / mean-reversion on one symbol).
    The walk-forward study proved none of them has a persistent edge on this
    universe/data. This module tests a STRUCTURALLY DIFFERENT idea:
    CROSS-SECTIONAL MOMENTUM — rank names by trailing return and hold the
    winners long / losers short as a portfolio. This is the Jegadeesh-Titman
    factor, one of the few with durable academic evidence, and it cannot be
    expressed as a per-name time-series signal.

It is evaluated with the SAME strict guardrails as walk_forward.py:
    optimize lookback/top_n on in-sample 60%, test STRICTLY forward on the
    unseen 40%, require a minimum number of rebalances, report OOS return +
    Sharpe + maxDD. No look-ahead, no re-optimization on test.

Reuses: data_fetcher, short_backtester.make_folds, INIT_CAP.
"""
import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
from data_fetcher import YFinanceDataFetcher
from walk_forward import make_folds
from short_backtester import UNIVERSE, INIT_CAP, POS_PCT


def _portfolio_metrics(equity: pd.Series):
    if equity is None or len(equity) < 5:
        return {"total_return_pct": 0.0, "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0, "cagr_pct": 0.0}
    rets = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    trading_days = len(equity)
    years = trading_days / 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    rf = 0.06 / 252
    excess = rets - rf
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    dd = (equity - equity.cummax()) / equity.cummax()
    maxdd = dd.min()
    return {"total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(maxdd * 100, 2),
            "cagr_pct": round(cagr * 100, 2)}


def run_cross_sectional_momentum(universe_data, lookback=60, top_n=3,
                                  rebalance_days=21, allow_short=True):
    """Long the `top_n` highest trailing-return names, short the `top_n` lowest.

    universe_data: dict sym -> DataFrame (DateIndex, has 'Close').
    Returns (metrics_dict, rebalance_count).
    """
    # Align all closes into one DataFrame
    closes = pd.DataFrame({sym: d["Close"] for sym, d in universe_data.items()})
    closes = closes.dropna(how="all")
    if closes.shape[0] < lookback + rebalance_days + 5:
        return _portfolio_metrics(None), 0

    cash = INIT_CAP
    positions = {}  # sym -> (qty, entry_price, direction)
    equity_curve = []
    n = closes.shape[1]
    k = min(top_n, n // 2)
    if k < 1:
        return _portfolio_metrics(None), 0

    dates = closes.index
    rb = 0
    rebalances = 0
    while rb < len(dates) - lookback:
        # Entry/hold window: [rb, rb+rebalance_days]
        end = min(rb + rebalance_days, len(dates) - 1)
        # Compute trailing return using data strictly up to rb (no look-ahead)
        mom = (closes.iloc[rb] / closes.iloc[rb - lookback] - 1)
        mom = mom.dropna().sort_values(ascending=False)
        if len(mom) < 2 * k:
            rb = end
            continue
        longs = list(mom.index[:k])
        shorts = list(mom.index[-k:]) if allow_short else []

        # Close existing positions at rb
        for sym, (qty, ep, direction) in positions.items():
            price = closes.loc[dates[rb], sym]
            if direction == "LONG":
                cash += qty * price
            else:
                cash -= qty * price  # cover short
        positions = {}

        # Open new positions, equal dollar weight
        per_name = cash / (k + len(shorts)) if (k + len(shorts)) > 0 else cash
        for sym in longs:
            price = closes.loc[dates[rb], sym]
            if price > 0:
                qty = int(per_name / price)
                if qty > 0:
                    positions[sym] = (qty, price, "LONG")
                    cash -= qty * price
        for sym in shorts:
            price = closes.loc[dates[rb], sym]
            if price > 0:
                qty = int(per_name / price)
                if qty > 0:
                    positions[sym] = (qty, price, "SHORT")
                    cash += qty * price  # proceeds from short sale

        rebalances += 1
        # Mark to market over the hold window
        for j in range(rb, end + 1):
            port = cash
            for sym, (qty, ep, direction) in positions.items():
                price = closes.iloc[j][sym]
                if direction == "LONG":
                    port += qty * price
                else:
                    port += qty * (ep - price) + qty * ep  # short P&L + proceeds
            equity_curve.append({"Date": dates[j], "Equity": port})
        rb = end

    eq = pd.Series({r["Date"]: r["Equity"] for r in equity_curve})
    return _portfolio_metrics(eq), rebalances


def walk_forward_factor(universe_data, min_rebalances=4):
    """Optimize (lookback, top_n) on train, test STRICTLY forward on OOS."""
    folds = make_folds(list(universe_data.values())[0])  # folds from first symbol
    # Build per-symbol folds aligned by index
    sym_folds = {sym: make_folds(d) for sym, d in universe_data.items()}
    n_folds = min(len(f) for f in sym_folds.values())
    if n_folds == 0:
        return None

    grid = list(itertools.product([20, 40, 60, 90], [2, 3, 4]))
    oos_results = []
    for f in range(n_folds):
        train = {sym: sym_folds[sym][f][0] for sym in universe_data}
        test = {sym: sym_folds[sym][f][1] for sym in universe_data}
        # Optimize on train
        best = None
        for lookback, top_n in grid:
            m, rb = run_cross_sectional_momentum(train, lookback=lookback, top_n=top_n)
            if rb < min_rebalances:
                continue
            score = m["sharpe_ratio"]
            if best is None or score > best[0]:
                best = (score, lookback, top_n)
        if best is None:
            continue
        _, lookback, top_n = best
        # Strict forward OOS test
        m_oos, rb_oos = run_cross_sectional_momentum(test, lookback=lookback, top_n=top_n)
        m_def, _ = run_cross_sectional_momentum(test, lookback=60, top_n=3)
        oos_results.append((lookback, top_n, m_oos, m_def, rb_oos))
        print(f"  fold{f}: opt(lookback={lookback}, top_n={top_n}) "
              f"OOS ret={m_oos['total_return_pct']:.2f}% sharpe={m_oos['sharpe_ratio']:.2f} "
              f"maxDD={m_oos['max_drawdown_pct']:.2f}% | default OOS ret={m_def['total_return_pct']:.2f}%",
              flush=True)

    if not oos_results:
        return None
    avg_ret = float(np.mean([r[2]["total_return_pct"] for r in oos_results]))
    avg_sharpe = float(np.mean([r[2]["sharpe_ratio"] for r in oos_results]))
    avg_dd = float(np.mean([r[2]["max_drawdown_pct"] for r in oos_results]))
    avg_def = float(np.mean([r[3]["total_return_pct"] for r in oos_results]))
    n_pos = sum(1 for r in oos_results if r[2]["total_return_pct"] > 0)
    return {"avg_oos_ret": avg_ret, "avg_oos_sharpe": avg_sharpe,
            "avg_oos_dd": avg_dd, "avg_def_ret": avg_def,
            "positive_folds": n_pos, "n_folds": len(oos_results),
            "detail": oos_results}


def main():
    fetcher = YFinanceDataFetcher()
    print("Loading 3y universe data for cross-sectional momentum...")
    data = {}
    for s in UNIVERSE:
        try:
            d = fetcher.get_historical_data(s, period="3y")
            if d is not None and not d.empty and len(d) > 200:
                data[s] = d
        except Exception:
            pass
    print(f"  got {len(data)} symbols")

    # Also show a quick full-window (no walk-forward) baseline for reference
    print("\n-- FULL WINDOW baseline (lookback=60, top_n=3) --")
    m, rb = run_cross_sectional_momentum(data, lookback=60, top_n=3)
    print(f"  ret={m['total_return_pct']:.2f}%  sharpe={m['sharpe_ratio']:.2f}  "
          f"maxDD={m['max_drawdown_pct']:.2f}%  rebalances={rb}")

    print("\n-- WALK-FORWARD (optimize on 60%, strict forward OOS 40%) --")
    res = walk_forward_factor(data)
    if res is None:
        print("  insufficient data")
        return
    print("\n" + "=" * 95)
    print("CROSS-SECTIONAL MOMENTUM — WALK-FORWARD OOS SUMMARY")
    print("=" * 95)
    print(f"  Avg OOS return% : {res['avg_oos_ret']:.2f}")
    print(f"  Avg OOS Sharpe  : {res['avg_oos_sharpe']:.2f}")
    print(f"  Avg OOS MaxDD%  : {res['avg_oos_dd']:.2f}")
    print(f"  Avg default OOS : {res['avg_def_ret']:.2f}")
    print(f"  Positive OOS folds: {res['positive_folds']}/{res['n_folds']}")
    verdict = "PROFITABLE EDGE" if (res['avg_oos_ret'] > 0 and res['positive_folds'] >= res['n_folds'] // 2 + 1) \
        else "no persistent edge"
    print(f"  VERDICT: {verdict}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
