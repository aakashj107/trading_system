"""
validate_momentum_rotation.py — HONEST OOS test of OUR momentum-rotation
recreation (RupeeCase-style cross-sectional momentum).

Reuses the repo's PaperTrader + IS/OOS split on the 10y UNIVERSE, exactly like
validate_walkforward.py. Reports return/PF/win/DD for:
  (A) full window
  (B) OOS window only (blind)
so we can state an honest verdict instead of trusting RupeeCase's screenshot.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from data_fetcher import YFinanceDataFetcher
from paper_trader import PaperTrader, Signal
from short_backtester import UNIVERSE
from strategy_momentum_rotation import MomentumRotation

INIT_CAP = 1_000_000.0
SPLIT = "2022-06-01"


def run(data, dates, strat):
    pt = PaperTrader(initial_capital=INIT_CAP, position_size_pct=0.10)
    held = set()
    for date in dates:
        targets = strat.targets_on(data, date)
        # rebalance only on cadence boundaries (monthly)
        if date.month != (date - pd.Timedelta(days=1)).month or date == dates[0]:
            # SELL names no longer in target
            for sym in list(held):
                if sym not in targets and sym in pt.positions:
                    px = data[sym].loc[date, "Close"]
                    pt.process_signal(sym, Signal.SELL.value, px, str(date.date()), "rotation")
            # BUY new targets
            for sym in targets:
                if sym not in held and sym in data and date in data[sym].index:
                    px = data[sym].loc[date, "Close"]
                    pt.process_signal(sym, Signal.BUY.value, px, str(date.date()), "rotation",
                                      atr=PaperTrader.calculate_atr(data[sym].loc[:date]).iloc[-1] if len(data[sym].loc[:date]) >= 2 else np.nan)
            held = set(targets)
    for sym in list(pt.positions.keys()):
        pt.process_signal(sym, Signal.SELL.value, data[sym].iloc[-1]["Close"], "END", "close")
    res = pt.get_performance_summary()
    eq = pd.Series([s["portfolio_value"] for s in pt.daily_snapshots])
    peak = eq.cummax()
    mdd = ((eq - peak) / peak).min() * 100 if len(eq) else 0.0
    res["max_drawdown_pct"] = round(float(mdd), 2)
    return res


def main():
    f = YFinanceDataFetcher()
    data = {}
    for s in UNIVERSE:
        try:
            d = f.get_historical_data(s, period="10y")
            if d is not None and not d.empty and len(d) > 252:
                data[s] = d
        except Exception:
            pass
    common = None
    for d in data.values():
        common = d.index if common is None else common.intersection(d.index)
    common = common.sort_values()
    mid = common.searchsorted(pd.Timestamp(SPLIT))
    is_d = common[:mid]; oos_d = common[mid:]
    print(f"Loaded {len(data)} symbols | IS {is_d[0].date()}..{is_d[-1].date()} | OOS {oos_d[0].date()}..{oos_d[-1].date()}")

    for (ln, sk, n, rb) in [(12, 1, 10, "M"), (6, 1, 10, "M"), (12, 1, 20, "M")]:
        strat = MomentumRotation(lookback_months=ln, skip_months=sk, hold_n=n, rebalance=rb)
        full = run(data, common, strat)
        oos = run(data, oos_d, strat)
        print(f"\n  [{strat.name}]")
        print(f"    FULL : ret={full['total_return_pct']:+.2f}%  PF={full['profit_factor']:.2f}  "
              f"win={full['win_rate']:.1f}%  maxDD={full['max_drawdown_pct']:.2f}%  trades={full['total_trades']}")
        print(f"    OOS : ret={oos['total_return_pct']:+.2f}%  PF={oos['profit_factor']:.2f}  "
              f"win={oos['win_rate']:.1f}%  maxDD={oos['max_drawdown_pct']:.2f}%  trades={oos['total_trades']}")
        verdict = "OOS VALIDATED ✅" if (oos['total_return_pct'] > 0 and oos['profit_factor'] >= 1.0) else "OOS FAILS ❌"
        print(f"    VERDICT: {verdict}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
