"""
validate_macd_moneymap.py — HONEST OOS test of OUR recreation of the YouTube
"MACD money map" (The Secret Mindset, claims 71% win rate).

Reuses the repo's PaperTrader + IS/OOS split on the 10y UNIVERSE, like
validate_walkforward.py. Tests System 1 (trend/distance-rule) across threshold
variants. Kept ONLY if OOS ret>0 and PF>=1.0.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from data_fetcher import YFinanceDataFetcher
from strategies import Signal
from paper_trader import PaperTrader
from short_backtester import UNIVERSE
from strategy_macd_moneymap import MACDMoneyMap

INIT_CAP = 1_000_000.0
SPLIT = "2022-06-01"


def run_window(data, dates, names, signals, min_agree=1):
    from market_intelligence import ConsensusFilter
    cf = ConsensusFilter()
    pt = PaperTrader(initial_capital=INIT_CAP, position_size_pct=0.10)
    for date in dates:
        rows = []
        for sym in data:
            if date not in data[sym].index:
                continue
            for sn in names:
                sval = signals[sn][sym].get(date, 0)
                if sval == Signal.BUY.value:
                    rows.append((sym, "BUY", sn))
                elif sval in (Signal.SELL.value, Signal.SHORT.value):
                    rows.append((sym, "SELL", sn))
        if not rows:
            pt.take_snapshot(str(date.date())); continue
        sdf = pd.DataFrame(rows, columns=["Symbol", "Signal", "Strategy"])
        consensus = [c for c in cf.find_consensus(sdf) if c.count >= min_agree]
        for c in consensus:
            sym = c.symbol; price = data[sym].loc[date, "Close"]
            atr = PaperTrader.calculate_atr(data[sym].loc[:date]).iloc[-1] if len(data[sym].loc[:date]) >= 2 else np.nan
            if c.signal_type == "BUY" and sym not in pt.positions:
                pt.process_signal(sym, Signal.BUY.value, price, str(date.date()), "macd", atr=atr)
            elif c.signal_type == "SELL":
                pt.process_signal(sym, Signal.SELL.value, price, str(date.date()), "macd", atr=atr)
        pt.take_snapshot(str(date.date()))
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

    for dt in [0.002, 0.003, 0.005]:
        strat = MACDMoneyMap(dist_thresh=dt, wait=2)
        sig = {strat.name: {sym: strat.generate_signals(d)["Signal"].reindex(d.index).fillna(0) for sym, d in data.items()}}
        full = run_window(data, common, [strat.name], sig)
        oos = run_window(data, oos_d, [strat.name], sig)
        print(f"\n  [{strat.name}]")
        print(f"    FULL : ret={full['total_return_pct']:+.2f}%  PF={full['profit_factor']:.2f}  win={full['win_rate']:.1f}%  maxDD={full['max_drawdown_pct']:.2f}%  trades={full['total_trades']}")
        print(f"    OOS : ret={oos['total_return_pct']:+.2f}%  PF={oos['profit_factor']:.2f}  win={oos['win_rate']:.1f}%  maxDD={oos['max_drawdown_pct']:.2f}%  trades={oos['total_trades']}")
        print(f"    VERDICT: {'OOS VALIDATED ✅' if (oos['total_return_pct']>0 and oos['profit_factor']>=1.0) else 'OOS FAILS ❌'}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
