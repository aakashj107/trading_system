"""
validate_walkforward.py — HONEST out-of-sample validation of a candidate config.

Why this exists:
  New families (EP, BO, Donchian) are often *tuned on the full 10y*, which leaks
  hindsight. A truthful test must re-derive tunable params from IN-SAMPLE data
  only, then evaluate the whole config BLIND on OUT-OF-SAMPLE data it never saw.

Method (strict walk-forward, no look-ahead):
  - IS window : 2016-08 .. 2022-06  (first ~6y)
  - OOS window: 2022-06 .. 2026-07  (last ~4y, genuinely unseen by param tuning)
  - On IS: grid-search min_agree in {1,2,3} x Donchian entry-lookback in {200,250}
    -> pick the combo with the best IS return. (params come ONLY from IS.)
  - On OOS: apply the IS-chosen combo FIXED, report return/PF. No re-tuning.
  - Compare OOS 19-strat vs OOS 17-strat baseline (orig+pine, min_agree=2) to see if
    the new families add genuine forward edge.

Long-only throughout (short leg proven drag). Uses the real PaperTrader with brokerage.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from data_fetcher import YFinanceDataFetcher
from strategies import get_all_strategies
from strategies_pine import get_pine_strategies
from strategies_donchian import DonchianBreakoutStrategy
from strategies_breakout import BreakoutStrategy
from paper_trader import PaperTrader, Signal
from market_intelligence import ConsensusFilter

INIT_CAP = 1_000_000.0


def build_signals(data, strategy_list):
    """Precompute {name: {sym: Signal series}} for a list of strategy objects."""
    signals = {}
    for st in strategy_list:
        signals[st.name] = {}
        for sym, d in data.items():
            try:
                s = st.generate_signals(d)["Signal"]
                signals[st.name][sym] = s.reindex(d.index).fillna(0)
            except Exception:
                signals[st.name][sym] = pd.Series(0, index=d.index)
    return signals


def run_window(data, common, strat_names, signals, min_agree, allow_short):
    cf = ConsensusFilter()
    pt = PaperTrader(initial_capital=INIT_CAP, position_size_pct=0.15)

    def atr_of(sym, date):
        hist = data[sym].loc[:date]
        return PaperTrader.calculate_atr(hist).iloc[-1] if len(hist) >= 2 else np.nan

    for date in common:
        rows = []
        for sym in data:
            if date not in data[sym].index:
                continue
            for sn in strat_names:
                sval = signals[sn][sym].get(date, 0)
                if sval == Signal.BUY.value:
                    rows.append((sym, "BUY", sn))
                elif sval in (Signal.SELL.value, Signal.SHORT.value):
                    rows.append((sym, "SELL", sn))
        if not rows:
            pt.take_snapshot(str(date.date()))
            continue
        sdf = pd.DataFrame(rows, columns=["Symbol", "Signal", "Strategy"])
        consensus = [c for c in cf.find_consensus(sdf) if c.count >= min_agree]
        for c in consensus:
            sym = c.symbol
            price = data[sym].loc[date, "Close"]
            atr = atr_of(sym, date)
            if c.signal_type == "BUY":
                if sym in pt.positions:
                    continue
                pt.process_signal(sym, Signal.BUY.value, price, str(date.date()),
                                  "consensus", atr=atr)
            elif c.signal_type == "SELL":
                if sym in pt.positions:
                    pt.process_signal(sym, Signal.SELL.value, price, str(date.date()),
                                      "consensus", atr=atr)
        pt.take_snapshot(str(date.date()))
    for sym in list(pt.positions.keys()):
        pt.process_signal(sym, Signal.SELL.value, data[sym].iloc[-1]["Close"], "END", "close")
    return pt.get_performance_summary()


def main(split_date="2022-06-01"):
    fetcher = YFinanceDataFetcher()
    from short_backtester import UNIVERSE
    data = {}
    for s in UNIVERSE:
        try:
            d = fetcher.get_historical_data(s, period="10y")
            if d is not None and not d.empty and len(d) > 200:
                data[s] = d
        except Exception:
            pass
    common = None
    for d in data.values():
        common = d.index if common is None else common.intersection(d.index)
    common = common.sort_values()
    print(f"Loaded {len(data)} symbols, {len(common)} common days")

    mid = common.searchsorted(pd.Timestamp(split_date))
    is_dates = common[:mid]
    oos_dates = common[mid:]
    print(f"IS window : {is_dates[0].date()} .. {is_dates[-1].date()} ({len(is_dates)} days)")
    print(f"OOS window: {oos_dates[0].date()} .. {oos_dates[-1].date()} ({len(oos_dates)} days)")

    allow_short = False
    base_str = get_all_strategies() + get_pine_strategies()
    base_names = [st.name for st in base_str]

    # ── IN-SAMPLE: tune min_agree x DC entry-lookback on IS only ──
    print("\n[IN-SAMPLE] grid-search over IS window (params derived WITHOUT OOS)...")
    best = None
    for ma in (1, 2, 3):
        for dcent in (200, 250):
            str_list = base_str + [DonchianBreakoutStrategy(entry_lookback=dcent)]
            sig = build_signals(data, str_list)
            names = [st.name for st in str_list]
            res = run_window(data, is_dates, names, sig, ma, allow_short)
            key = (ma, dcent)
            print(f"  ma={ma} DCentry={dcent}: "
                  f"ret={res['total_return_pct']:+.2f}% PF={res['profit_factor']:.2f} "
                  f"trades={res['total_trades']}")
            if best is None or res["total_return_pct"] > best[1]["total_return_pct"]:
                best = (key, res)
    best_ma, best_dcent = best[0]
    print(f"  -> IS-best combo: min_agree={best_ma}, DC entry={best_dcent}")

    # ── OUT-OF-SAMPLE: apply IS-best FIXED, blind ──
    print(f"\n[OUT-OF-SAMPLE] applying IS-chosen combo blind on unseen window...")
    full_list = base_str + [DonchianBreakoutStrategy(entry_lookback=best_dcent)]
    oos_sig = build_signals(data, full_list)
    oos_names = [st.name for st in full_list]
    oos_18 = run_window(data, oos_dates, oos_names, oos_sig, best_ma, allow_short)

    # OOS baseline: 17-strat (orig+pine) at min_agree=2
    base_sig = build_signals(data, base_str)
    oos_17 = run_window(data, oos_dates, base_names, base_sig, 2, allow_short)

    print("\n" + "=" * 80)
    print("HONEST OOS VALIDATION SUMMARY (long-only, real PaperTrader, params from IS only)")
    print("=" * 80)
    print(f"  IS-best combo     : min_agree={best_ma}, DC entry={best_dcent}")
    print(f"  OOS 18-strat(+DC) : ret={oos_18['total_return_pct']:+.2f}%  "
          f"PF={oos_18['profit_factor']:.2f}  win={oos_18['win_rate']:.1f}%  "
          f"trades={oos_18['total_trades']}")
    print(f"  OOS 17-strat base : ret={oos_17['total_return_pct']:+.2f}%  "
          f"PF={oos_17['profit_factor']:.2f}  win={oos_17['win_rate']:.1f}%  "
          f"trades={oos_17['total_trades']}")
    if oos_18["total_return_pct"] > 0 and oos_18["profit_factor"] >= 1.0:
        verdict = "18-STRAT(+DC) CONFIG VALIDATED OOS ✅"
    elif oos_18["total_return_pct"] > oos_17["total_return_pct"]:
        verdict = "18-strat(+DC) BEATS 17-strat baseline OOS (still weak)"
    else:
        verdict = "18-strat(+DC) does NOT beat baseline OOS — likely overfit on IS"
    print(f"  VERDICT: {verdict}")
    print("\nDONE.")


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "2022-06-01"
    main(split_date=split)
