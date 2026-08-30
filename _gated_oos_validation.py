"""
Honest OOS check: does the Nifty low-vol GATE make the REAL 19-strategy
consensus MORE profitable, or is it the same?

Reuses validate_walkforward.py's exact harness (real PaperTrader, real 19-strat
consensus, IS/OOS split on the 10y UNIVERSE). Adds the vol-gate: on OOS dates
where Nifty realized vol is HIGH, skip new consensus entries (stand aside).

Compares:
  (A) UNGATED 19-strat consensus on OOS
  (B) VOL-GATED 19-strat consensus on OOS
Both use the SAME IS-tuned min_agree (derived from IS only, no OOS leakage).

Answer to "is our system profitable / more or same":
  If (B) ret > (A) ret AND (B) DD <= (A) DD  -> gate ADDS value.
  If (B) ~= (A)                                -> gate is neutral (no harm).
  If (B) < (A)                                 -> gate HURTS (skip it).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from data_fetcher import YFinanceDataFetcher
from strategies import get_all_strategies
from strategies_pine import get_pine_strategies
from strategies_donchian import DonchianBreakoutStrategy
from paper_trader import PaperTrader, Signal
from market_intelligence import ConsensusFilter
from short_backtester import UNIVERSE
from volatility_gate import VolatilityGate

INIT_CAP = 1_000_000.0
SPLIT = "2022-06-01"


def build_signals(data, strat_list):
    signals = {}
    for st in strat_list:
        signals[st.name] = {}
        for sym, d in data.items():
            try:
                s = st.generate_signals(d)["Signal"]
                signals[st.name][sym] = s.reindex(d.index).fillna(0)
            except Exception:
                signals[st.name][sym] = pd.Series(0, index=d.index)
    return signals


def run_window(data, dates, names, signals, min_agree, vol_flag=None):
    cf = ConsensusFilter()
    pt = PaperTrader(initial_capital=INIT_CAP, position_size_pct=0.15)

    def atr_of(sym, date):
        hist = data[sym].loc[:date]
        return PaperTrader.calculate_atr(hist).iloc[-1] if len(hist) >= 2 else np.nan

    for date in dates:
        # vol-gate: stand aside when flag False on this date
        if vol_flag is not None and not bool(vol_flag.get(date, True)):
            pt.take_snapshot(str(date.date()))
            continue
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
            pt.take_snapshot(str(date.date()))
            continue
        sdf = pd.DataFrame(rows, columns=["Symbol", "Signal", "Strategy"])
        consensus = [c for c in cf.find_consensus(sdf) if c.count >= min_agree]
        for c in consensus:
            sym = c.symbol
            price = data[sym].loc[date, "Close"]
            atr = atr_of(sym, date)
            if c.signal_type == "BUY" and sym not in pt.positions:
                pt.process_signal(sym, Signal.BUY.value, price, str(date.date()), "consensus", atr=atr)
            elif c.signal_type == "SELL":
                pt.process_signal(sym, Signal.SELL.value, price, str(date.date()), "consensus", atr=atr)
        pt.take_snapshot(str(date.date()))
    for sym in list(pt.positions.keys()):
        pt.process_signal(sym, Signal.SELL.value, data[sym].iloc[-1]["Close"], "END", "close")
    res = pt.get_performance_summary()
    # Max drawdown from daily equity snapshots
    eq = pd.Series([s["portfolio_value"] for s in pt.daily_snapshots])
    peak = eq.cummax()
    mdd = ((eq - peak) / peak).min() * 100 if len(eq) else 0.0
    res["max_drawdown_pct"] = round(float(mdd), 2)
    return res


def main():
    fetcher = YFinanceDataFetcher()
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
    mid = common.searchsorted(pd.Timestamp(SPLIT))
    is_dates = common[:mid]; oos_dates = common[mid:]
    print(f"Loaded {len(data)} symbols | IS {is_dates[0].date()}..{is_dates[-1].date()} | OOS {oos_dates[0].date()}..{oos_dates[-1].date()}")

    # IS-tune min_agree on IS only (no OOS)
    base_str = get_all_strategies() + get_pine_strategies()
    full = base_str + [DonchianBreakoutStrategy(entry_lookback=250)]
    full_names = [st.name for st in full]
    sig_full = build_signals(data, full)
    best = None
    for ma in (1, 2, 3):
        r = run_window(data, is_dates, full_names, sig_full, ma)
        print(f"  IS ma={ma}: ret={r['total_return_pct']:+.2f}% PF={r['profit_factor']:.2f} trades={r['total_trades']}")
        if best is None or r['total_return_pct'] > best[1]['total_return_pct']:
            best = (ma, r)
    best_ma = best[0]
    print(f"  -> IS-best min_agree={best_ma}")

    # Vol flag from Nifty on OOS dates
    nifty = fetcher.get_nifty_index(period="10y")
    gate = VolatilityGate()
    vflag = gate.low_vol_flag(nifty).reindex(oos_dates).fillna(False)
    n_low = int(vflag.sum()); n_high = len(oos_dates) - n_low
    print(f"  OOS vol state: LOW(permit)={n_low} days, HIGH(standaside)={n_high} days")

    # OOS ungated
    oos_ungated = run_window(data, oos_dates, full_names, sig_full, best_ma)
    # OOS gated
    oos_gated = run_window(data, oos_dates, full_names, sig_full, best_ma, vol_flag=vflag)

    print("\n" + "=" * 82)
    print("HONEST OOS: 19-STRAT CONSENSUS — GATED vs UNGATED (real PaperTrader)")
    print("=" * 82)
    print(f"  min_agree (IS-tuned) : {best_ma}")
    for label, r in [("UNGATED", oos_ungated), ("VOL-GATED", oos_gated)]:
        print(f"  {label:10s}: ret={r['total_return_pct']:+.2f}%  PF={r['profit_factor']:.2f}  "
              f"win={r['win_rate']:.1f}%  maxDD={r['max_drawdown_pct']:.2f}%  trades={r['total_trades']}")
    delta = oos_gated['total_return_pct'] - oos_ungated['total_return_pct']
    dd_delta = oos_gated['max_drawdown_pct'] - oos_ungated['max_drawdown_pct']
    # dd_delta > 0 means gated DD is LESS negative (smaller drawdown) = improvement
    if delta > 0 and dd_delta >= 0:
        v = "GATE ADDS VALUE ✅ (higher return, lower/equal DD)"
    elif abs(delta) < 1.0 and dd_delta >= 0:
        v = "GATE IS NEUTRAL (~same profitability, DD not worse)"
    else:
        v = "GATE HURTS ❌ (lower return or worse DD)"
    print(f"\n  Δreturn={delta:+.2f}pp  ΔmaxDD={dd_delta:+.2f}pp (positive DD Δ = smaller drawdown = better)")
    print(f"  VERDICT: {v}")
    print("DONE.")


if __name__ == "__main__":
    main()
