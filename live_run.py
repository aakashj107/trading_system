"""
live_run.py — Live-equivalent portfolio simulation through the REAL PaperTrader.

Drives the system's own broker simulation (PaperTrader) with consensus-gated
signals from ALL strategies, over a long (10y) window. Exercises the actual
production code path (ATR stops, costs, long/short, risk manager) — not the
simplified short_backtester. This is the "would this trade live" test.

Performance fix vs v1: each strategy's full-window signal series is computed
ONCE per symbol (strategies are state-free given the full df), then the daily
loop only reads precomputed arrays and runs consensus. This removes the
O(days x strategies) signal regeneration that made 10y infeasible.

Method:
  - Precompute signals[st.name][sym] = full Series aligned to that symbol's index.
  - For each common trading date: read the last value of every strategy/symbol,
    build the consensus df, gate via ConsensusFilter (min_agree).
  - consensus BUY  -> process_signal(BUY)
    consensus SELL -> process_signal(SELL) for open longs; (allow_short) SHORT
  - PaperTrader handles stops/exits/costs natively.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from data_fetcher import YFinanceDataFetcher
from strategies import get_all_strategies
from strategies_pine import get_pine_strategies
# EP + BO families tested: real STANDALONE edge (EP PF 2.6, BO PF 1.25 on 10y) but
# do NOT improve the CONSENSUS portfolio OOS (walk-forward: 19-strat == 17-strat or
# slightly worse). Kept as standalone modules; excluded from default consensus.
# from strategies_pead import get_pead_strategies
# from strategies_breakout import get_breakout_strategies
from paper_trader import PaperTrader, Signal
from market_intelligence import ConsensusFilter

INIT_CAP = 1_000_000.0


def main(min_agree=2, allow_short=False, period="10y", cooldown_days=0,
         pos_pct=0.15, sizing="volatility", circuit_breaker=False,
         cb_thresholds=None, slippage_bps=3.0, universe=None):
    fetcher = YFinanceDataFetcher()
    orig = get_all_strategies()
    pine = get_pine_strategies()
    all_str = orig + pine
    print(f"Loaded {len(all_str)} strategies ({len(orig)} orig + {len(pine)} pine)"
          f"  [+EP/BO available as standalone modules, excluded from consensus]")

    print(f"Loading {period} data for universe...")
    from short_backtester import UNIVERSE as DEFAULT_UNIVERSE
    from config import get_active_stocks
    sym_list = universe if universe is not None else DEFAULT_UNIVERSE
    if universe == "active":
        sym_list = get_active_stocks()
    data = {}
    for s in sym_list:
        try:
            d = fetcher.get_historical_data(s, period=period)
            if d is not None and not d.empty and len(d) > 200:
                data[s] = d
        except Exception:
            pass
    print(f"  got {len(data)} symbols")

    common = None
    for d in data.values():
        common = d.index if common is None else common.intersection(d.index)
    common = common.sort_values()
    print(f"  common trading days: {len(common)}")

    # Precompute each strategy's full signal series per symbol (once).
    print("Precomputing strategy signals (once per symbol)...")
    signals = {}
    for st in all_str:
        signals[st.name] = {}
        for sym, d in data.items():
            try:
                sig = st.generate_signals(d)
                s = sig["Signal"] if "Signal" in sig else pd.Series(0, index=d.index)
                signals[st.name][sym] = s.reindex(d.index).fillna(0)
            except Exception:
                signals[st.name][sym] = pd.Series(0, index=d.index)
    strat_names = list(signals.keys())
    print(f"  precomputed {len(strat_names)} strategies x {len(data)} symbols")

    pt = PaperTrader(initial_capital=INIT_CAP, position_size_pct=pos_pct,
                     slippage_bps=slippage_bps)
    # Apply selected sizing method (volatility / kelly / fixed)
    if sizing in ("volatility", "kelly"):
        pt.sizing_method = sizing
    cb = circuit_breaker
    if not cb:
        # Disable all circuit-breaker limits
        pt.cb_daily_limit = pt.cb_weekly_limit = pt.cb_monthly_limit = 0.0
        pt.cb_losing_streak = 10**9
    elif cb_thresholds is not None:
        # (daily, weekly, monthly, losing_streak) overrides on the instance
        d, w, m, ls = cb_thresholds
        pt.cb_daily_limit = d
        pt.cb_weekly_limit = w
        pt.cb_monthly_limit = m
        pt.cb_losing_streak = ls
    cf = ConsensusFilter()

    # Cooldown: record last EXIT date per symbol; skip new entries within N days.
    last_exit = {}

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
                if allow_short and sym in pt.short_positions:
                    pt.process_signal(sym, Signal.COVER.value, price, str(date.date()),
                                      "consensus", atr=atr)
                    last_exit[sym] = date
                # cooldown gate for fresh long entry
                if sym in pt.positions:
                    continue
                if cooldown_days > 0 and sym in last_exit:
                    if (date - last_exit[sym]).days < cooldown_days:
                        continue
                pt.process_signal(sym, Signal.BUY.value, price, str(date.date()),
                                  "consensus", atr=atr)
            elif c.signal_type == "SELL":
                if sym in pt.positions:
                    pt.process_signal(sym, Signal.SELL.value, price, str(date.date()),
                                      "consensus", atr=atr)
                    last_exit[sym] = date
                if allow_short and sym not in pt.short_positions:
                    if cooldown_days > 0 and sym in last_exit:
                        if (date - last_exit[sym]).days < cooldown_days:
                            continue
                    pt.process_signal(sym, Signal.SHORT.value, price, str(date.date()),
                                      "consensus", atr=atr)
        pt.take_snapshot(str(date.date()))

    for sym in list(pt.positions.keys()):
        price = data[sym].iloc[-1]["Close"]
        pt.process_signal(sym, Signal.SELL.value, price, "END", "close")
    for sym in list(pt.short_positions.keys()):
        price = data[sym].iloc[-1]["Close"]
        pt.process_signal(sym, Signal.COVER.value, price, "END", "close")

    summ = pt.get_performance_summary()
    # Weekly / annualized return from the daily equity snapshots
    snaps = pd.DataFrame(pt.daily_snapshots)
    weekly_pct = None
    ann_pct = None
    if not snaps.empty and len(snaps) > 5:
        snaps["date"] = pd.to_datetime(snaps["date"], errors="coerce")
        snaps = snaps.dropna(subset=["date"]).sort_values("date").set_index("date")
        eq = snaps["portfolio_value"]
        total_ret = eq.iloc[-1] / eq.iloc[0] - 1
        n_days = (snaps.index[-1] - snaps.index[0]).days
        n_weeks = n_days / 7.0
        weekly_pct = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / n_weeks) - 1 if n_weeks > 0 else None
        ann_pct = (eq.iloc[-1] / eq.iloc[0]) ** (365.0 / max(n_days, 1)) - 1 if n_days > 0 else None
    print("\n" + "=" * 95)
    print(f"LIVE-EQUIVALENT RUN (PaperTrader, min_agree={min_agree}, short={allow_short}, "
          f"cooldown={cooldown_days}d, sizing={sizing}, cb={circuit_breaker}, {period})")
    print("=" * 95)
    for k, v in summ.items():
        if isinstance(v, dict):
            print(f"  {k}: (dict)")
        else:
            print(f"  {k}: {v}")
    if weekly_pct is not None:
        roll_max = eq.cummax()
        max_dd = (eq / roll_max - 1).min()
        print(f"  --- horizon ---")
        print(f"  total_return      : {total_ret*100:+.2f}%")
        print(f"  weekly_return     : {weekly_pct*100:+.3f}%  (target >=+2.00%)")
        print(f"  annualized_return : {ann_pct*100:+.2f}%")
        print(f"  max_drawdown      : {max_dd*100:+.2f}%")
        print(f"  sizing_method     : {sizing}")
        print(f"  cb_halt_days      : {pt._cb_halt_days}")
        verdict = "MEETS 2%/week" if weekly_pct >= 0.02 else "BELOW 2%/week target"
        print(f"  weekly target     : {verdict}")
    print("\nDONE.")


if __name__ == "__main__":
    # WINNING CONFIG (validated 10y live PaperTrader): min_agree=2 + long-only.
    # volatility sizing lifts annualized ~6.9%->10.4% (ATR-stop risk-based deploy).
    # circuit_breaker OFF by default: tested on this portfolio it cut return AND
    # worsened drawdown (halts during healthy pullbacks); kept as an opt-in switch.
    main(min_agree=2, allow_short=False, period="10y", cooldown_days=0,
         sizing="volatility", circuit_breaker=False)
