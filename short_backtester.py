"""
short_backtester.py — Extended backtesting tool for the Indian Market Trading System.

Role in the codebase:
    backtester.py (the production engine) handles LONG + SHORT directions with
    correct capital accounting and feeds daily_runner. This module is the
    RESEARCH-VALIDATION layer on top of it: it reuses
    strategies.generate_signals() and backtester._calculate_metrics() so numbers
    stay comparable, and adds the analysis the production path does not:
    - Out-of-sample validation (in-sample 60% / out-of-sample 40% split)
    - Transaction-cost sensitivity (0.5x / 1x / 2x costs)
    - Longer multi-year windows (regime robustness)

Consumed by: factors.py (cross-sectional momentum study) and live_run.py
(live-equivalent portfolio simulation).

Usage:
    python short_backtester.py            # full extended analysis
    from short_backtester import run_short_backtest, metrics
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from strategies import get_all_strategies, Signal
from strategies_pine import get_pine_strategies
from data_fetcher import YFinanceDataFetcher
from backtester import Backtester, TradingCosts


UNIVERSE = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
            "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
            "SHRIRAMFIN.NS", "MUTHOOTFIN.NS", "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS",
            "TECHM.NS", "MPHASIS.NS", "HINDUNILVR.NS"]

INIT_CAP = 500_000.0
POS_PCT = 0.15
SL_PCT = 0.03
TRAIL_PCT = 0.02
_BASE_COSTS = TradingCosts()


def _charges(bv, sv, cost_mult=1.0):
    """Brokerage + STT + exchange + SEBI + stamp + GST, scalable by cost_mult."""
    c = _BASE_COSTS
    brokerage = c.brokerage_per_order * 2
    stt = bv * c.stt_delivery_buy + sv * c.stt_delivery_sell
    exchange = (bv + sv) * c.exchange_txn_charge
    sebi = (bv + sv) * c.sebi_turnover_fee
    stamp = bv * c.stamp_duty_buy
    gst = brokerage * c.gst_on_brokerage
    return (brokerage + stt + exchange + sebi + stamp + gst) * cost_mult


def _atr(df, n=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _adx(high, low, close, n=14):
    """Wilder's ADX — directional conviction. Low ADX => range/non-trending."""
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    alpha = 1.0 / n
    s_tr = tr.ewm(alpha=alpha, min_periods=n, adjust=False).mean()
    s_pdm = plus_dm.ewm(alpha=alpha, min_periods=n, adjust=False).mean()
    s_mdm = minus_dm.ewm(alpha=alpha, min_periods=n, adjust=False).mean()
    pdi = 100 * s_pdm / s_tr.replace(0, np.nan)
    mdi = 100 * s_mdm / s_tr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=alpha, min_periods=n, adjust=False).mean()



def run_short_backtest(strategy, data, allow_short=True, cost_mult=1.0,
                       trend_gate=False, use_atr_stop=False, sma_n=200, atr_n=14,
                       atr_mult=2.0, adx_gate=False, adx_len=14, adx_min=20.0):
    """Short-aware single-stock backtest.

    LONG on BUY, SHORT on SELL/SHORT (if allow_short).
    - trend_gate: only take LONG when Close > SMA(sma_n), SHORT when Close < SMA.
    - use_atr_stop: stop/trail sized to atr_mult x ATR instead of fixed %.
    - adx_gate: suppress NEW entries when ADX(adx_len) < adx_min (range market).
      Faithful port of the Pine "Ultimate Market Structure" TREND/RANGE filter:
      don't initiate trades when the market lacks directional conviction.
    Charges scaled by cost_mult. Returns (trades, equity_curve_df).
    """
    df = strategy.generate_signals(data).dropna()
    if df is None or df.empty:
        return [], pd.DataFrame()

    # Optional indicators aligned by date
    sma = None
    atr_series = None
    adx_series = None
    if trend_gate or use_atr_stop or adx_gate:
        ind = data.copy()
        if trend_gate:
            sma = ind["Close"].rolling(sma_n).mean()
        if use_atr_stop:
            atr_series = _atr(ind, atr_n)
        if adx_gate:
            adx_series = _adx(ind["High"], ind["Low"], ind["Close"], adx_len)
        df = df.join(pd.DataFrame({"_SMA": sma, "_ATR": atr_series,
                                   "_ADX": adx_series}), how="left")

    capital = INIT_CAP
    long_pos = None
    short_pos = None
    trades = []
    curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        close = row["Close"]
        sig = row["Signal"]

        # Resolve gate / stop sizing for this bar
        bullish = (close > row["_SMA"]) if trend_gate else True
        bearish = (close < row["_SMA"]) if trend_gate else True
        adx_ok = (not adx_gate) or (not np.isnan(row["_ADX"]) and row["_ADX"] >= adx_min)
        atr_v = row["_ATR"] if use_atr_stop else np.nan
        long_stop_dist = atr_v * atr_mult if use_atr_stop else close * SL_PCT
        short_stop_dist = atr_v * atr_mult if use_atr_stop else close * SL_PCT

        # ── Manage long ──
        if long_pos is not None:
            trail = (close - long_stop_dist) if use_atr_stop else close * (1 - TRAIL_PCT)
            trail = max(long_pos.trailing_stop, trail)
            if close <= trail:
                sv = close * long_pos.quantity
                ch = _charges(long_pos.entry_price * long_pos.quantity, sv, cost_mult)
                pnl = (close - long_pos.entry_price) * long_pos.quantity - ch
                trades.append(_trade(long_pos, date, close, ch, pnl, "LONG", "STOP_LOSS"))
                capital += sv - ch
                long_pos = None
            elif sig == Signal.SELL.value:
                sv = close * long_pos.quantity
                ch = _charges(long_pos.entry_price * long_pos.quantity, sv, cost_mult)
                pnl = (close - long_pos.entry_price) * long_pos.quantity - ch
                trades.append(_trade(long_pos, date, close, ch, pnl, "LONG", "SIGNAL"))
                capital += sv - ch
                long_pos = None

        # ── Manage short ──
        if short_pos is not None and allow_short:
            trail = (close + short_stop_dist) if use_atr_stop else close * (1 + TRAIL_PCT)
            trail = min(short_pos.trailing_stop, trail)
            if close >= trail:
                sv = close * short_pos.quantity
                ch = _charges(short_pos.entry_price * short_pos.quantity, sv, cost_mult)
                pnl = (short_pos.entry_price - close) * short_pos.quantity - ch
                trades.append(_trade(short_pos, date, close, ch, pnl, "SHORT", "STOP_LOSS"))
                capital += short_pos.entry_price * short_pos.quantity - sv
                short_pos = None
            elif sig == Signal.COVER.value:
                sv = close * short_pos.quantity
                ch = _charges(short_pos.entry_price * short_pos.quantity, sv, cost_mult)
                pnl = (short_pos.entry_price - close) * short_pos.quantity - ch
                trades.append(_trade(short_pos, date, close, ch, pnl, "SHORT", "SIGNAL"))
                capital += short_pos.entry_price * short_pos.quantity - sv
                short_pos = None

        # ── Entries (no overlapping long + short) ──
        if long_pos is None and short_pos is None and adx_ok:
            if sig == Signal.BUY.value and (not trend_gate or bullish):
                q = int(capital * POS_PCT / close)
                if q > 0:
                    long_pos = _pos(date, close, q, long_stop_dist / close)
            elif sig in (Signal.SHORT.value, Signal.SELL.value) and allow_short \
                    and (not trend_gate or bearish):
                q = int(capital * POS_PCT / close)
                if q > 0:
                    short_pos = _pos(date, close, q, short_stop_dist / close, short=True)

        port = capital
        if long_pos:
            port += close * long_pos.quantity
        if short_pos:
            port += (short_pos.entry_price - close) * short_pos.quantity
        curve.append({"Date": date, "Equity": port})

    # Close remaining at end
    lc, ld = df.iloc[-1]["Close"], df.index[-1]
    for pos, side in [(long_pos, "LONG"), (short_pos, "SHORT")]:
        if pos is None:
            continue
        sv = lc * pos.quantity
        ch = _charges(pos.entry_price * pos.quantity, sv, cost_mult)
        if side == "LONG":
            pnl = (lc - pos.entry_price) * pos.quantity - ch
            capital += sv - ch
        else:
            pnl = (pos.entry_price - lc) * pos.quantity - ch
            capital += pos.entry_price * pos.quantity - sv
        trades.append(_trade(pos, ld, lc, ch, pnl, side, "END_OF_DATA"))

    eq = pd.DataFrame(curve).set_index("Date") if curve else pd.DataFrame()
    return trades, eq


def _pos(date, close, q, sl, short=False):
    sl_price = close * (1 - sl) if not short else close * (1 + sl)
    return type("P", (), {"entry_date": date, "entry_price": close, "quantity": q,
                          "stop_loss": sl_price, "trailing_stop": sl_price})()


def _trade(pos, date, close, ch, pnl, side, reason):
    return type("T", (), {"pnl": pnl, "holding_days": (date - pos.entry_date).days,
                          "total_charges": ch, "exit_reason": reason,
                          "direction": side, "entry_price": pos.entry_price,
                          "quantity": pos.quantity, "symbol": "X",
                          "entry_date": pos.entry_date, "exit_date": date,
                          "exit_price": close, "pnl_pct": 0.0, "brokerage": ch})()


def metrics(trades, eq):
    bt = Backtester(initial_capital=INIT_CAP, position_size_pct=POS_PCT)
    return bt._calculate_metrics(trades, eq, "PORTFOLIO")


class ConsensusStrategy:
    """Wraps the full strategy set and emits a bar-level BUY/SELL only when
    >= min_agree strategies agree on that bar. Mirrors the system's own
    ConsensusFilter logic (count of agreeing strategies per day)."""

    name = "CONSENSUS"

    def __init__(self, strategies, min_agree=2):
        self._strategies = strategies
        self._min_agree = min_agree

    def generate_signals(self, data):
        frames = []
        for st in self._strategies:
            sig = st.generate_signals(data)["Signal"].rename(st.name)
            frames.append(sig)
        sig_matrix = pd.concat(frames, axis=1)
        out = pd.DataFrame(index=data.index)
        out["Close"] = data["Close"]
        decision = []
        for _, row in sig_matrix.iterrows():
            buy = int((row == Signal.BUY.value).sum())
            sell = int((row == Signal.SELL.value).sum() + (row == Signal.SHORT.value).sum())
            if buy >= self._min_agree and buy >= sell:
                decision.append(Signal.BUY.value)
            elif sell >= self._min_agree and sell > buy:
                decision.append(Signal.SELL.value)
            else:
                decision.append(Signal.HOLD.value)
        out["Signal"] = decision
        return out


def rolling_windows(data, n=4):
    length = len(data)
    size = length // n
    return [data.iloc[i * size:(i + 1) * size if i < n - 1 else length] for i in range(n)]


def split_is_oos(data, frac=0.6):
    cut = int(len(data) * frac)
    return data.iloc[:cut], data.iloc[cut:]


def load_universe(fetcher, universe, period="3y"):
    """Try a longer window (regime robustness), fall back gracefully."""
    data = {}
    for s in universe:
        try:
            d = fetcher.get_historical_data(s, period=period)
            if d is not None and not d.empty and len(d) > 60:
                data[s] = d
        except Exception:
            pass
    return data


def aggregate(strategies, data, allow_short=True, cost_mult=1.0,
               trend_gate=False, use_atr_stop=False, sma_n=200, atr_n=14, atr_mult=2.0):
    rows = []
    for st in strategies:
        all_t, all_eq = [], []
        for sym, d in data.items():
            t, e = run_short_backtest(st, d, allow_short=allow_short, cost_mult=cost_mult,
                                      trend_gate=trend_gate, use_atr_stop=use_atr_stop,
                                      sma_n=sma_n, atr_n=atr_n, atr_mult=atr_mult)
            all_t.extend(t)
            if not e.empty:
                all_eq.append(e)
        m = metrics(all_t, pd.concat(all_eq)) if all_t else {"total_trades": 0}
        rows.append((st.name, m))
    return rows


def print_table(rows, label):
    out = []
    for name, m in rows:
        if m.get("total_trades", 0) == 0:
            out.append((name, 0, 0.0, 0.0, 0.0, 0.0))
        else:
            out.append((name, m["total_trades"], m["win_rate"], m["total_return_pct"],
                        m["sharpe_ratio"], m["profit_factor"]))
    df = pd.DataFrame(out, columns=["Strategy", "Trades", "WinRate%", "Return%",
                                    "Sharpe", "ProfitFactor"]).sort_values("Return%", ascending=False)
    print(df.to_string(index=False))
    pos = df[df["Return%"] > 0]
    print(f"\nPOSITIVE (Return%>0): {len(pos)} / {len(df)}")
    if len(pos):
        print(pos[["Strategy", "Return%", "Sharpe", "ProfitFactor", "Trades"]].to_string(index=False))
    return df


def main():
    fetcher = YFinanceDataFetcher()

    # ── Longer window (regime robustness) ──
    print("Loading longer window (3y) data for full universe...")
    data = load_universe(fetcher, UNIVERSE, period="3y")
    if len(data) < 5:
        print("  3y sparse, trying 2y...")
        data = load_universe(fetcher, UNIVERSE, period="2y")
    print(f"  got {len(data)} symbols, avg length "
          f"{int(np.mean([len(d) for d in data.values()]))} bars")

    strategies = get_all_strategies() + get_pine_strategies()

    # ── 1) FULL WINDOW, LONG/SHORT ──
    print("\n" + "=" * 95)
    print("1) FULL WINDOW — LONG/SHORT (longer window)")
    print("=" * 95)
    print_table(aggregate(strategies, data, allow_short=True), "LS")

    # ── 2) OUT-OF-SAMPLE: in-sample 60% vs out-of-sample 40% ──
    print("\n" + "=" * 95)
    print("2) OUT-OF-SAMPLE — fixed params, IS(60%) vs OOS(40%), LONG/SHORT")
    print("=" * 95)
    is_rows, oos_rows = [], []
    for st in strategies:
        is_t, is_eq, oos_t, oos_eq = [], [], [], []
        for sym, d in data.items():
            ins, oos = split_is_oos(d)
            for src, tgt_t, tgt_e in ((ins, is_t, is_eq), (oos, oos_t, oos_eq)):
                if len(src) < 40:
                    continue
                t, e = run_short_backtest(st, src, allow_short=True)
                tgt_t.extend(t)
                if not e.empty:
                    tgt_e.append(e)
        is_m = metrics(is_t, pd.concat(is_eq)) if is_t else {"total_trades": 0}
        oos_m = metrics(oos_t, pd.concat(oos_eq)) if oos_t else {"total_trades": 0}
        is_rows.append((st.name, is_m))
        oos_rows.append((st.name, oos_m))

    print("\n-- IN-SAMPLE (first 60%) --")
    is_df = print_table(is_rows, "IS")
    print("\n-- OUT-OF-SAMPLE (last 40%) --")
    oos_df = print_table(oos_rows, "OOS")

    # Consistency: positive in BOTH IS and OOS
    merged = is_df.merge(oos_df, on="Strategy", suffixes=("_is", "_oos"))
    robust = merged[(merged["Return%_is"] > 0) & (merged["Return%_oos"] > 0)]
    print(f"\nROBUST (positive in BOTH IS & OOS): {len(robust)} / {len(merged)}")
    if len(robust):
        print(robust[["Strategy", "Return%_is", "Return%_oos"]].to_string(index=False))

    # ── 3) COST SENSITIVITY: 0.5x / 1x / 2x ──
    print("\n" + "=" * 95)
    print("3) COST SENSITIVITY — full window LONG/SHORT at 0.5x / 1x / 2x costs")
    print("=" * 95)
    for mult in (0.5, 1.0, 2.0):
        rows = aggregate(strategies, data, allow_short=True, cost_mult=mult)
        n_pos = sum(1 for name, m in rows if m.get("total_return_pct", 0) > 0
                    and m.get("total_trades", 0) > 0)
        best = max(rows, key=lambda r: r[1].get("total_return_pct", -1e9))
        print(f"\n  costs x{mult}: positive={n_pos}/16 | "
              f"best={best[0]} ({best[1].get('total_return_pct',0):.2f}%)")

    print("\n" + "=" * 95)
    print("4) IMPROVED — TREND GATE + ATR STOP (3y window, LONG/SHORT)")
    print("=" * 95)
    print("   Only trade LONG when Close>SMA200, SHORT when Close<SMA200;")
    print("   stops sized to 2x ATR.")
    improved = aggregate(strategies, data, allow_short=True,
                         trend_gate=True, use_atr_stop=True, atr_mult=2.0)
    imp_df = print_table(improved, "IMPROVED")

    # Summary comparison vs original (section 1) on the same 3y data
    print("\n" + "=" * 95)
    print("5) SUMMARY — best config per metric (3y, LONG/SHORT)")
    print("=" * 95)
    base = aggregate(strategies, data, allow_short=True)  # fixed % stops, no gate
    base_d = {n: m for n, m in base}
    imp_d = {n: m for n, m in improved}
    print(f"{'Strategy':32}{'BaseRet%':>10}{'ImpRet%':>10}{'ImpΔ':>9}")
    for name in imp_d:
        br = base_d[name].get("total_return_pct", 0.0)
        ir = imp_d[name].get("total_return_pct", 0.0)
        print(f"{name:32}{br:>10.2f}{ir:>10.2f}{ir - br:>9.2f}")
    n_pos_base = sum(1 for n, m in base if m.get("total_return_pct", 0) > 0 and m.get("total_trades", 0) > 0)
    n_pos_imp = sum(1 for n, m in improved if m.get("total_return_pct", 0) > 0 and m.get("total_trades", 0) > 0)
    print(f"\nPositive strategies: base={n_pos_base}/16  improved={n_pos_imp}/16")
    best = max(improved, key=lambda r: r[1].get("total_return_pct", -1e9))
    print(f"Best improved: {best[0]} ({best[1].get('total_return_pct',0):.2f}%, "
          f"Sharpe {best[1].get('sharpe_ratio',0):.2f}, PF {best[1].get('profit_factor',0):.2f})")

    # ── 6) CONSENSUS GATING: require >= N strategies to agree (trend gate + ATR) ──
    print("\n" + "=" * 95)
    print("6) CONSENSUS GATING — require N strategies to agree (3y, trend gate + ATR)")
    print("=" * 95)
    for min_agree in (2, 3):
        cons = ConsensusStrategy(strategies, min_agree=min_agree)
        all_t, all_eq = [], []
        for sym, d in data.items():
            t, e = run_short_backtest(cons, d, allow_short=True,
                                      trend_gate=True, use_atr_stop=True, atr_mult=2.0)
            all_t.extend(t)
            if not e.empty:
                all_eq.append(e)
        m = metrics(all_t, pd.concat(all_eq)) if all_t else {"total_trades": 0}
        pos = m.get("total_return_pct", 0.0)
        pf = m.get("profit_factor", 0.0)
        sh = m.get("sharpe_ratio", 0.0)
        print(f"\n  min_agree={min_agree}: Return%={pos:.2f}  Sharpe={sh:.2f}  "
              f"ProfitFactor={pf:.2f}  Trades={m.get('total_trades',0)}  "
              f"WinRate%={m.get('win_rate',0.0):.2f}")
        # Out-of-sample check for the consensus portfolio
        is_t, is_e, oos_t, oos_e = [], [], [], []
        for sym, d in data.items():
            ins, oos = split_is_oos(d)
            for src, tg_t, tg_e in ((ins, is_t, is_e), (oos, oos_t, oos_e)):
                if len(src) < 40:
                    continue
                t, e = run_short_backtest(cons, src, allow_short=True,
                                          trend_gate=True, use_atr_stop=True, atr_mult=2.0)
                tg_t.extend(t)
                if not e.empty:
                    tg_e.append(e)
        is_m = metrics(is_t, pd.concat(is_e)) if is_t else {"total_return_pct": 0.0}
        oos_m = metrics(oos_t, pd.concat(oos_e)) if oos_t else {"total_return_pct": 0.0}
        print(f"     IS(60%)={is_m.get('total_return_pct',0):.2f}%  "
              f"OOS(40%)={oos_m.get('total_return_pct',0):.2f}%  "
              f"{'ROBUST' if (is_m.get('total_return_pct',0)>0 and oos_m.get('total_return_pct',0)>0) else 'not robust'}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
