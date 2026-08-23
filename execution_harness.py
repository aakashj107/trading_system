"""
execution_harness.py — SAFE signal→execution wiring (simulation only, no real orders).

Purpose:
    Validate the execution PLUMBING (items #12/#13) WITHOUT risking real capital.
    The walk-forward proved the *edge* is tradable; this proves the *pipeline*
    from consensus signal → order → portfolio → (paper) broker → alert actually
    behaves: fills BUYs, closes SELLs, computes P&L, and enforces sector limits.

Safety:
    - LiveTrader runs in mode="paper" (place_order is a print stub, LIVE locked).
    - AlertSystem has no TELEGRAM_BOT_TOKEN, so it silently no-ops.
    - No network, no credentials, no real orders. This is a structural test, not a
      live deployment.

Run:
    python3 execution_harness.py
    (also imported by tests/test_execution_harness.py)
"""
import numpy as np
import pandas as pd

from strategies import get_all_strategies, Signal
from strategies_pine import get_pine_strategies
from strategies_donchian import DonchianBreakoutStrategy
from market_intelligence import ConsensusFilter
from portfolio_manager import PortfolioManager
from order_manager import OrderManager, OrderSide, OrderStatus
from angel_one_live import LiveTrader
from alerts import AlertSystem


# Validated config from walk-forward (validate_walkforward.py):
#   18-strat (+Donchian, entry=200) at min_agree=3 -> OOS +27% / PF 1.78
MIN_AGREE = 3
DONCHIAN_ENTRY = 200


def build_signals(data, strategy_list):
    """{name: {symbol: Signal-series}} for a list of strategy objects."""
    out = {}
    for st in strategy_list:
        out[st.name] = {}
        for sym, d in data.items():
            try:
                s = st.generate_signals(d)["Signal"]
                out[st.name][sym] = s.reindex(d.index).fillna(0).astype(int)
            except Exception:
                out[st.name][sym] = pd.Series(0, index=d.index, dtype=int)
    return out


def run_harness(data, min_agree=MIN_AGREE, capital=1_000_000.0,
                live_mode="paper"):
    """
    Wire consensus -> OrderManager -> PortfolioManager -> LiveTrader -> Alerts.
    Returns a dict with lifecycle stats for assertions.
    """
    strat_list = get_all_strategies() + get_pine_strategies() \
        + [DonchianBreakoutStrategy(entry_lookback=DONCHIAN_ENTRY)]
    signals = build_signals(data, strat_list)

    cf = ConsensusFilter()
    portfolio = PortfolioManager(total_capital=capital, max_positions=10)
    om = OrderManager(portfolio=portfolio)
    live = LiveTrader(mode=live_mode)
    alerts = AlertSystem()  # no token -> no-op

    common = None
    for d in data.values():
        common = d.index if common is None else common.intersection(d.index)
    common = common.sort_values()

    buy_orders = sell_orders = 0
    rejected = 0
    alert_calls = 0
    last_consensus = []

    for date in common:
        rows = []
        for sym in data:
            if date not in data[sym].index:
                continue
            for sn in signals:
                val = signals[sn].get(sym, pd.Series(0, index=data[sym].index))
                sv = int(val.get(date, 0)) if date in val.index else 0
                if sv == Signal.BUY.value:
                    rows.append((sym, "BUY", sn))
                elif sv in (Signal.SELL.value, Signal.SHORT.value):
                    rows.append((sym, "SELL", sn))
        if not rows:
            continue
        sdf = pd.DataFrame(rows, columns=["Symbol", "Signal", "Strategy"])
        consensus = [c for c in cf.find_consensus(sdf) if c.count >= min_agree]
        day_alert_rows = []
        for c in consensus:
            last_consensus = consensus
            sym = c.symbol
            price = float(data[sym].loc[date, "Close"])
            # OMS decides + (paper) broker + alert
            order = om.process_signal(sym, c.signal_type, price,
                                      "consensus", stop_loss=price * 0.95)
            if order is None:
                rejected += 1
                continue
            # Mirror to (paper) live trader
            live.place_order(sym, c.signal_type, order.quantity, price)
            # Collect for a single end-of-day alert with a Price column
            day_alert_rows.append(
                {"Symbol": sym, "Signal": c.signal_type, "Price": price})
            if order.side == OrderSide.BUY:
                buy_orders += 1
            else:
                sell_orders += 1
        # One alert per day (AlertSystem expects a Price column)
        if day_alert_rows:
            alert_df = pd.DataFrame(day_alert_rows)
            alerts.send_signal_alert(alert_df)
            alert_calls += 1

    # Force-close any open positions at the end
    for sym in list(portfolio.positions.keys()):
        last = data[sym].iloc[-1]["Close"]
        om.process_signal(sym, "SELL", float(last), "close")

    return {
        "buy_orders": buy_orders,
        "sell_orders": sell_orders,
        "rejected": rejected,
        "alert_calls": alert_calls,
        "open_positions_end": len(portfolio.positions),
        "orders_total": len(om.orders),
        "last_consensus_count": len(last_consensus),
    }
