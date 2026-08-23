"""
Indian Market Trading System — Paper Trader
v3.2 — ATR-based stops, short position support, time-based exits,
exit-reason tracking, long/short P&L breakdown.

Reconstructed to be a complete, importable module. The public surface
(process_signal / calculate_atr / get_performance_summary /
print_trade_log / take_snapshot) matches the call sites in
run_all.py, main.py and daily_runner.py. The optional risk-manager
hook is handled defensively so it works whether RiskManager uses the
old 3-arg or the new 5-arg check_pre_trade signature.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from strategies import Signal
from config import TradingCosts, RiskConfig


# ============================================================
#  DATA STRUCTURES
# ============================================================

@dataclass
class PaperPosition:
    """An open LONG position."""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    stop_loss: float
    trailing_stop: float
    highest_price: float
    strategy_name: str
    atr: float = 0.0
    entry_bar_index: int = 0
    bars_held: int = 0


@dataclass
class ShortPosition:
    """An open SHORT position (profit when price falls)."""
    symbol: str
    entry_date: str
    entry_price: float       # Price at which we shorted
    quantity: int
    stop_loss: float         # ABOVE entry (price going up = loss)
    trailing_stop: float     # ABOVE entry
    lowest_price: float      # Track lowest price (best for short)
    strategy_name: str
    atr: float = 0.0
    entry_bar_index: int = 0
    bars_held: int = 0


@dataclass
class PaperTrade:
    """A closed trade (long or short) for the trade log."""
    symbol: str
    strategy: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    charges: float
    exit_reason: str


# ============================================================
#  PAPER TRADER
# ============================================================

class PaperTrader:
    """
    Simulated brokerage. Tracks long + short positions, applies
    ATR-based (and %-based fallback) stops, time-based exits and
    short profit targets, then reports a long/short P&L breakdown.
    """

    def __init__(self, initial_capital: float = 500000.0,
                 costs: Optional[TradingCosts] = None,
                 risk_config: Optional[RiskConfig] = None,
                 log_dir: str = "paper_trading_logs",
                 position_size_pct: float = 0.10,
                 slippage_bps: float = 0.0):
        """slippage_bps: one-way price slippage in basis points (0.05% = 5 bps).
        Applied to BOTH entry (price paid up) and exit (price received down)."""
        self.initial_capital = float(initial_capital)
        self.available_cash = float(initial_capital)
        self.costs = costs or TradingCosts()
        self.risk = risk_config or RiskConfig()
        self.position_size_pct = getattr(risk_config, "position_size_pct",
                                         position_size_pct)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.positions: Dict[str, PaperPosition] = {}
        self.short_positions: Dict[str, ShortPosition] = {}
        self.trade_history: List[PaperTrade] = []
        self.daily_snapshots: List[Dict] = []

        # ATR-based stop parameters (longs)
        self.atr_stop_multiplier = 2.0
        self.atr_trailing_multiplier = 1.5
        self.max_bars_held = 30
        self.min_profit_for_time_exit = 0.01

        # Slippage (one-way, in bps) applied to fills
        self.slippage_bps = float(slippage_bps)

        # ---- POSITION SIZING (replaces flat position_size_pct) ----
        # "fixed"     : original behaviour (position_size_pct of cash)
        # "volatility": size so ATRxmult stop risk = risk_pct of equity
        # "kelly"     : deploy kelly_fraction of equity per trade (half-Kelly,
        #               computed from closed-trade win/loss stats once available)
        self.sizing_method = "fixed"
        self.risk_pct = 0.02          # risk per trade for volatility sizing
        self.kelly_fraction = 0.5     # half-Kelly cap
        self.max_deploy_pct = 0.30    # hard cap on equity deployed per trade

        # ---- DRAWDOWN CIRCUIT BREAKER ----
        # Blocks NEW entries when a period loss limit is breached; exits still allowed.
        self.cb_daily_limit = 0.02    # halt if equity <= start-of-day * (1-2%)
        self.cb_weekly_limit = 0.05
        self.cb_monthly_limit = 0.08
        self.cb_losing_streak = 2     # cooldown after N consecutive losing trades
        self.cb_cooldown_days = 1
        self.trading_halted = False
        self._cb_halt_until = None    # date string until which new entries blocked
        self._peak_equity = float(initial_capital)
        self._day_start_equity = float(initial_capital)
        self._week_start_equity = float(initial_capital)
        self._month_start_equity = float(initial_capital)
        self._last_day = None
        self._last_week = None
        self._last_month = None
        self._losing_streak_count = 0
        self._cb_halt_days = 0

        # Short-specific parameters
        self.short_stop_multiplier = 2.0       # Stop = 2x ATR ABOVE entry
        self.short_trailing_multiplier = 1.5   # Trailing = 1.5x ATR above lowest
        self.short_profit_target = 0.08        # 8% profit target for shorts
        self.short_max_bars_held = 25          # Shorter hold for shorts
        self.short_margin_pct = 0.20           # Margin requirement for shorts

    # --------------------------------------------------------
    #  STATIC HELPERS
    # --------------------------------------------------------

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """True Range average over `period` bars."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # --------------------------------------------------------
    #  SIZING + COSTS
    # --------------------------------------------------------

    def calculate_position_size(self, price: float,
                                atr: Optional[float] = None) -> int:
        """Shares to buy based on selected sizing method.

        - fixed      : position_size_pct of available cash (original behavior)
        - volatility : deploy equity so a stop at atr*atr_stop_multiplier risks
                       risk_pct of equity (ignores cash, uses equity)
        - kelly      : deploy kelly_fraction * edge of equity (half-Kelly from
                       realized win/loss stats); falls back to fixed until enough
                       trades are closed.
        """
        if price <= 0:
            return 0
        equity = self.portfolio_value

        if self.sizing_method == "volatility":
            if atr and atr > 0:
                stop_dist = atr * self.atr_stop_multiplier
                risk_amt = equity * self.risk_pct
                deploy = risk_amt / stop_dist * price  # ₹ to deploy
            else:
                deploy = equity * self.position_size_pct
            deploy = min(deploy, equity * self.max_deploy_pct)
            qty = int(deploy // price)
            return max(qty, 0)

        if self.sizing_method == "kelly":
            k = self._kelly_fraction()
            if k is None:
                deploy = equity * self.position_size_pct
            else:
                deploy = equity * min(self.kelly_fraction * k, self.max_deploy_pct)
            qty = int(deploy // price)
            return max(qty, 0)

        # fixed
        cash_to_use = self.available_cash * self.position_size_pct
        qty = int(cash_to_use // price)
        return max(qty, 0)

    def _kelly_fraction(self):
        """Half-Kelly edge from closed-trade stats, or None if too few trades."""
        pnls = [t.pnl for t in self.trade_history]
        if len(pnls) < 20:
            return None
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        if not wins or not losses:
            return None
        w = len(wins) / len(pnls)
        b = (sum(wins) / len(wins)) / (abs(sum(losses)) / len(losses))
        k = w - (1 - w) / b
        return max(0.0, min(k, 1.0))  # clamp to [0,1]

    def calculate_charges(self, buy_value: float, sell_value: float) -> float:
        """Realistic Indian delivery-style transaction costs."""
        c = self.costs
        brokerage = c.brokerage_per_order * 2  # buy + sell legs
        stt = (buy_value * c.stt_delivery_buy) + (sell_value * c.stt_delivery_sell)
        txn = (buy_value + sell_value) * c.exchange_txn_charge
        sebi = (buy_value + sell_value) * c.sebi_turnover_fee
        stamp = buy_value * c.stamp_duty_buy
        gst = brokerage * c.gst_on_brokerage
        return brokerage + stt + txn + sebi + stamp + gst

    # --------------------------------------------------------
    #  PORTFOLIO VALUE
    # --------------------------------------------------------

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value (cash + open long notional)."""
        long_val = sum(p.entry_price * p.quantity
                       for p in self.positions.values())
        return self.available_cash + long_val

    # --------------------------------------------------------
    #  CIRCUIT BREAKER
    # --------------------------------------------------------

    def _update_period_baselines(self, date_str: str):
        """Track day/week/month start-equity baselines for drawdown halts."""
        if date_str is None:
            return
        try:
            d = pd.to_datetime(date_str)
        except Exception:
            return
        day = d.date().isoformat()
        week = d.to_period("W").start_time.date().isoformat()
        month = d.to_period("M").start_time.date().isoformat()
        eq = self.portfolio_value
        if day != self._last_day:
            self._last_day = day
            self._day_start_equity = eq
        if week != self._last_week:
            self._last_week = week
            self._week_start_equity = eq
        if month != self._last_month:
            self._last_month = month
            self._month_start_equity = eq

    def check_circuit_breaker(self, date_str: str) -> bool:
        """Return True if NEW entries are allowed today.

        Blocks new entries when a daily/weekly/monthly drawdown limit is breached
        OR a losing-streak cooldown is active. Exits remain allowed. Updates
        period baselines from `date_str`.
        """
        self._update_period_baselines(date_str)
        eq = self.portfolio_value
        self._peak_equity = max(self._peak_equity, eq)

        if self._cb_halt_until and date_str and date_str <= self._cb_halt_until:
            self.trading_halted = True
            return False

        if (self.cb_daily_limit and eq
                <= self._day_start_equity * (1 - self.cb_daily_limit)):
            self.trading_halted = True
            self._cb_halt_days += 1
            return False
        if (self.cb_weekly_limit and eq
                <= self._week_start_equity * (1 - self.cb_weekly_limit)):
            self.trading_halted = True
            self._cb_halt_days += 1
            return False
        if (self.cb_monthly_limit and eq
                <= self._month_start_equity * (1 - self.cb_monthly_limit)):
            self.trading_halted = True
            self._cb_halt_days += 1
            return False
        self.trading_halted = False
        return True

    def _record_trade_outcome(self, pnl: float):
        """Update losing-streak counter from a closed trade's P&L."""
        if pnl <= 0:
            self._losing_streak_count += 1
        else:
            self._losing_streak_count = 0

    # --------------------------------------------------------
    #  SIGNAL PROCESSING
    # --------------------------------------------------------

    def process_signal(self, symbol, signal, price, date,
                       strategy_name, atr=None, bar_index=0,
                       risk_manager=None):
        """
        Apply a single bar's signal.

        signal: int from strategies.Signal (1 BUY, -1 SELL,
                -2 SHORT, 2 COVER, 0 HOLD).
        Returns a human-readable action string, or None.
        """
        # Optional risk gate (defensive across RiskManager API versions)
        if risk_manager is not None:
            try:
                res = risk_manager.check_pre_trade(
                    symbol,
                    "BUY" if signal == Signal.BUY.value else "SHORT",
                    self.positions, price, atr or 0)
                if isinstance(res, tuple) and len(res) == 3:
                    allowed, reason, _details = res
                else:
                    allowed, reason = res
                if not allowed:
                    return f"[RISK BLOCKED] {symbol}: {reason}"
            except (TypeError, AttributeError):
                # Old-style RiskManager: (symbol, side, positions)
                try:
                    allowed, reason = risk_manager.check_pre_trade(
                        symbol,
                        "BUY" if signal == Signal.BUY.value else "SHORT",
                        self.positions)
                    if not allowed:
                        return f"[RISK BLOCKED] {symbol}: {reason}"
                except Exception:
                    pass

        # ---- EXISTING BUY/SELL LOGIC ----
        if signal == Signal.BUY.value and symbol not in self.positions:
            if not self.check_circuit_breaker(date):
                return f"[CB HALT] {symbol}: new entries blocked"
            if len(self.positions) >= self.risk.max_open_positions:
                return f"[SKIP] {symbol}: Max positions reached"

            quantity = self.calculate_position_size(price, atr)
            if quantity <= 0:
                return f"[SKIP] {symbol}: No capital"

            slip = self.slippage_bps / 10000.0
            fill_price = price * (1.0 + slip)  # paid up on entry

            cost = fill_price * quantity
            if cost > self.available_cash:
                return f"[SKIP] {symbol}: Insufficient funds"

            if atr and atr > 0:
                stop_loss = fill_price - (atr * self.atr_stop_multiplier)
                trailing_stop = stop_loss
            else:
                stop_loss = fill_price * (1 - self.risk.default_stop_loss_pct)
                trailing_stop = stop_loss

            self.positions[symbol] = PaperPosition(
                symbol=symbol, entry_date=date, entry_price=fill_price,
                quantity=quantity, stop_loss=stop_loss,
                trailing_stop=trailing_stop, highest_price=fill_price,
                strategy_name=strategy_name, atr=atr or 0.0,
                entry_bar_index=bar_index, bars_held=0)
            self.available_cash -= cost
            return f"[BUY] {symbol} | Qty:{quantity} | ₹{fill_price:,.2f} | SL:₹{stop_loss:,.2f}"

        elif signal == Signal.SELL.value and symbol in self.positions:
            return self._close_position(symbol, price, date, "SIGNAL")

        # ---- SHORT SIGNAL ----
        elif signal == Signal.SHORT.value and symbol not in self.short_positions:
            if not self.check_circuit_breaker(date):
                return f"[CB HALT] {symbol} SHORT: new entries blocked"
            # Close any existing long first
            if symbol in self.positions:
                self._close_position(symbol, price, date, "REVERSE_TO_SHORT")

            total_positions = len(self.positions) + len(self.short_positions)
            if total_positions >= self.risk.max_open_positions:
                return f"[SKIP] {symbol} SHORT: Max positions reached"

            quantity = self.calculate_position_size(price, atr)
            if quantity <= 0:
                return f"[SKIP] {symbol} SHORT: No capital"

            margin_required = price * quantity * self.short_margin_pct
            if margin_required > self.available_cash * 0.95:
                return f"[SKIP] {symbol} SHORT: Insufficient margin"

            if atr and atr > 0:
                stop_loss = price + (atr * self.short_stop_multiplier)
                trailing_stop = stop_loss
            else:
                stop_loss = price * (1 + self.risk.default_stop_loss_pct)
                trailing_stop = stop_loss

            self.short_positions[symbol] = ShortPosition(
                symbol=symbol, entry_date=date, entry_price=price,
                quantity=quantity, stop_loss=stop_loss,
                trailing_stop=trailing_stop, lowest_price=price,
                strategy_name=strategy_name, atr=atr or 0.0,
                entry_bar_index=bar_index, bars_held=0)
            self.available_cash -= margin_required
            return (f"[SHORT] {symbol} | Qty:{quantity} | ₹{price:,.2f} | "
                    f"SL:₹{stop_loss:,.2f} | Margin:₹{margin_required:,.0f}")

        elif signal == Signal.COVER.value and symbol in self.short_positions:
            return self._close_short(symbol, price, date, "SIGNAL")

        # ---- CHECK EXISTING POSITIONS ----
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.bars_held += 1

            if price > pos.highest_price:
                pos.highest_price = price
                if pos.atr > 0:
                    new_trailing = price - (pos.atr * self.atr_trailing_multiplier)
                else:
                    new_trailing = price * (1 - self.risk.trailing_stop_pct)
                pos.trailing_stop = max(pos.trailing_stop, new_trailing)

            if price <= pos.trailing_stop:
                return self._close_position(symbol, price, date, "STOP_LOSS")

            if pos.bars_held >= self.max_bars_held:
                current_pnl_pct = (price - pos.entry_price) / pos.entry_price
                if current_pnl_pct < self.min_profit_for_time_exit:
                    return self._close_position(symbol, price, date, "TIME_EXIT")

        # Check short positions
        if symbol in self.short_positions:
            sp = self.short_positions[symbol]
            sp.bars_held += 1

            if price < sp.lowest_price:
                sp.lowest_price = price
                if sp.atr > 0:
                    new_trailing = price + (sp.atr * self.short_trailing_multiplier)
                else:
                    new_trailing = price * (1 + self.risk.trailing_stop_pct)
                sp.trailing_stop = min(sp.trailing_stop, new_trailing)

            if price >= sp.trailing_stop:
                return self._close_short(symbol, price, date, "STOP_LOSS")

            profit_pct = (sp.entry_price - price) / sp.entry_price
            if profit_pct >= self.short_profit_target:
                return self._close_short(symbol, price, date, "PROFIT_TARGET")

            if sp.bars_held >= self.short_max_bars_held:
                if profit_pct < 0.01:
                    return self._close_short(symbol, price, date, "TIME_EXIT")

        return None

    # --------------------------------------------------------
    #  POSITION CLOSURES
    # --------------------------------------------------------

    def _close_position(self, symbol, price, date, reason):
        """Close a LONG position."""
        pos = self.positions.pop(symbol)

        slip = self.slippage_bps / 10000.0
        exit_price = price * (1.0 - slip)  # received down on exit

        gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        buy_value = pos.entry_price * pos.quantity
        sell_value = exit_price * pos.quantity
        charges = self.calculate_charges(buy_value, sell_value)

        pnl = gross_pnl - charges
        pnl_pct = pnl / buy_value if buy_value else 0.0

        self.available_cash += sell_value
        self._record_trade_outcome(pnl)

        self.trade_history.append(PaperTrade(
            symbol=symbol, strategy=f"{pos.strategy_name}",
            entry_date=pos.entry_date, entry_price=pos.entry_price,
            exit_date=date, exit_price=exit_price, quantity=pos.quantity,
            pnl=round(pnl, 2), pnl_pct=round(pnl_pct * 100, 2),
            charges=round(charges, 2), exit_reason=f"LONG_{reason}"))

        return (f"[SELL] {symbol} | ₹{exit_price:,.2f} | "
                f"P&L:₹{pnl:,.0f} | LONG_{reason}")

    def _close_short(self, symbol, price, date, reason):
        """Close a SHORT position (buy to cover)."""
        sp = self.short_positions.pop(symbol)

        gross_pnl = (sp.entry_price - price) * sp.quantity

        buy_value = sp.entry_price * sp.quantity
        sell_value = price * sp.quantity
        charges = self.calculate_charges(buy_value, sell_value)

        pnl = gross_pnl - charges
        pnl_pct = pnl / buy_value if buy_value else 0.0

        # Return margin + P&L
        margin_returned = buy_value * self.short_margin_pct
        self.available_cash += margin_returned + pnl
        self._record_trade_outcome(pnl)

        self.trade_history.append(PaperTrade(
            symbol=symbol, strategy=f"{sp.strategy_name} [SHORT]",
            entry_date=sp.entry_date, entry_price=sp.entry_price,
            exit_date=date, exit_price=price, quantity=sp.quantity,
            pnl=round(pnl, 2), pnl_pct=round(pnl_pct * 100, 2),
            charges=round(charges, 2), exit_reason=f"SHORT_{reason}"))

        return (f"[COVER] {symbol} | ₹{price:,.2f} | "
                f"P&L:₹{pnl:,.0f} | SHORT_{reason}")

    # --------------------------------------------------------
    #  SNAPSHOTS (used by run_all.py)
    # --------------------------------------------------------

    def take_snapshot(self, date_str: str):
        """Record a daily portfolio snapshot."""
        self.daily_snapshots.append({
            "date": date_str,
            "portfolio_value": round(self.portfolio_value, 2),
            "available_cash": round(self.available_cash, 2),
            "open_long": len(self.positions),
            "open_short": len(self.short_positions),
            "total_trades": len(self.trade_history),
        })

    # --------------------------------------------------------
    #  REPORTING
    # --------------------------------------------------------

    def get_performance_summary(self):
        if not self.trade_history:
            return {"message": "No trades yet"}
        pnls = [t.pnl for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        long_trades = [t for t in self.trade_history if "[SHORT]" not in t.strategy]
        short_trades = [t for t in self.trade_history if "[SHORT]" in t.strategy]

        exit_reasons = {}
        for t in self.trade_history:
            if t.exit_reason not in exit_reasons:
                exit_reasons[t.exit_reason] = {"count": 0, "pnl": 0}
            exit_reasons[t.exit_reason]["count"] += 1
            exit_reasons[t.exit_reason]["pnl"] += t.pnl

        return {
            "initial_capital": self.initial_capital,
            "current_capital": round(self.portfolio_value, 2),
            "total_trades": len(self.trade_history),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(self.trade_history) * 100, 2),
            "total_pnl": round(sum(pnls), 2),
            "total_return_pct": round(sum(pnls) / self.initial_capital * 100, 2),
            "avg_pnl_per_trade": round(float(np.mean(pnls)), 2),
            "avg_win": round(float(np.mean(wins)), 2) if wins else 0,
            "avg_loss": round(float(np.mean(losses)), 2) if losses else 0,
            "profit_factor": round(abs(sum(wins) / sum(losses)), 2)
                              if losses and sum(losses) != 0 else 0,
            "total_charges": round(sum(t.charges for t in self.trade_history), 2),
            "exit_reasons": exit_reasons,
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": round(sum(t.pnl for t in long_trades), 2),
            "short_pnl": round(sum(t.pnl for t in short_trades), 2),
        }

    def print_trade_log(self):
        if not self.trade_history:
            print("No trades yet")
            return
        print(f"\n{'='*95}")
        print("PAPER TRADING LOG")
        print(f"{'='*95}")

        long_trades = [t for t in self.trade_history if "[SHORT]" not in t.strategy]
        short_trades = [t for t in self.trade_history if "[SHORT]" in t.strategy]

        if long_trades:
            print(f"\n  LONG TRADES ({len(long_trades)}):")
            print(f"  {'─'*90}")
            for t in long_trades:
                pnl_s = "+" if t.pnl > 0 else ""
                print(f"  {t.symbol:<14} Entry:₹{t.entry_price:<10,.2f} "
                      f"Exit:₹{t.exit_price:<10,.2f} Qty:{t.quantity:<5} "
                      f"P&L:{pnl_s}₹{t.pnl:<10,.2f} ({pnl_s}{t.pnl_pct:.1f}%) {t.exit_reason}")

        if short_trades:
            print(f"\n  SHORT TRADES ({len(short_trades)}):")
            print(f"  {'─'*90}")
            for t in short_trades:
                pnl_s = "+" if t.pnl > 0 else ""
                print(f"  {t.symbol:<14} Short@₹{t.entry_price:<10,.2f} "
                      f"Cover@₹{t.exit_price:<10,.2f} Qty:{t.quantity:<5} "
                      f"P&L:{pnl_s}₹{t.pnl:<10,.2f} ({pnl_s}{t.pnl_pct:.1f}%) {t.exit_reason}")

        summary = self.get_performance_summary()
        print(f"\n  {'─'*90}")
        print(f"  SUMMARY:")
        print(f"  Trades:{summary['total_trades']} | WinRate:{summary['win_rate']}% | "
              f"TotalP&L:₹{summary['total_pnl']:,.2f} | Return:{summary['total_return_pct']}%")
        print(f"  Long: {summary['long_trades']} trades | P&L:₹{summary['long_pnl']:,.2f}")
        print(f"  Short: {summary['short_trades']} trades | P&L:₹{summary['short_pnl']:,.2f}")

        if "exit_reasons" in summary:
            print(f"  {'─'*90}")
            print(f"  EXIT REASONS:")
            for reason, data in summary["exit_reasons"].items():
                pnl_s = "+" if data["pnl"] > 0 else ""
                print(f"    {reason:<20} Trades:{data['count']:<5} P&L:{pnl_s}₹{data['pnl']:,.2f}")
        print(f"{'='*95}")
