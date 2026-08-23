"""
Indian Market Trading System — Backtesting Engine
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from strategies import Strategy, Signal
from config import TradingCosts


@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    quantity: int
    direction: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    brokerage: float = 0.0
    total_charges: float = 0.0
    holding_days: int = 0
    exit_reason: str = ""


@dataclass
class Position:
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    stop_loss: float
    target_price: Optional[float] = None
    trailing_stop: float = 0.0
    direction: str = "LONG"  # "LONG" or "SHORT"


class Backtester:
    def __init__(self, initial_capital=500000.0, position_size_pct=0.20,
                 costs=None, stop_loss_pct=0.03, use_trailing_stop=True,
                 trailing_stop_pct=0.02):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.costs = costs or TradingCosts()
        self.stop_loss_pct = stop_loss_pct
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct

    def calculate_charges(self, buy_value, sell_value, is_intraday=False):
        c = self.costs
        brokerage = c.brokerage_per_order * 2
        if is_intraday:
            stt = sell_value * c.stt_intraday_sell
        else:
            stt = buy_value * c.stt_delivery_buy + sell_value * c.stt_delivery_sell
        exchange = (buy_value + sell_value) * c.exchange_txn_charge
        sebi = (buy_value + sell_value) * c.sebi_turnover_fee
        stamp = buy_value * c.stamp_duty_buy
        gst = brokerage * c.gst_on_brokerage
        return brokerage + stt + exchange + sebi + stamp + gst

    def run_backtest(self, strategy, data, symbol="STOCK"):
        """Run a single-strategy backtest. Handles LONG and SHORT directions.

        Capital accounting (delivery-style, simplified margin):
          LONG  open : capital -= buy_value;   equity = capital + close*qty
          LONG  close: capital += sell_value - charges
          SHORT open : capital += short_proceeds; equity = capital - close*qty
          SHORT close: capital -= cover_value + charges
        """
        df = strategy.generate_signals(data)
        df = df.dropna()
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = []

        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]
            close = row["Close"]
            signal = row["Signal"]
            exit_reason = None

            # ── Manage open position ──
            if position is not None:
                direction = position.direction

                # Trailing stop update
                if self.use_trailing_stop:
                    if direction == "LONG":
                        new_trailing = close * (1 - self.trailing_stop_pct)
                        position.trailing_stop = max(
                            position.trailing_stop, new_trailing)
                    else:  # SHORT: trailing stop moves DOWN as price falls
                        new_trailing = close * (1 + self.trailing_stop_pct)
                        position.trailing_stop = min(
                            position.trailing_stop, new_trailing)
                    effective_stop = position.trailing_stop
                else:
                    effective_stop = position.stop_loss

                # Stop-loss hit?
                if direction == "LONG" and close <= effective_stop:
                    exit_reason = "LONG_STOP_LOSS"
                elif direction == "SHORT" and close >= effective_stop:
                    exit_reason = "SHORT_STOP_LOSS"

                # Signal exit?
                if direction == "LONG" and signal == Signal.SELL.value:
                    exit_reason = exit_reason or "LONG_SIGNAL"
                elif direction == "SHORT" and signal == Signal.COVER.value:
                    exit_reason = exit_reason or "SHORT_SIGNAL"

                if exit_reason:
                    qty = position.quantity
                    if direction == "LONG":
                        sell_value = close * qty
                        charges = self.calculate_charges(
                            position.entry_price * qty, sell_value)
                        pnl = (close - position.entry_price) * qty - charges
                        capital += sell_value - charges
                    else:  # SHORT
                        cover_value = close * qty
                        charges = self.calculate_charges(
                            cover_value, position.entry_price * qty)
                        pnl = (position.entry_price - close) * qty - charges
                        capital -= cover_value + charges
                    pnl_pct = pnl / (position.entry_price * qty)
                    holding = (date - position.entry_date).days
                    trades.append(Trade(
                        symbol=symbol, entry_date=position.entry_date,
                        entry_price=position.entry_price, exit_date=date,
                        exit_price=close, quantity=qty,
                        direction=direction, pnl=pnl, pnl_pct=pnl_pct,
                        brokerage=charges, total_charges=charges,
                        holding_days=holding, exit_reason=exit_reason
                    ))
                    position = None

            # ── Open new position if flat ──
            if position is None:
                if signal == Signal.BUY.value:
                    position_value = capital * self.position_size_pct
                    quantity = int(position_value / close)
                    if quantity > 0:
                        buy_value = close * quantity
                        stop_loss = close * (1 - self.stop_loss_pct)
                        position = Position(
                            symbol=symbol, entry_date=date, entry_price=close,
                            quantity=quantity, stop_loss=stop_loss,
                            trailing_stop=stop_loss, direction="LONG"
                        )
                        capital -= buy_value
                elif signal == Signal.SHORT.value:
                    position_value = capital * self.position_size_pct
                    quantity = int(position_value / close)
                    if quantity > 0:
                        short_value = close * quantity
                        stop_loss = close * (1 + self.stop_loss_pct)
                        position = Position(
                            symbol=symbol, entry_date=date, entry_price=close,
                            quantity=quantity, stop_loss=stop_loss,
                            trailing_stop=stop_loss, direction="SHORT"
                        )
                        capital += short_value  # proceeds from short sale

            # ── Mark-to-market equity ──
            portfolio_value = capital
            if position is not None:
                if position.direction == "LONG":
                    portfolio_value += close * position.quantity
                else:  # SHORT: net = capital - liability (qty shares @ close)
                    portfolio_value -= close * position.quantity
            equity_curve.append({"Date": date, "Equity": portfolio_value})

        # ── Force-close any open position at end of data ──
        if position is not None:
            last_close = df.iloc[-1]["Close"]
            last_date = df.index[-1]
            qty = position.quantity
            if position.direction == "LONG":
                sell_value = last_close * qty
                charges = self.calculate_charges(
                    position.entry_price * qty, sell_value)
                pnl = (last_close - position.entry_price) * qty - charges
                capital += sell_value - charges
                reason = "END_OF_DATA"
            else:  # SHORT
                cover_value = last_close * qty
                charges = self.calculate_charges(
                    cover_value, position.entry_price * qty)
                pnl = (position.entry_price - last_close) * qty - charges
                capital -= cover_value + charges
                reason = "END_OF_DATA"
            pnl_pct = pnl / (position.entry_price * qty)
            holding = (last_date - position.entry_date).days
            trades.append(Trade(
                symbol=symbol, entry_date=position.entry_date,
                entry_price=position.entry_price, exit_date=last_date,
                exit_price=last_close, quantity=qty,
                direction=position.direction, pnl=pnl, pnl_pct=pnl_pct,
                brokerage=charges, total_charges=charges,
                holding_days=holding, exit_reason=reason
            ))

        eq_df = pd.DataFrame(equity_curve)
        if not eq_df.empty:
            eq_df.set_index("Date", inplace=True)
        metrics = self._calculate_metrics(trades, eq_df, symbol)
        return {"strategy": strategy.name, "symbol": symbol,
                "trades": trades, "equity_curve": eq_df, "metrics": metrics}

    def _calculate_metrics(self, trades, equity_curve, symbol):
        if not trades:
            return {"total_trades": 0, "message": "No trades"}
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_charges = sum(t.total_charges for t in trades)
        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        total_pnl = sum(pnls)
        total_return = total_pnl / self.initial_capital
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses else float("inf")
        avg_holding = np.mean([t.holding_days for t in trades])

        if not equity_curve.empty:
            equity = equity_curve["Equity"]
            daily_returns = equity.pct_change().dropna()
            rolling_max = equity.cummax()
            drawdown = (equity - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            trading_days = len(equity)
            years = trading_days / 252
            cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
            risk_free_daily = 0.06 / 252
            excess_returns = daily_returns - risk_free_daily
            sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                      if excess_returns.std() > 0 else 0)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std()
            sortino = ((daily_returns.mean() - risk_free_daily) / downside_std * np.sqrt(252)
                       if downside_std > 0 else 0)
        else:
            max_drawdown = cagr = sharpe = sortino = 0

        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        return {
            "symbol": symbol, "total_trades": total_trades,
            "winning_trades": len(wins), "losing_trades": len(losses),
            "win_rate": round(win_rate * 100, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return * 100, 2),
            "avg_pnl_per_trade": round(np.mean(pnls), 2),
            "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
            "largest_win": round(max(pnls), 2) if pnls else 0,
            "largest_loss": round(min(pnls), 2) if pnls else 0,
            "profit_factor": round(profit_factor, 2),
            "avg_holding_days": round(avg_holding, 1),
            "total_charges": round(total_charges, 2),
            "cagr_pct": round(cagr * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "initial_capital": self.initial_capital,
            "final_capital": round(self.initial_capital + total_pnl, 2),
            "exit_reasons": exit_reasons,
        }

    def run_multi_stock_backtest(self, strategy, data_dict):
        all_trades = []
        all_equity = []
        per_stock = {}
        for symbol, data in data_dict.items():
            result = self.run_backtest(strategy, data, symbol)
            per_stock[symbol] = result["metrics"]
            all_trades.extend(result["trades"])
            if not result["equity_curve"].empty:
                all_equity.append(result["equity_curve"])

        if all_trades:
            aggregate = self._calculate_metrics(
                all_trades,
                pd.concat(all_equity) if all_equity else pd.DataFrame(),
                "PORTFOLIO"
            )
        else:
            aggregate = {"total_trades": 0}
        return {"strategy": strategy.name, "aggregate_metrics": aggregate,
                "per_stock_metrics": per_stock, "all_trades": all_trades}
