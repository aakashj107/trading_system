"""
Order Manager — Signal to execution pipeline.

v3.3 — Risk-aware:
  • Entries pass through RiskManager.check_pre_trade (circuit breaker,
    correlation limits) when a risk_manager is provided.
  • Realized P&L from closed trades is reported to the RiskManager so
    daily-loss / drawdown circuit breakers actually trip.
  • Stop losses are stored on the position and exposed via open_stop_orders()
    for monitoring by live_runner.
  • Optional slippage on fills (default 0 = exact price, unchanged behavior).
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from enum import Enum
from portfolio_manager import PortfolioManager
from config import TradingCosts


class OrderStatus(Enum):
    PENDING="PENDING"; FILLED="FILLED"; CANCELLED="CANCELLED"


class OrderSide(Enum):
    BUY="BUY"; SELL="SELL"


@dataclass
class Order:
    order_id: str; symbol: str; side: OrderSide; quantity: int
    price: float; status: OrderStatus; timestamp: str; strategy: str
    fill_price: float = 0.0; stop_loss: float = 0.0


class OrderManager:
    def __init__(self, portfolio: PortfolioManager, trader=None,
                 risk_manager=None, slippage_pct: float = 0.0):
        self.portfolio = portfolio
        self.trader = trader
        self.risk_manager = risk_manager
        self.slippage_pct = slippage_pct
        self.orders: List[Order] = []
        self._counter = 0

    def _fill_price(self, price: float, side: OrderSide) -> float:
        """Apply slippage: pay more when buying, receive less when selling."""
        if side == OrderSide.BUY:
            return round(price * (1 + self.slippage_pct), 2)
        return round(price * (1 - self.slippage_pct), 2)

    def process_signal(self, symbol, signal, price, strategy, stop_loss=0.0,
                       target=0.0):
        now = datetime.now().isoformat()
        if signal == "BUY":
            # Risk gate (circuit breaker + correlation limits)
            if self.risk_manager is not None:
                allowed, reason, _ = self.risk_manager.check_pre_trade(
                    symbol, "BUY", list(self.portfolio.positions.keys()), price)
                if not allowed:
                    print(f"  [RISK-BLOCKED] {symbol}: {reason}"); return None
            can, reason = self.portfolio.can_add(symbol)
            if not can:
                print(f"  [REJECTED] {symbol}: {reason}"); return None
            qty = self.portfolio.calc_qty(symbol, price)
            if qty <= 0: return None
            fill = self._fill_price(price, OrderSide.BUY)
            self._counter += 1
            order = Order(f"ORD-{self._counter:06d}", symbol, OrderSide.BUY,
                         qty, price, OrderStatus.FILLED, now, strategy, fill, stop_loss)
            self.portfolio.positions[symbol] = {"symbol":symbol,"price":fill,"qty":qty,
                                                "stop_loss":stop_loss,"target":target}
            self.orders.append(order)
            print(f"  [FILLED] BUY {qty}x {symbol} @ ₹{fill:,.2f}")
            return order
        elif signal == "SELL" and symbol in self.portfolio.positions:
            pos = self.portfolio.positions.pop(symbol)
            fill = self._fill_price(price, OrderSide.SELL)
            self._counter += 1
            order = Order(f"ORD-{self._counter:06d}", symbol, OrderSide.SELL,
                         pos["qty"], price, OrderStatus.FILLED, now, strategy, fill)
            pnl = (fill - pos["price"]) * pos["qty"]
            if self.risk_manager is not None:
                self.risk_manager.record_realized_pnl(pnl)
            self.orders.append(order)
            print(f"  [FILLED] SELL {pos['qty']}x {symbol} @ ₹{fill:,.2f} | P&L:₹{pnl:,.0f}")
            return order
        return None

    def check_exits(self, prices: Dict[str, float]) -> List[Tuple[str, float, str]]:
        """
        Check open positions against stop losses AND profit targets.

        Args:
            prices: {symbol: current_price} for positions to check.

        Returns:
            List of (symbol, current_price, reason) where reason is
            "STOP_LOSS" or "TARGET".
        """
        exits = []
        for sym, pos in self.portfolio.positions.items():
            cur = prices.get(sym)
            if cur is None:
                continue
            stop = pos.get("stop_loss", 0.0)
            target = pos.get("target", 0.0)
            if stop and cur <= stop:
                exits.append((sym, cur, "STOP_LOSS"))
            elif target and cur >= target:
                exits.append((sym, cur, "TARGET"))
        return exits

    def check_stops(self, prices: Dict[str, float]) -> List[Tuple[str, float]]:
        """Backward-compatible stop-only check."""
        return [(s, p) for s, p, r in self.check_exits(prices) if r == "STOP_LOSS"]
