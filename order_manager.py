"""
Order Manager — Signal to execution pipeline.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
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
    def __init__(self, portfolio: PortfolioManager, trader=None):
        self.portfolio = portfolio
        self.trader = trader
        self.orders: List[Order] = []
        self._counter = 0

    def process_signal(self, symbol, signal, price, strategy, stop_loss=0.0):
        now = datetime.now().isoformat()
        if signal == "BUY":
            can, reason = self.portfolio.can_add(symbol)
            if not can:
                print(f"  [REJECTED] {symbol}: {reason}"); return None
            qty = self.portfolio.calc_qty(symbol, price)
            if qty <= 0: return None
            self._counter += 1
            order = Order(f"ORD-{self._counter:06d}", symbol, OrderSide.BUY,
                         qty, price, OrderStatus.FILLED, now, strategy, price, stop_loss)
            self.portfolio.positions[symbol] = {"symbol":symbol,"price":price,"qty":qty}
            self.orders.append(order)
            print(f"  [FILLED] BUY {qty}x {symbol} @ ₹{price:,.2f}")
            return order
        elif signal == "SELL" and symbol in self.portfolio.positions:
            pos = self.portfolio.positions.pop(symbol)
            self._counter += 1
            order = Order(f"ORD-{self._counter:06d}", symbol, OrderSide.SELL,
                         pos["qty"], price, OrderStatus.FILLED, now, strategy, price)
            pnl = (price - pos["price"]) * pos["qty"]
            self.orders.append(order)
            print(f"  [FILLED] SELL {pos['qty']}x {symbol} @ ₹{price:,.2f} | P&L:₹{pnl:,.0f}")
            return order
        return None
