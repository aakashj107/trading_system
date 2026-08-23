"""
Portfolio Manager — Multi-stock allocation with sector limits.
"""
from typing import Dict, Tuple
from config import RiskConfig


SECTOR_MAP = {
    "RELIANCE.NS":"Oil & Gas","TCS.NS":"IT","HDFCBANK.NS":"Banking",
    "INFY.NS":"IT","ICICIBANK.NS":"Banking","HINDUNILVR.NS":"FMCG",
    "ITC.NS":"FMCG","SBIN.NS":"Banking","BHARTIARTL.NS":"Telecom",
    "KOTAKBANK.NS":"Banking","LT.NS":"Construction","AXISBANK.NS":"Banking",
    "BAJFINANCE.NS":"NBFC","ASIANPAINT.NS":"Paint","MARUTI.NS":"Auto",
    "HCLTECH.NS":"IT","SUNPHARMA.NS":"Pharma","TITAN.NS":"Consumer",
    "TATAMOTORS.NS":"Auto","WIPRO.NS":"IT","ULTRACEMCO.NS":"Cement",
    "NTPC.NS":"Power","ONGC.NS":"Oil & Gas","POWERGRID.NS":"Power",
    "M&M.NS":"Auto","JSWSTEEL.NS":"Metals","TATASTEEL.NS":"Metals",
}


class PortfolioManager:
    def __init__(self, total_capital=500000.0, risk_config=None,
                 max_sector_alloc=0.30, max_single=0.20, max_positions=8):
        self.total_capital = total_capital
        self.risk = risk_config or RiskConfig()
        self.max_sector_alloc = max_sector_alloc
        self.max_single = max_single
        self.max_positions = max_positions
        self.positions: Dict = {}

    def can_add(self, symbol):
        if len(self.positions) >= self.max_positions:
            return False, "Max positions reached"
        if symbol in self.positions:
            return False, "Already holding"
        sector = SECTOR_MAP.get(symbol, "Unknown")
        exposure = sum(p["price"]*p["qty"] for p in self.positions.values()
                       if SECTOR_MAP.get(p["symbol"],"")==sector) / self.total_capital
        if exposure >= self.max_sector_alloc:
            return False, f"Sector {sector} at limit"
        return True, "OK"

    def calc_qty(self, symbol, price):
        val = self.total_capital * self.max_single
        return max(int(val / price), 0)

    def print_portfolio(self):
        total = sum(p["price"]*p["qty"] for p in self.positions.values())
        print(f"\n  Portfolio: {len(self.positions)} positions, ₹{total:,.0f} invested")
        for sym, p in self.positions.items():
            print(f"    {sym:<16} Qty:{p['qty']:<5} ₹{p['price']:,.2f}")
