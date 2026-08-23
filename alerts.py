"""
Telegram Alert System — v2 (added live trade alerts).
"""
import requests
import os
from datetime import datetime


class AlertSystem:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    def send_message(self, message):
        if not self.bot_token or not self.chat_id:
            return False
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False

    def send_signal_alert(self, signals_df):
        if signals_df.empty:
            return
        msg = "*Trading Alert*\n\n"
        for sig in ["BUY", "SELL"]:
            sub = signals_df[signals_df["Signal"] == sig]
            if sub.empty:
                continue
            msg += f"*{sig}:*\n"
            for _, r in sub.iterrows():
                msg += f"  {r['Symbol']} @ ₹{r['Price']:,.2f}\n"
        return self.send_message(msg)

    def send_live_trade_alert(self, executed_signals: list, regime: str):
        """
        Send alert for live trade executions.

        Args:
            executed_signals: list of dicts with keys:
                symbol, signal, price, quantity, strength, count, strategies
            regime: current market regime
        """
        if not executed_signals:
            return

        now = datetime.now().strftime("%H:%M:%S")
        msg = f"*Live Trade Signal* ({now})\nRegime: {regime}\n\n"

        buys = [s for s in executed_signals if s["signal"] == "BUY"]
        sells = [s for s in executed_signals if s["signal"] == "SELL"]

        if buys:
            msg += "*BUY:*\n"
            for s in buys:
                strats = ", ".join(s.get("strategies", [])[:3])
                msg += (
                    f"  {s['symbol']} @ ₹{s['price']:,.2f} x {s['quantity']}\n"
                    f"    Strength: {s['strength']} ({s['count']} agree) — {strats}\n"
                )

        if sells:
            msg += "*SELL:*\n"
            for s in sells:
                strats = ", ".join(s.get("strategies", [])[:3])
                msg += (
                    f"  {s['symbol']} @ ₹{s['price']:,.2f} x {s['quantity']}\n"
                    f"    Strength: {s['strength']} ({s['count']} agree) — {strats}\n"
                )

        return self.send_message(msg)

    def send_daily_summary(self, value, pnl, positions, trades):
        msg = (
            f"*Daily Summary*\nPortfolio: ₹{value:,.0f}\n"
            f"P&L: {'+' if pnl > 0 else ''}₹{pnl:,.0f}\n"
            f"Positions: {positions} | Trades: {trades}"
        )
        return self.send_message(msg)
