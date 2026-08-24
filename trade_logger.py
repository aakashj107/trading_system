"""
Trade Logger — CSV logging for all signals and trades.

Tracks:
  1. signals.csv   — Every signal generated (entry/exit, price, strategy, expected P&L)
  2. trades.csv    — Every executed trade (entry/exit, actual P&L, charges, hold days)

Both files append-only and auto-create headers on first write.
"""
import os
import csv
from datetime import datetime
from typing import Dict, List, Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_logs")
os.makedirs(LOG_DIR, exist_ok=True)

SIGNALS_FILE = os.path.join(LOG_DIR, "signals.csv")
TRADES_FILE = os.path.join(LOG_DIR, "trades.csv")

SIGNAL_HEADERS = [
    "date", "scan_time", "symbol", "signal", "strength", "consensus_count",
    "price", "stop_loss", "target_price", "expected_return_pct",
    "strategies", "regime", "universe", "cycle",
]

TRADE_HEADERS = [
    "date", "scan_time", "symbol", "action", "entry_date", "entry_price",
    "exit_date", "exit_price", "quantity", "pnl", "pnl_pct", "charges",
    "hold_days", "exit_reason", "strategy", "regime", "cycle",
]


def _ensure_header(filepath: str, headers: List[str]):
    """Write header row if file doesn't exist or is empty."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(headers)


def log_signal(
    symbol: str,
    signal: str,
    strength: str,
    consensus_count: int,
    price: float,
    strategies: List[str],
    regime: str,
    universe: str,
    cycle: int,
    stop_loss_pct: float = 0.05,
    target_pct: float = 0.08,
):
    """Log a consensus signal to signals.csv."""
    _ensure_header(SIGNALS_FILE, SIGNAL_HEADERS)

    now = datetime.now(IST)
    stop_loss = round(price * (1 - stop_loss_pct), 2) if signal == "BUY" else round(price * (1 + stop_loss_pct), 2)
    target = round(price * (1 + target_pct), 2) if signal == "BUY" else round(price * (1 - target_pct), 2)
    expected_return = round(target_pct * 100, 2)

    row = [
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        symbol,
        signal,
        strength,
        consensus_count,
        round(price, 2),
        stop_loss,
        target,
        expected_return,
        " | ".join(strategies[:5]),
        regime,
        universe,
        cycle,
    ]

    with open(SIGNALS_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


def log_trade(
    symbol: str,
    action: str,
    entry_date: str,
    entry_price: float,
    quantity: int,
    strategy: str,
    regime: str,
    cycle: int,
    exit_date: str = "",
    exit_price: float = 0.0,
    pnl: float = 0.0,
    pnl_pct: float = 0.0,
    charges: float = 0.0,
    hold_days: int = 0,
    exit_reason: str = "",
):
    """Log a trade to trades.csv."""
    _ensure_header(TRADES_FILE, TRADE_HEADERS)

    now = datetime.now(IST)
    row = [
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        symbol,
        action,
        entry_date,
        round(entry_price, 2),
        exit_date,
        round(exit_price, 2),
        quantity,
        round(pnl, 2),
        round(pnl_pct, 2),
        round(charges, 2),
        hold_days,
        exit_reason,
        strategy,
        regime,
        cycle,
    ]

    with open(TRADES_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


def get_signals_summary() -> Dict:
    """Read signals.csv and return summary stats."""
    if not os.path.exists(SIGNALS_FILE):
        return {"total": 0}

    with open(SIGNALS_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"total": 0}

    buys = [r for r in rows if r["signal"] == "BUY"]
    sells = [r for r in rows if r["signal"] == "SELL"]
    symbols = set(r["symbol"] for r in rows)

    return {
        "total": len(rows),
        "buys": len(buys),
        "sells": len(sells),
        "unique_symbols": len(symbols),
        "symbols": list(symbols),
    }


def get_trades_summary() -> Dict:
    """Read trades.csv and return P&L summary."""
    if not os.path.exists(TRADES_FILE):
        return {"total": 0}

    with open(TRADES_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"total": 0}

    pnls = [float(r["pnl"]) for r in rows if r["pnl"]]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    return {
        "total": len(rows),
        "total_pnl": round(sum(pnls), 2),
        "winning": len(wins),
        "losing": len(losses),
        "win_rate": round(len(wins) / len(rows) * 100, 1) if rows else 0,
        "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0,
    }
