"""
Trade Logger — CSV logging for all signals and trades.

Tracks:
  1. signals.csv   — Every signal generated (entry/exit, price, strategy, expected P&L)
  2. trades.csv    — Every executed trade (entry/exit, actual P&L, charges, hold days)

Both files append-only and auto-create headers on first write.
"""
import os
import csv
import json
from datetime import datetime
from typing import Dict, List, Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")
_LOG_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_LOG_ROOT, "trade_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Persistent open-position state — survives process restarts and is carried
# across GitHub Actions runs via actions/cache (trade_logs/ + live_state/)
STATE_DIR = os.path.join(_LOG_ROOT, "live_state")
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "open_positions.json")

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
    "hold_days", "exit_reason", "strategy", "consensus_strategies", "regime", "cycle",
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
    consensus_strategies: List[str] = None,
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
        " | ".join(consensus_strategies[:8]) if consensus_strategies else "",
        regime,
        cycle,
    ]

    with open(TRADES_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


def load_open_positions() -> Dict:
    """Load open-position state from disk (empty dict if missing/corrupt).
    Normalises legacy key names so callers always see entry_price/quantity."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        for sym, p in data.items():
            if isinstance(p, dict):
                if "entry_price" not in p and "price" in p:
                    p["entry_price"] = p.pop("price")
                if "quantity" not in p and "qty" in p:
                    p["quantity"] = p.pop("qty")
        return data
    except (json.JSONDecodeError, OSError):
        return {}


def save_open_positions(positions: Dict):
    """Persist open-position state to disk."""
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def record_entry(symbol: str, entry_price: float, quantity: int,
                 entry_date: str, stop_loss: float = 0.0,
                 target: float = 0.0, strategy: str = ""):
    """Record an open position in persistent state."""
    positions = load_open_positions()
    positions[symbol] = {
        "entry_price": round(entry_price, 2),
        "quantity": quantity,
        "entry_date": entry_date,
        "stop_loss": round(stop_loss, 2) if stop_loss else 0.0,
        "target": round(target, 2) if target else 0.0,
        "strategy": strategy,
    }
    save_open_positions(positions)


def pop_entry(symbol: str) -> Optional[Dict]:
    """Remove and return a symbol's entry info (None if not tracked)."""
    positions = load_open_positions()
    info = positions.pop(symbol, None)
    if info is not None:
        save_open_positions(positions)
    return info


SNAPSHOT_FILE = os.path.join(LOG_DIR, "equity.csv")
SNAPSHOT_HEADERS = [
    "date", "time", "cycle", "regime", "vix", "risk_mode",
    "symbols_scanned", "raw_signals", "consensus_signals",
    "buy_orders", "sell_orders", "exits", "rejected",
    "open_positions", "invested_value", "circuit_breaker",
]


def log_snapshot(cycle: int, regime: str, vix: Optional[float],
                 risk_mode: str, symbols_scanned: int, raw_signals: int,
                 consensus_signals: int, buy_orders: int, sell_orders: int,
                 exits: int, rejected: int, open_positions: int,
                 invested_value: float, circuit_breaker: bool):
    """Append one cycle-health snapshot row to equity.csv."""
    _ensure_header(SNAPSHOT_FILE, SNAPSHOT_HEADERS)
    now = datetime.now(IST)
    row = [
        now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), cycle,
        regime, vix if vix is not None else "", risk_mode,
        symbols_scanned, raw_signals, consensus_signals,
        buy_orders, sell_orders, exits, rejected,
        open_positions, round(invested_value, 2), circuit_breaker,
    ]
    with open(SNAPSHOT_FILE, "a", newline="") as f:
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
