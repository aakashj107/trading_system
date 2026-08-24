"""
Daily Report — Markdown summary of today's signals, trades and open positions.

Reads trade_logs/signals.csv, trade_logs/trades.csv and
live_state/open_positions.json, filters to --date (default: today IST),
and writes markdown suitable for $GITHUB_STEP_SUMMARY.

Usage:
    python daily_report.py                    # today's report
    python daily_report.py --date 2026-08-25  # specific date
    python daily_report.py --all              # everything on record

Pure standard library — safe to run in CI without heavy deps.
"""
import os
import csv
import json
import argparse
from datetime import datetime
from typing import List, Dict

import pytz

IST = pytz.timezone("Asia/Kolkata")
ROOT = os.path.dirname(os.path.abspath(__file__))
SIGNALS_FILE = os.path.join(ROOT, "trade_logs", "signals.csv")
TRADES_FILE = os.path.join(ROOT, "trade_logs", "trades.csv")
SNAPSHOT_FILE = os.path.join(ROOT, "trade_logs", "equity.csv")
STATE_FILE = os.path.join(ROOT, "live_state", "open_positions.json")


def _read_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fmt(v, prefix="", suffix="", na="-"):
    if v in ("", None):
        return na
    return f"{prefix}{v}{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Daily trading report (markdown)")
    parser.add_argument("--date", default=datetime.now(IST).strftime("%Y-%m-%d"))
    parser.add_argument("--all", action="store_true", help="Report all dates")
    args = parser.parse_args()
    date_filter = None if args.all else args.date

    print(f"# Trading Report — {date_filter or 'All dates'}")
    print()

    # ── Open positions ─────────────────────────────────────────
    open_pos = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                open_pos = json.load(f)
        except json.JSONDecodeError:
            pass

    print("## Open Positions (%d)" % len(open_pos))
    if open_pos:
        print()
        print("| Symbol | Entry Date | Entry | Qty | Stop Loss | Target | Risk/Share |")
        print("|---|---|---|---|---|---|---|")
        for sym, p in sorted(open_pos.items()):
            risk = ""
            try:
                risk = _fmt(round(float(p["entry_price"]) - float(p.get("stop_loss") or 0), 2), prefix="₹")
            except (KeyError, TypeError, ValueError):
                pass
            tgt = _fmt(p.get("target"), prefix="₹") if p.get("target") else "-"
            print(f"| {sym} | {_fmt(p.get('entry_date'))} | ₹{p['entry_price']} "
                  f"| {p['quantity']} | ₹{p.get('stop_loss', '-')} | {tgt} | {risk} |")
    else:
        print("_None_")
    print()

    # ── Signals today ──────────────────────────────────────────
    signals = [r for r in _read_csv(SIGNALS_FILE)
               if date_filter is None or r["date"] == date_filter]
    print("## Signals Today (%d)" % len(signals))
    if signals:
        print()
        print("| Time | Symbol | Signal | Strength | Agree | Price | Stop | Target | Exp. Return | Strategies |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in signals:
            strats = r["strategies"].replace("|", "/")
            print(f"| {r['scan_time']} | {r['symbol']} | {r['signal']} | {r['strength']} "
                  f"| {r['consensus_count']} | ₹{r['price']} | ₹{r['stop_loss']} "
                  f"| ₹{r['target_price']} | {r['expected_return_pct']}% "
                  f"| {_fmt(strats)} |")
    else:
        print("_None_")
    print()

    # ── Closed trades today ────────────────────────────────────
    trades = [r for r in _read_csv(TRADES_FILE)
              if (date_filter is None or r["date"] == date_filter)
              and r["action"] == "SELL"]
    print("## Trades Closed Today (%d)" % len(trades))
    if trades:
        print()
        print("| Exit Time | Symbol | Entry → Exit | P&L | P&L % | Hold Days | Reason |")
        print("|---|---|---|---|---|---|---|")
        for r in trades:
            entry_txt = f"₹{r['entry_price']}" if r["entry_price"] not in ("", "0.0", "0") else "?"
            pnl_color = "" if not r["pnl"] else (
                "🟢 +" if float(r["pnl"]) > 0 else "🔴 ")
            print(f"| {r['scan_time']} | {r['symbol']} | {entry_txt} → ₹{r['exit_price']} "
                  f"| {pnl_color}₹{r['pnl']} | {pnl_color}{r['pnl_pct']}% "
                  f"| {r['hold_days'] or '-'} | {r['exit_reason'] or '-'} |")
    else:
        print("_None_")
    print()

    # ── Cycle health (recent snapshots) ─────────────────────────
    snapshots = [r for r in _read_csv(SNAPSHOT_FILE)
                 if date_filter is None or r["date"] == date_filter]
    print("## Cycle Health (%s)" % (date_filter or "all dates"))
    if snapshots:
        print()
        print("| Time | Cyc | VIX | Mode | Scanned | Signals | Consensus | Buys | Exits | Open | Breaker |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in snapshots[-12:]:
            breaker = "⚠️ YES" if r["circuit_breaker"] in ("True", "true") else "ok"
            print(f"| {r['time']} | {r['cycle']} | {_fmt(r['vix'])} | {_fmt(r['risk_mode'])} "
                  f"| {r['symbols_scanned']} | {r['raw_signals']} | {r['consensus_signals']} "
                  f"| {r['buy_orders']} | {r['exits']} | {r['open_positions']} | {breaker} |")
        failed = sum(1 for r in snapshots if r["symbols_scanned"] == "0")
        if failed:
            print()
            print(f"⚠️ **{failed} cycle(s) scanned 0 symbols — check data/API health**")
    else:
        print("_No cycles recorded yet_")
    print()

    # ── Cumulative summary ─────────────────────────────────────
    all_closed = [r for r in _read_csv(TRADES_FILE) if r["action"] == "SELL" and r["pnl"]]
    if all_closed:
        pnls = [float(r["pnl"]) for r in all_closed]
        wins = [p for p in pnls if p > 0]
        total_ret = sum(float(r["pnl_pct"]) for r in all_closed)
        print("## All-Time Paper P&L")
        print()
        print(f"- **Closed trades:** {len(all_closed)}")
        print(f"- **Win rate:** {len(wins)}/{len(pnls)} ({len(wins)/len(pnls)*100:.1f}%)")
        print(f"- **Total P&L:** ₹{sum(pnls):,.2f}")
        print(f"- **Avg return/trade:** {total_ret/len(all_closed):+.2f}%")
        print()


if __name__ == "__main__":
    main()
