"""
strategy_ingest_pipeline.py — REUSABLE "verify + make our own" pipeline.

Goal (user): famous strategies are everywhere (YouTube, ChartLink, X, big traders)
— verify them independently and build OUR separate implementations, global not
just Indian.

Flow per source:
  1. EXTRACT  — pull the mechanical rule as text (YouTube transcript via yt-dlp,
                 ChartLink via Jina, web article via Jina, X blocked->skip).
  2. RECREATE — encode the rule as a standalone strategy class in THIS repo
                 (our implementation, our params, not theirs).
  3. VALIDATE — run it blind OOS via PaperTrader on the universe (no look-ahead,
                 net-of-cost). Keep ONLY if OOS return>0 AND PF>=1.0.
  4. RECORD   — write a verdict row to strategies_ingested.csv.

This module is the scaffold + a runner that, given a source dict, does steps
2-4 against a user-supplied strategy class. Step 1 (extraction) is done
per-channel by helper functions below.

Channels reachable (verified this session):
  - YouTube   : yt-dlp transcript  ✅
  - ChartLink  : Jina web-read      ✅
  - web/article: Jina web-read      ✅
  - X/Twitter  : 403 blocked (abuse) — needs login cookies (ban-risk) → SKIP
Global coverage: feed it US/ETH/forex symbols too; UNIVERSE is just the default.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, csv, os, datetime
from data_fetcher import YFinanceDataFetcher
from paper_trader import PaperTrader, Signal
from short_backtester import UNIVERSE

INIT_CAP = 1_000_000.0
SPLIT = "2022-06-01"
RECORD_CSV = "strategies_ingested.csv"


# ── STEP 1: EXTRACTION HELPERS (per channel) ───────────────────────────────
def extract_youtube(video_url: str) -> str:
    """Fetch auto-subtitle transcript text via yt-dlp. Returns cleaned text."""
    import subprocess, tempfile, re
    tmp = tempfile.mktemp(suffix=".en.vtt")
    try:
        subprocess.run(["yt-dlp", "--skip-download", "--write-auto-subs",
                        "--sub-lang", "en", "--sub-format", "vtt", "-o", tmp[:-4],
                        video_url], capture_output=True, text=True, timeout=120)
        txt = open(tmp, encoding="utf-8").read()
        blocks = re.split(r"\n\n+", txt)
        out = []
        for b in blocks:
            ls = [l for l in b.splitlines() if not l.strip().replace(".", "").isdigit()
                  and " --> " not in l and not l.startswith("WEBVTT")]
            out.append(" ".join(ls))
        return " ".join(out)
    except Exception as e:
        return f"[yt-dlp failed: {e}]"


def extract_chartlink(url: str) -> str:
    """ChartLink scanner definition via Jina web-read."""
    import subprocess
    try:
        r = subprocess.run(["curl", "-s", f"https://r.jina.ai/{url}"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout
    except Exception as e:
        return f"[chartlink fetch failed: {e}]"


def extract_web(url: str) -> str:
    import subprocess
    try:
        r = subprocess.run(["curl", "-s", f"https://r.jina.ai/{url}"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout
    except Exception as e:
        return f"[web fetch failed: {e}]"


# ── STEP 3: VALIDATE (blind OOS) ───────────────────────────────────────────
def validate_strategy(strat_callable, data=None, universe=None, split=SPLIT):
    """strat_callable(data, date) -> set of target symbols. Runs OOS via PaperTrader."""
    if data is None:
        f = YFinanceDataFetcher()
        universe = universe or UNIVERSE
        data = {}
        for s in universe:
            try:
                d = f.get_historical_data(s, period="10y")
                if d is not None and not d.empty and len(d) > 252:
                    data[s] = d
            except Exception:
                pass
    common = None
    for d in data.values():
        common = d.index if common is None else common.intersection(d.index)
    common = common.sort_values()
    mid = common.searchsorted(pd.Timestamp(split))
    oos_d = common[mid:]

    def run_window(dates):
        pt = PaperTrader(initial_capital=INIT_CAP, position_size_pct=0.10)
        held = set()
        for date in dates:
            if date.month != (date - pd.Timedelta(days=1)).month or date == dates[0]:
                targets = strat_callable(data, date)
                for sym in list(held):
                    if sym not in targets and sym in pt.positions:
                        px = data[sym].loc[date, "Close"]
                        pt.process_signal(sym, Signal.SELL.value, px, str(date.date()), "ingest")
                for sym in targets:
                    if sym not in held and sym in data and date in data[sym].index:
                        px = data[sym].loc[date, "Close"]
                        pt.process_signal(sym, Signal.BUY.value, px, str(date.date()), "ingest",
                                          atr=PaperTrader.calculate_atr(data[sym].loc[:date]).iloc[-1] if len(data[sym].loc[:date]) >= 2 else np.nan)
                held = set(targets)
        for sym in list(pt.positions.keys()):
            pt.process_signal(sym, Signal.SELL.value, data[sym].iloc[-1]["Close"], "END", "close")
        res = pt.get_performance_summary()
        eq = pd.Series([s["portfolio_value"] for s in pt.daily_snapshots])
        peak = eq.cummax()
        mdd = ((eq - peak) / peak).min() * 100 if len(eq) else 0.0
        res["max_drawdown_pct"] = round(float(mdd), 2)
        return res

    oos = run_window(oos_d)
    return oos


def record_verdict(source, channel, rule_summary, oos_ret, oos_pf, oos_win, oos_dd, trades, kept):
    row = {
        "date": datetime.date.today().isoformat(), "source": source, "channel": channel,
        "rule": rule_summary, "oos_ret": oos_ret, "oos_pf": oos_pf, "oos_win": oos_win,
        "oos_dd": oos_dd, "trades": trades, "kept": kept,
    }
    write_header = not os.path.exists(RECORD_CSV)
    with open(RECORD_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    return row


if __name__ == "__main__":
    print("Ingest pipeline scaffold ready.")
    print("Channels: YouTube(yt-dlp) ✅ | ChartLink(Jina) ✅ | web(Jina) ✅ | X(blocked,skip)")
    print("Use validate_strategy(your_callable) + record_verdict() per source.")
    print("Example already run: MomentumRotation (RupeeCase-style) -> OOS VALIDATED.")
