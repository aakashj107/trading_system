"""
youtube_batch_extract.py — BATCH YouTube strategy discovery + shortlist.

Given a search query, this:
  1. Searches YouTube (via yt-dlp generic search) for top N videos.
  2. Extracts each transcript (yt-dlp auto-caps).
  3. Scores each for a CLEAR, MECHANICAL, TESTABLE rule (keyword heuristics:
     indicator names, crossovers, thresholds, entry/exit, timeframe).
  4. Shortlists the highest-scoring videos with a one-line rule guess.

Outputs a ranked shortlist to youtube_shortlist.csv. The HUMAN (or agent) then
recreates + OOS-tests the top ones faithfully via strategy_ingest_pipeline.py.

Honest limits:
  - Transcript quality varies (auto-caps are rough).
  - Rule "guess" is heuristic only — faithful recreation still needs manual read.
  - Intraday/non-Indian videos are flagged (our system is daily .NS), not dropped,
    so the user decides.
"""
import warnings
warnings.filterwarnings("ignore")
import subprocess, tempfile, re, csv, os, sys, json


def search_youtube(query: str, n: int = 10) -> list:
    """Return list of (url, title, uploader, views) via yt-dlp generic search."""
    out = subprocess.run(
        ["yt-dlp", f"ytsearch{n}:{query}", "--print",
         "%(webpage_url)s | %(title)s | %(uploader)s | %(view_count)s",
         "--no-warnings"],
        capture_output=True, text=True, timeout=120)
    vids = []
    for line in out.stdout.splitlines():
        parts = line.split(" | ")
        if len(parts) >= 4 and parts[0].startswith("http"):
            vids.append({"url": parts[0], "title": parts[1],
                         "uploader": parts[2], "views": parts[3]})
    return vids


def extract_transcript(url: str, tmpdir: str = "/tmp") -> str:
    base = os.path.join(tmpdir, "yt_batch.%(ext)s")
    try:
        subprocess.run(["yt-dlp", "--skip-download", "--write-auto-subs",
                        "--sub-lang", "en", "--sub-format", "vtt", "-o", base,
                        "--no-warnings", url], capture_output=True, text=True, timeout=120)
        path = os.path.join(tmpdir, "yt_batch.en.vtt")
        txt = open(path, encoding="utf-8").read()
        txt = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", txt)
        txt = re.sub(r"</?c>", "", txt)
        out = []
        for l in txt.splitlines():
            l = l.strip()
            if not l or l.startswith("WEBVTT") or l.startswith("Kind:"):
                continue
            if re.match(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->", l):
                continue
            if re.match(r"^\d{2}:\d{2}\.\d{3}", l):
                continue
            if out and out[-1] == l:
                continue
            out.append(l)
        return " ".join(out)
    except Exception as e:
        return f"[extract failed: {e}]"


# Mechanical-rule signal keywords (weighted)
RULE_SIGNALS = {
    "rsi": 2, "macd": 2, "moving average": 2, "ema": 2, "sma": 2,
    "crossover": 3, "cross above": 3, "cross below": 3,
    "support": 1, "resistance": 1, "breakout": 3, "break of": 2,
    "candle": 1, "engulfing": 2, "hammer": 2, "pivot": 1,
    "atr": 2, "bollinger": 2, "supertrend": 2, "ichimoku": 2,
    "entry": 2, "exit": 2, "stop loss": 2, "target": 1, "take profit": 1,
    "buy when": 3, "sell when": 3, "long when": 3, "short when": 3,
    "above": 1, "below": 1, "period": 1, "lookback": 1, "threshold": 1,
}
INTRADAY_FLAGS = ["15 minute", "15-min", "5 minute", "5-min", "1 minute",
                  "intraday", "scalp", "scalping", "nasdaq", "s&p", "forex",
                  "1m ", "5m ", "15m "]
TESTABLE_HINTS = ["daily", "swing", "positional", "nifty", "bank nifty",
                  "stock", ".ns", "delivery"]


def score_rule(text: str) -> tuple:
    low = text.lower()
    score = 0
    hits = []
    for kw, w in RULE_SIGNALS.items():
        if kw in low:
            score += w
            hits.append(kw)
    intraday = any(f in low for f in INTRADAY_FLAGS)
    testable = any(h in low for h in TESTABLE_HINTS)
    # penalize if clearly intraday/non-indian but not clearly testable
    if intraday and not testable:
        score -= 2
    return score, hits, intraday, testable


def main(query: str = "nifty trading strategy backtest rules", n: int = 10):
    print(f"Searching YouTube: '{query}' (top {n})...")
    vids = search_youtube(query, n)
    print(f"  found {len(vids)} videos")
    rows = []
    for v in vids:
        print(f"  transcript: {v['title'][:50]}...")
        txt = extract_transcript(v["url"])
        if txt.startswith("[extract failed"):
            print(f"    SKIP ({txt})")
            continue
        score, hits, intraday, testable = score_rule(txt)
        rows.append({
            "url": v["url"], "title": v["title"], "uploader": v["uploader"],
            "views": v["views"], "score": score, "signals": ",".join(hits[:8]),
            "intraday_nonindian": intraday, "testable_daily_ns": testable,
            "chars": len(txt),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    if not rows:
        print("\nNo transcripts extracted (all failed). Check yt-dlp/network.")
        return rows
    with open("youtube_shortlist.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSHORTLIST (top by mechanical-rule score):")
    for r in rows[:8]:
        flag = "INTRADAY/NON-IN" if r["intraday_nonindian"] and not r["testable_daily_ns"] else ("DAILY/IN" if r["testable_daily_ns"] else "")
        print(f"  [{r['score']:2d}] {r['title'][:48]:48s} | {flag:14s} | {r['signals'][:40]}")
    print(f"\nSaved youtube_shortlist.csv ({len(rows)} videos scored)")
    return rows


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "nifty trading strategy backtest rules"
    nn = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(q, nn)
