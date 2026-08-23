"""app.py — Streamlit web UI: paste/scrape a TradingView indicator, backtest it.

Run:  streamlit run app.py   (from the trading_system dir, venv active)
"""

import os
import sys

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import pandas as pd
import streamlit as st

# local package
from pine_web.pine_parser import parse, summarize
from pine_web.pine_transpiler import transpile, TranspileError
from pine_web.tv_scraper import fetch_pine_source
from pine_web.run_backtest import run_pine_backtest, metrics_summary

from data_fetcher import YFinanceDataFetcher


st.set_page_config(page_title="Pine → Backtest", layout="wide")
st.title("TradingView Pine → Backtest Pipeline")


def load_sample():
    return '''//@version=5
strategy("RSI Pullback", overlay=true)
rsiLen = input.int(14, title="RSI Length")
rsiLevel = input.int(30, title="RSI Oversold")
maLen = input.int(50, title="MA Length")
rsiVal = ta.rsi(close, rsiLen)
ma = ta.sma(close, maLen)
longCond  = ta.crossunder(rsiVal, rsiLevel) and close > ma
shortCond = ta.cross(ma, close)
if (longCond)
    strategy.entry("Long", strategy.long)
if (shortCond)
    strategy.close("Long")
'''


# ── Sidebar: data source ───────────────────────────────────────────────────
st.sidebar.header("Data")
symbol = st.sidebar.text_input("Symbol (yfinance)", value="RELIANCE.NS")
period = st.sidebar.selectbox("History", ["1y", "2y", "3y", "5y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1h"], index=0)

st.sidebar.header("Backtest sizing")
capital = st.sidebar.number_input("Capital (INR)", 10000, 10000000, 100000, step=10000)
pos_pct = st.sidebar.slider("Position %", 0.05, 1.0, 0.95, 0.05)
sl_pct = st.sidebar.slider("Stop-loss %", 0.01, 0.20, 0.05, 0.01)
trail = st.sidebar.checkbox("Trailing stop", value=True)


# ── Source input ─────────────────────────────────────────────────────────
tab_paste, tab_url = st.tabs(["Paste Pine", "From TradingView URL"])

with tab_paste:
    default_src = st.session_state.get("pine_src", load_sample())
    src = st.text_area("Pine Script source", value=default_src, height=320)

with tab_url:
    url = st.text_input("Public TradingView script URL")
    if st.button("Fetch from URL") and url:
        try:
            with st.spinner("Scraping (public scripts only)…"):
                fetched = fetch_pine_source(url)
            st.session_state["pine_src"] = fetched
            st.success("Fetched. Switch to 'Paste Pine' tab to view/edit.")
            st.code(fetched[:2000], language="pinescript")
        except Exception as e:
            st.error(f"Fetch failed: {e}")


# ── Parse + show warnings ─────────────────────────────────────────────────
if st.button("Parse & Analyze", type="primary"):
    if not src.strip():
        st.warning("Paste some Pine source first.")
        st.stop()
    parsed = parse(src)
    st.subheader("Analysis")
    st.text(summarize(parsed))
    if parsed.unsupported:
        st.error("Unsupported constructs found - backtest will NOT be faithful. "
                 "See list above. Fix or remove them before relying on results.")
    elif parsed.smells:
        st.warning("Heavier constructs present (e.g. security/var); fidelity "
                   "may be reduced. Treat results as indicative, not exact.")
    st.session_state["parsed"] = parsed
    st.session_state["pine_src"] = src


# ── Run backtest ─────────────────────────────────────────────────────────
if st.button("Run Backtest"):
    src = st.session_state.get("pine_src")
    if not src:
        st.warning("Parse & Analyze first (paste Pine source).")
        st.stop()
    try:
        with st.spinner("Transpiling…"):
            strategy, parsed = transpile(src)
    except TranspileError as e:
        st.error(f"Transpile error: {e}")
        st.stop()

    with st.spinner(f"Fetching {symbol} ({period}, {interval})…"):
        df = YFinanceDataFetcher().get_historical_data(
            symbol, period=period, interval=interval, use_cache=True)

    if df is None or df.empty:
        st.error("No data returned for that symbol/interval.")
        st.stop()

    with st.spinner("Backtesting…"):
        res = run_pine_backtest(
            strategy, df, initial_capital=capital,
            position_size_pct=pos_pct, stop_loss_pct=sl_pct,
            trailing_stop=trail, symbol=symbol)

    metrics = metrics_summary(res)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trades", metrics["total_trades"])
    col2.metric("Win rate", f"{metrics['win_rate']}%")
    col3.metric("Total return", f"{metrics['total_return_pct']}%")
    col4.metric("Max DD", f"{metrics['max_drawdown_pct']}%")

    col5, col6, col7 = st.columns(3)
    col5.metric("Sharpe", metrics["sharpe_ratio"])
    col6.metric("Profit factor", metrics["profit_factor"])
    col7.metric("CAGR", f"{metrics['cagr_pct']}%")

    eq = res.get("equity_curve")
    if eq is not None and not eq.empty:
        st.subheader("Equity Curve")
        st.line_chart(eq["Equity"])

    trades = res.get("trades", [])
    if trades:
        st.subheader("Trades")
        tdf = pd.DataFrame([{
            "entry": t.entry_date, "exit": t.exit_date,
            "dir": t.direction, "pnl": round(t.pnl, 1),
            "pnl%": round(t.pnl_pct * 100, 2),
            "hold": t.holding_days, "reason": t.exit_reason,
        } for t in trades])
        st.dataframe(tdf, use_container_width=True)
    else:
        st.info("No trades generated for this symbol/params.")

    if parsed.unsupported:
        st.error("Results may be INVALID - unsupported Pine constructs were "
                 "detected. The engine skipped what it couldn't faithfully model.")
