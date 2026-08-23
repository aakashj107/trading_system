"""
Visualizer v3.1 — Per-stock analysis, sector charts, watchlist, risk dashboards.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import os
from typing import Dict, List, Optional


class Visualizer:
    """Charts and dashboards for the trading system."""

    # Color palette
    COLORS = {
        "green": "#00e676",
        "red": "#ff1744",
        "yellow": "#ffd600",
        "blue": "#42a5f5",
        "purple": "#ab47bc",
        "orange": "#ff9100",
        "cyan": "#00e5ff",
        "white": "#f0ece4",
        "muted": "#7a7570",
        "bg": "#0d0d0d",
        "surface": "#1a1a1a",
        "grid": "#2a2a2a",
    }

    ACTION_COLORS = {
        "STRONG_BUY": "#00e676",
        "BUY": "#66bb6a",
        "HOLD": "#ffd600",
        "AVOID": "#ff9100",
        "STRONG_AVOID": "#ff1744",
    }

    def __init__(self, output_dir="charts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update({
            "figure.facecolor": self.COLORS["bg"],
            "axes.facecolor": self.COLORS["surface"],
            "axes.edgecolor": "#333",
            "axes.labelcolor": self.COLORS["white"],
            "text.color": self.COLORS["white"],
            "xtick.color": self.COLORS["muted"],
            "ytick.color": self.COLORS["muted"],
            "grid.color": self.COLORS["grid"],
            "grid.alpha": 0.5,
            "font.family": "monospace",
        })

    # ============================================================
    #  EXISTING CHARTS (kept for backward compatibility)
    # ============================================================

    def plot_equity_curve(self, equity_curve, title="Equity Curve",
                          filename="equity.png"):
        fig, ax = plt.subplots(figsize=(12, 6))
        eq = equity_curve["Equity"]
        ax.plot(eq.index, eq.values, color=self.COLORS["green"], linewidth=1.5)
        rm = eq.cummax()
        ax.fill_between(eq.index, eq.values, rm.values,
                        alpha=0.15, color=self.COLORS["red"])
        ax.axhline(y=eq.iloc[0], color=self.COLORS["yellow"],
                   linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Portfolio Value (₹)")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f"₹{x:,.0f}"))
        plt.tight_layout()
        fp = os.path.join(self.output_dir, filename)
        plt.savefig(fp, dpi=150)
        plt.close()
        print(f"  Chart saved: {fp}")
        return fp

    def plot_strategy_comparison(self, data, filename="comparison.png"):
        if not data:
            return
        df = pd.DataFrame(data)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        strategies = df["Strategy"]
        x = np.arange(len(strategies))
        configs = [
            (axes[0, 0], "Return%", self.COLORS["green"], "Return %"),
            (axes[0, 1], "Win_Rate%", self.COLORS["blue"], "Win Rate %"),
            (axes[1, 0], "Sharpe", self.COLORS["yellow"], "Sharpe Ratio"),
            (axes[1, 1], "Max_DD%", self.COLORS["red"], "Max Drawdown %"),
        ]
        for ax, col, color, title in configs:
            vals = df[col].astype(float)
            ax.barh(x, vals, color=color, alpha=0.8)
            ax.set_yticks(x)
            ax.set_yticklabels(strategies, fontsize=8)
            ax.set_title(title, fontweight="bold")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fp = os.path.join(self.output_dir, filename)
        plt.savefig(fp, dpi=150)
        plt.close()
        print(f"  Chart saved: {fp}")

    def plot_paper_dashboard(self, snapshots, trades, filename="paper_dash.png"):
        if not snapshots:
            return
        sdf = pd.DataFrame(snapshots)
        sdf["date"] = pd.to_datetime(sdf["date"])
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(sdf["date"], sdf["portfolio_value"],
                        color=self.COLORS["green"])
        axes[0, 0].set_title("Portfolio Value", fontweight="bold")
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f"₹{x:,.0f}"))
        axes[0, 1].plot(sdf["date"], sdf["return_pct"],
                        color=self.COLORS["yellow"])
        axes[0, 1].axhline(y=0, color="#fff", linewidth=0.5, linestyle="--")
        axes[0, 1].set_title("Return %", fontweight="bold")
        axes[1, 0].plot(sdf["date"], sdf["open_positions"],
                        color=self.COLORS["blue"])
        axes[1, 0].set_title("Open Positions", fontweight="bold")
        if trades:
            pnls = [t.pnl for t in trades]
            colors = [self.COLORS["green"] if p > 0
                      else self.COLORS["red"] for p in pnls]
            axes[1, 1].bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
            axes[1, 1].axhline(y=0, color="#fff", linewidth=0.5)
        axes[1, 1].set_title("Trade P&L", fontweight="bold")
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fp = os.path.join(self.output_dir, filename)
        plt.savefig(fp, dpi=150)
        plt.close()
        print(f"  Chart saved: {fp}")

    # ============================================================
    #  NEW: PER-STOCK ANALYSIS CHART
    # ============================================================

    def plot_stock_analysis(self, data: pd.DataFrame, symbol: str,
                            verdict: Optional[Dict] = None,
                            filename: Optional[str] = None) -> str:
        """
        Comprehensive per-stock chart with:
        - Price + SMA50 + SMA200 + Bollinger Bands
        - Volume bars (color-coded)
        - RSI indicator
        - MACD indicator
        - Verdict overlay (if provided)
        """
        if data.empty or len(data) < 50:
            print(f"  [WARNING] Insufficient data for {symbol}")
            return ""

        if filename is None:
            filename = f"{symbol.replace('.NS', '')}_analysis.png"

        # Calculate indicators
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]
        dates = data.index

        # Moving averages
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean() if len(close) >= 200 else None

        # Bollinger Bands
        bb_mid = sma20
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # RSI
        rsi = self._calculate_rsi(close, 14)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        # Create figure — 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                                 gridspec_kw={"height_ratios": [4, 1, 1, 1]},
                                 sharex=True)

        # ── Panel 1: Price + MAs + Bollinger ──
        ax_price = axes[0]

        # Candlestick-style coloring
        for i in range(1, len(data)):
            color = self.COLORS["green"] if close.iloc[i] >= close.iloc[i-1] \
                    else self.COLORS["red"]
            ax_price.plot([dates[i], dates[i]], [low.iloc[i], high.iloc[i]],
                         color=color, linewidth=0.5)
            ax_price.plot([dates[i], dates[i]],
                         [min(open_val := data["Open"].iloc[i], close.iloc[i]),
                          max(open_val, close.iloc[i])],
                         color=color, linewidth=2)

        # Moving averages
        ax_price.plot(dates, sma20, color=self.COLORS["yellow"],
                     linewidth=1, alpha=0.8, label="SMA20")
        ax_price.plot(dates, sma50, color=self.COLORS["blue"],
                     linewidth=1.2, alpha=0.8, label="SMA50")
        if sma200 is not None:
            ax_price.plot(dates, sma200, color=self.COLORS["purple"],
                         linewidth=1.5, alpha=0.8, label="SMA200")

        # Bollinger Bands (last 60 days)
        last_n = min(60, len(dates))
        ax_price.fill_between(dates[-last_n:], bb_upper.values[-last_n:],
                              bb_lower.values[-last_n:],
                              alpha=0.08, color=self.COLORS["cyan"])
        ax_price.plot(dates[-last_n:], bb_upper.values[-last_n:],
                     color=self.COLORS["cyan"], linewidth=0.5, alpha=0.5)
        ax_price.plot(dates[-last_n:], bb_lower.values[-last_n:],
                     color=self.COLORS["cyan"], linewidth=0.5, alpha=0.5)

                # ── PHASE 1B: Regression SuperTrend overlay (optional) ──
        # To enable, pass strategy_df in the verdict dict or as a parameter.
        # This checks if verdict contains regression data.
        if verdict and 'reg_strategy_df' in verdict:
            self.add_regression_supertrend_to_stock_chart(
                ax_price, data, verdict['reg_strategy_df']
            )

        # Price line
        ax_price.plot(dates, close, color=self.COLORS["white"],
                     linewidth=1, alpha=0.3)

        ax_price.set_title(f"{symbol} — Technical Analysis",
                          fontsize=14, fontweight="bold", pad=15)
        ax_price.set_ylabel("Price (₹)")
        ax_price.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f"₹{x:,.0f}"))
        ax_price.legend(loc="upper left", fontsize=8, framealpha=0.3)
        ax_price.grid(True, alpha=0.3)

        # Current price annotation
        last_price = close.iloc[-1]
        ax_price.annotate(
            f"  ₹{last_price:,.2f}",
            xy=(dates[-1], last_price),
            fontsize=10, fontweight="bold",
            color=self.COLORS["green"] if last_price >= sma50.iloc[-1]
            else self.COLORS["red"],
            va="center"
        )

        # ── Panel 2: Volume ──
        ax_vol = axes[1]
        vol_colors = []
        for i in range(len(close)):
            if i == 0:
                vol_colors.append(self.COLORS["green"])
            else:
                vol_colors.append(
                    self.COLORS["green"] if close.iloc[i] >= close.iloc[i-1]
                    else self.COLORS["red"]
                )
        ax_vol.bar(dates, volume.values, color=vol_colors, alpha=0.6, width=0.8)

        # Volume SMA
        vol_sma = volume.rolling(20).mean()
        ax_vol.plot(dates, vol_sma, color=self.COLORS["yellow"],
                   linewidth=1, alpha=0.7, label="Vol SMA20")

        ax_vol.set_ylabel("Volume")
        ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
        ax_vol.legend(loc="upper left", fontsize=7, framealpha=0.3)
        ax_vol.grid(True, alpha=0.3)

        # ── Panel 3: RSI ──
        ax_rsi = axes[2]
        ax_rsi.plot(dates, rsi, color=self.COLORS["purple"],
                   linewidth=1.2, label="RSI(14)")
        ax_rsi.axhline(y=70, color=self.COLORS["red"],
                       linestyle="--", linewidth=0.8, alpha=0.7)
        ax_rsi.axhline(y=30, color=self.COLORS["green"],
                       linestyle="--", linewidth=0.8, alpha=0.7)
        ax_rsi.axhline(y=50, color=self.COLORS["muted"],
                       linestyle="--", linewidth=0.5, alpha=0.5)
        ax_rsi.fill_between(dates, rsi, 70,
                           where=(rsi > 70), alpha=0.2,
                           color=self.COLORS["red"])
        ax_rsi.fill_between(dates, rsi, 30,
                           where=(rsi < 30), alpha=0.2,
                           color=self.COLORS["green"])
        ax_rsi.set_ylabel("RSI")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend(loc="upper left", fontsize=7, framealpha=0.3)
        ax_rsi.grid(True, alpha=0.3)

        # RSI value annotation
        last_rsi = rsi.iloc[-1]
        ax_rsi.annotate(
            f"  {last_rsi:.1f}",
            xy=(dates[-1], last_rsi),
            fontsize=9, fontweight="bold",
            color=self.COLORS["red"] if last_rsi > 70
            else self.COLORS["green"] if last_rsi < 30
            else self.COLORS["white"]
        )

        # ── Panel 4: MACD ──
        ax_macd = axes[3]

        # Histogram
        hist_colors = [self.COLORS["green"] if v >= 0
                       else self.COLORS["red"] for v in macd_hist]
        ax_macd.bar(dates, macd_hist, color=hist_colors, alpha=0.5, width=0.8)

        # MACD and Signal lines
        ax_macd.plot(dates, macd, color=self.COLORS["blue"],
                    linewidth=1, label="MACD")
        ax_macd.plot(dates, macd_signal, color=self.COLORS["orange"],
                    linewidth=1, label="Signal")
        ax_macd.axhline(y=0, color=self.COLORS["muted"],
                        linewidth=0.5, linestyle="--")

        ax_macd.set_ylabel("MACD")
        ax_macd.set_xlabel("Date")
        ax_macd.legend(loc="upper left", fontsize=7, framealpha=0.3)
        ax_macd.grid(True, alpha=0.3)

        # ── Verdict overlay (if provided) ──
        if verdict:
            self._add_verdict_overlay(fig, verdict)

        plt.tight_layout()
        fp = os.path.join(self.output_dir, filename)
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 {symbol} chart saved: {fp}")
        return fp

    def _add_verdict_overlay(self, fig, verdict: Dict):
        """Add verdict box overlay to stock chart."""
        action = verdict.get("action", "HOLD")
        conviction = verdict.get("conviction_score", 0)
        action_color = self.ACTION_COLORS.get(action, self.COLORS["yellow"])

        box_text = (
            f"VERDICT: {action}\n"
            f"Conviction: {conviction}/10\n"
            f"Fund: {verdict.get('fundamental_score', '-')}/10  "
            f"Val: {verdict.get('valuation_score', '-')}/10\n"
            f"Tech: {verdict.get('technical_score', '-')}/10  "
            f"Risk: {verdict.get('risk_score', '-')}/10"
        )

        fig.text(
            0.98, 0.98, box_text,
            transform=fig.transFigure,
            fontsize=10, fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=self.COLORS["surface"],
                edgecolor=action_color,
                linewidth=2,
                alpha=0.9,
            ),
            color=action_color,
        )

    # ============================================================
    #  NEW: SECTOR DISTRIBUTION CHART
    # ============================================================

    def plot_sector_distribution(self, sector_data: Dict[str, int],
                                  filename="sector_distribution.png"):
        """
        Sector distribution pie + bar chart.

        Args:
            sector_data: {"Financial Services": 8, "Technology": 5, ...}
        """
        if not sector_data:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        sectors = list(sector_data.keys())
        counts = list(sector_data.values())
        total = sum(counts)

        # Color cycle
        colors = [
            "#00e676", "#42a5f5", "#ffd600", "#ff9100", "#ab47bc",
            "#00e5ff", "#ff1744", "#8d6e63", "#78909c", "#ec407a",
            "#26a69a", "#7e57c2",
        ][:len(sectors)]

        # Pie chart
        ax_pie = axes[0]
        wedges, texts, autotexts = ax_pie.pie(
            counts, labels=None, autopct="%1.0f%%",
            colors=colors, startangle=90, pctdistance=0.85,
            textprops={"fontsize": 9, "color": self.COLORS["white"]}
        )
        ax_pie.legend(
            wedges, [f"{s} ({c})" for s, c in zip(sectors, counts)],
            loc="center left", bbox_to_anchor=(-0.1, 0.5),
            fontsize=8, framealpha=0.3
        )
        ax_pie.set_title(f"Sector Distribution ({total} stocks)",
                        fontweight="bold", fontsize=12)

        # Bar chart
        ax_bar = axes[1]
        y = np.arange(len(sectors))
        bars = ax_bar.barh(y, counts, color=colors, alpha=0.8)
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(sectors, fontsize=9)
        ax_bar.set_xlabel("Number of Stocks")
        ax_bar.set_title("Stocks per Sector", fontweight="bold", fontsize=12)
        ax_bar.grid(True, alpha=0.3, axis="x")

        # Count labels on bars
        for bar, count in zip(bars, counts):
            ax_bar.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                       str(count), va="center", fontsize=9,
                       color=self.COLORS["white"])

        plt.tight_layout()
        fp = os.path.join(self.output_dir, filename)
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Sector chart saved: {fp}")
        return fp

    # ============================================================
    #  NEW: STOCK RANKINGS CHART
    # ============================================================

    def plot_stock_rankings(self, verdicts: List[Dict],
                             filename="stock_rankings.png"):
        """
        Visual stock rankings with score breakdown.

        Args:
            verdicts: List of verdict dicts from VerdictEngine
        """
        if not verdicts:
            return ""

        ranked = sorted(verdicts, key=lambda x: x["conviction_score"],
                       reverse=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(ranked) * 0.45)))

        # ── Left: Conviction scores with action colors ──
        ax_conv = axes[0]
        symbols = [v["symbol"].replace(".NS", "") for v in ranked]
        scores = [v["conviction_score"] for v in ranked]
        actions = [v["action"] for v in ranked]
        bar_colors = [self.ACTION_COLORS.get(a, self.COLORS["yellow"])
                      for a in actions]

        y = np.arange(len(symbols))
        bars = ax_conv.barh(y, scores, color=bar_colors, alpha=0.8, height=0.7)
        ax_conv.set_yticks(y)
        ax_conv.set_yticklabels(symbols, fontsize=9)
        ax_conv.set_xlabel("Conviction Score")
        ax_conv.set_xlim(0, 10)
        ax_conv.set_title("Stock Rankings by Conviction",
                         fontweight="bold", fontsize=12)
        ax_conv.grid(True, alpha=0.3, axis="x")

        # Score labels
        for bar, score, action in zip(bars, scores, actions):
            ax_conv.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height()/2,
                f"{score:.1f} ({action})",
                va="center", fontsize=8,
                color=self.ACTION_COLORS.get(action, self.COLORS["white"])
            )

        # ── Right: Score breakdown (stacked) ──
        ax_stack = axes[1]
        categories = ["fundamental_score", "management_score",
                      "valuation_score", "technical_score", "risk_score"]
        cat_labels = ["Fund", "Mgmt", "Val", "Tech", "Risk"]
        cat_colors = [self.COLORS["green"], self.COLORS["blue"],
                      self.COLORS["yellow"], self.COLORS["purple"],
                      self.COLORS["red"]]

        # Build stacked data
        bottom = np.zeros(len(ranked))
        for cat, label, color in zip(categories, cat_labels, cat_colors):
            values = [v.get(cat, 0) * 0.2 for v in ranked]  # Weighted
            ax_stack.barh(y, values, left=bottom, color=color,
                         alpha=0.8, height=0.7, label=label)
            bottom += values

        ax_stack.set_yticks(y)
        ax_stack.set_yticklabels(symbols, fontsize=9)
        ax_stack.set_xlabel("Weighted Score Breakdown")
        ax_stack.set_title("Score Breakdown (Weighted)",
                          fontweight="bold", fontsize=12)
        ax_stack.legend(loc="lower right", fontsize=8, framealpha=0.3,
                       ncol=5)
        ax_stack.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        fp = os.path.join(self.output_dir, filename)
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Rankings chart saved: {fp}")
        return fp

    # ============================================================
    #  NEW: WATCHLIST CHART
    # ============================================================

    def plot_watchlist(self, watchlist: List[Dict],
                       filename="watchlist.png"):
        """
        Visual watchlist showing gap to buy zone.

        Args:
            watchlist: List of watchlist dicts with price, buy_zone, gap_pct
        """
        if not watchlist:
            return ""

        fig, ax = plt.subplots(figsize=(14, max(6, len(watchlist) * 0.45)))

        symbols = [w["symbol"].replace(".NS", "") for w in watchlist]
        gaps = [w["gap_pct"] for w in watchlist]
        prices = [w["price"] for w in watchlist]
        buy_zones = [w["buy_zone"] for w in watchlist]

        y = np.arange(len(symbols))

        # Color based on gap
        bar_colors = []
        for gap in gaps:
            if gap <= 0:
                bar_colors.append(self.COLORS["green"])
            elif gap <= 10:
                bar_colors.append(self.COLORS["yellow"])
            elif gap <= 20:
                bar_colors.append(self.COLORS["orange"])
            else:
                bar_colors.append(self.COLORS["red"])

        bars = ax.barh(y, gaps, color=bar_colors, alpha=0.7, height=0.6)

        # Zero line (buy zone)
        ax.axvline(x=0, color=self.COLORS["green"],
                   linewidth=2, linestyle="--", alpha=0.7, label="Buy Zone")

        # 15% approach line
        ax.axvline(x=15, color=self.COLORS["yellow"],
                   linewidth=1, linestyle="--", alpha=0.5, label="15% Away")

        ax.set_yticks(y)
        ax.set_yticklabels(symbols, fontsize=9)
        ax.set_xlabel("Gap to Buy Zone (%)")
        ax.set_title("Watchlist — Distance to Buy Zone",
                    fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.3)

        # Labels on bars
        for bar, gap, price, bz in zip(bars, gaps, prices, buy_zones):
            label = f"{gap:+.1f}%  (₹{price:,.0f} → ₹{bz:,.0f})"
            x_pos = bar.get_width() + 0.5 if gap >= 0 else bar.get_width() - 0.5
            ha = "left" if gap >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   label, va="center", ha=ha, fontsize=7,
                   color=self.COLORS["white"])

        plt.tight_layout()
        fp = os.path.join(self.output_dir, filename)
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Watchlist chart saved: {fp}")
        return fp

    # ============================================================
    #  NEW: CONSENSUS HEATMAP
    # ============================================================

    def plot_consensus_heatmap(self, signals_df: pd.DataFrame,
                                filename="consensus_heatmap.png"):
        """
        Heatmap showing which strategies fired signals for which stocks.

        Args:
            signals_df: DataFrame from SignalScanner.scan()
        """
        if signals_df is None or signals_df.empty:
            print("  No signals for heatmap")
            return ""

        # Pivot: strategies as rows, stocks as columns
        pivot = signals_df.pivot_table(
            index="Strategy", columns="Symbol",
            values="Signal", aggfunc="first"
        )

        # Convert to numeric for coloring
        signal_map = {"BUY": 1, "SELL": -1, "SHORT": -2}
        numeric = pivot.replace(signal_map).fillna(0).astype(float)

        fig, ax = plt.subplots(
            figsize=(max(10, len(pivot.columns) * 1.2),
                     max(6, len(pivot.index) * 0.4))
        )

        # Custom colormap: green for BUY, red for SELL/SHORT
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "signals",
            [self.COLORS["red"], self.COLORS["surface"],
             self.COLORS["green"]],
            N=3
        )

        im = ax.imshow(numeric.values, cmap=cmap, aspect="auto",
                       vmin=-2, vmax=1)

        # Labels
        symbols = [s.replace(".NS", "") for s in pivot.columns]
        ax.set_xticks(range(len(symbols)))
        ax.set_xticklabels(symbols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)

        # Text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    color = self.COLORS["white"]
                    ax.text(j, i, str(val), ha="center", va="center",
                           fontsize=7, color=color, fontweight="bold")

        ax.set_title("Signal Consensus Heatmap",
                    fontweight="bold", fontsize=12)

        # Legend
        legend_elements = [
            mpatches.Patch(color=self.COLORS["green"], label="BUY"),
            mpatches.Patch(color=self.COLORS["surface"], label="HOLD"),
            mpatches.Patch(color=self.COLORS["red"], label="SELL/SHORT"),
        ]
        ax.legend(handles=legend_elements, loc="upper right",
                 fontsize=8, framealpha=0.3)

        plt.tight_layout()
        fp = os.path.join(self.output_dir, filename)
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Consensus heatmap saved: {fp}")
        return fp

    # ============================================================
    #  NEW: MARKET BREADTH CHART
    # ============================================================

    def plot_market_breadth(self, breadth_data: Dict,
                             filename="market_breadth.png"):
        """
        Visual market breadth indicators.

        Args:
            breadth_data: Dict from DailyRunner._calculate_breadth()
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ── Left: Breadth gauge ──
        ax = axes[0]
        pct = breadth_data.get("pct_above_20sma", 0)

        # Gauge segments
        segments = [
            (0, 25, self.COLORS["red"], "VERY WEAK"),
            (25, 40, self.COLORS["orange"], "WEAK"),
            (40, 60, self.COLORS["yellow"], "MODERATE"),
            (60, 75, self.COLORS["green"], "STRONG"),
            (75, 100, self.COLORS["cyan"], "VERY STRONG"),
        ]

        for low, high, color, label in segments:
            ax.barh(0, high - low, left=low, height=0.5,
                   color=color, alpha=0.3)
            ax.text((low + high) / 2, -0.35, label,
                   ha="center", fontsize=7, color=color)

        # Current value marker
        ax.plot(pct, 0, marker="v", markersize=20,
               color=self.COLORS["white"], zorder=5)
        ax.text(pct, 0.35, f"{pct:.1f}%", ha="center",
               fontsize=14, fontweight="bold", color=self.COLORS["white"])

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.7, 0.7)
        ax.set_title("Market Breadth (% Above SMA20)",
                    fontweight="bold", fontsize=12)
        ax.set_xticks([0, 25, 40, 60, 75, 100])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis="x")

        # ── Right: Stats ──
        ax2 = axes[1]
        ax2.axis("off")

        stats = [
            ("Stocks Analyzed", str(breadth_data.get("stocks_analyzed", 0))),
            ("Above SMA20",
             f"{breadth_data.get('above_20sma', 0)} / "
             f"{breadth_data.get('stocks_analyzed', 0)} "
             f"({breadth_data.get('pct_above_20sma', 0):.1f}%)"),
            ("Above SMA50",
             f"{breadth_data.get('above_50sma', 0)} / "
             f"{breadth_data.get('stocks_analyzed', 0)} "
             f"({breadth_data.get('pct_above_50sma', 0):.1f}%)"),
            ("Breadth Regime", breadth_data.get("breadth_regime", "N/A")),
        ]

        for i, (label, value) in enumerate(stats):
            y_pos = 0.85 - i * 0.2
            ax2.text(0.1, y_pos, label, fontsize=11,
                    color=self.COLORS["muted"], transform=ax2.transAxes)
            ax2.text(0.6, y_pos, value, fontsize=12, fontweight="bold",
                    color=self.COLORS["white"], transform=ax2.transAxes)

        plt.tight_layout()
        fp = os.path.join(self.output_dir, filename)
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Breadth chart saved: {fp}")
        return fp

    # ============================================================
    #  NEW: RISK DASHBOARD CHART
    # ============================================================

    def plot_risk_dashboard(self, risk_data: Dict,
                             filename="risk_dashboard.png"):
        """
        Visual risk dashboard from RiskManager.get_dashboard().

        Args:
            risk_data: Dict from risk_manager.get_dashboard()
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ── Top Left: Risk Score Gauge ──
        ax = axes[0, 0]
        score = risk_data.get("risk_score", 0)
        level = risk_data.get("risk_level", "LOW")

        level_colors = {
            "LOW": self.COLORS["green"],
            "MODERATE": self.COLORS["yellow"],
            "HIGH": self.COLORS["orange"],
            "CRITICAL": self.COLORS["red"],
        }
        color = level_colors.get(level, self.COLORS["white"])

        # Draw gauge
        theta = np.linspace(np.pi, 0, 100)
        for i, t in enumerate(theta):
            c = self.COLORS["green"] if i < 25 else \
                self.COLORS["yellow"] if i < 50 else \
                self.COLORS["orange"] if i < 75 else \
                self.COLORS["red"]
            ax.plot([np.cos(t) * 0.8, np.cos(t)],
                   [np.sin(t) * 0.8, np.sin(t)],
                   color=c, linewidth=3, alpha=0.3)

        # Needle
        needle_angle = np.pi - (score / 100) * np.pi
        ax.plot([0, np.cos(needle_angle) * 0.9],
               [0, np.sin(needle_angle) * 0.9],
               color=color, linewidth=3)
        ax.plot(np.cos(needle_angle) * 0.9,
               np.sin(needle_angle) * 0.9,
               marker="o", markersize=10, color=color)

        ax.text(0, -0.2, f"{score:.0f}", ha="center", va="center",
               fontsize=28, fontweight="bold", color=color)
        ax.text(0, -0.45, level, ha="center", va="center",
               fontsize=14, color=color)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Portfolio Risk Score", fontweight="bold")

        # ── Top Right: VIX Info ──
        ax2 = axes[0, 1]
        ax2.axis("off")
        vix = risk_data.get("vix", "N/A")
        vix_regime = risk_data.get("vix_regime", "N/A")
        pos_pct = risk_data.get("vix_position_pct", 0)
        max_pos = risk_data.get("vix_max_positions", 0)

        vix_info = [
            ("India VIX", str(vix) if vix else "N/A"),
            ("Regime", vix_regime),
            ("Position Size", f"{pos_pct*100:.0f}%"),
            ("Max Positions", str(max_pos)),
            ("Open Positions", str(risk_data.get("open_positions", 0))),
        ]
        for i, (label, value) in enumerate(vix_info):
            y_pos = 0.85 - i * 0.17
            ax2.text(0.1, y_pos, label, fontsize=11,
                    color=self.COLORS["muted"], transform=ax2.transAxes)
            ax2.text(0.65, y_pos, value, fontsize=12, fontweight="bold",
                    color=self.COLORS["white"], transform=ax2.transAxes)
        ax2.set_title("VIX Risk Gate", fontweight="bold")

        # ── Bottom Left: Sector Exposure ──
        ax3 = axes[1, 0]
        sector_data = risk_data.get("sector_exposure", {})
        if sector_data:
            sectors = list(sector_data.keys())
            pcts = [d.get("pct", 0) for d in sector_data.values()]
            counts = [d.get("count", 0) for d in sector_data.values()]
            colors = plt.cm.Set3(np.linspace(0, 1, len(sectors)))

            bars = ax3.barh(range(len(sectors)), pcts, color=colors, alpha=0.8)
            ax3.set_yticks(range(len(sectors)))
            ax3.set_yticklabels(sectors, fontsize=8)
            ax3.set_xlabel("Exposure %")
            ax3.set_title("Sector Exposure", fontweight="bold")
            ax3.grid(True, alpha=0.3, axis="x")

            for bar, pct, count in zip(bars, pcts, counts):
                ax3.text(bar.get_width() + 0.5,
                        bar.get_y() + bar.get_height()/2,
                        f"{pct:.1f}% ({count})", va="center",
                        fontsize=8, color=self.COLORS["white"])
        else:
            ax3.text(0.5, 0.5, "No Positions", ha="center", va="center",
                    fontsize=14, color=self.COLORS["muted"],
                    transform=ax3.transAxes)
            ax3.set_title("Sector Exposure", fontweight="bold")

        # ── Bottom Right: Risk Factors ──
        ax4 = axes[1, 1]
        ax4.axis("off")
        factors = risk_data.get("risk_factors", [])
        if factors:
            ax4.text(0.05, 0.95, "RISK FACTORS:", fontsize=11,
                    fontweight="bold", color=self.COLORS["yellow"],
                    transform=ax4.transAxes, va="top")
            for i, factor in enumerate(factors):
                y_pos = 0.80 - i * 0.15
                ax4.text(0.05, y_pos, f"⚠ {factor}", fontsize=10,
                        color=self.COLORS["orange"],
                        transform=ax4.transAxes, va="top")
        else:
            ax4.text(0.5, 0.5, "✅ No Active Risk Factors",
                    ha="center", va="center", fontsize=14,
                    color=self.COLORS["green"],
                    transform=ax4.transAxes)
        ax4.set_title("Risk Factors", fontweight="bold")

        plt.tight_layout()
        fp = os.path.join(self.output_dir, filename)
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Risk dashboard saved: {fp}")
        return fp

    # ============================================================
    #  NEW: MULTI-STOCK OVERVIEW (batch chart)
    # ============================================================

    def plot_multi_stock_overview(self, verdicts: List[Dict],
                                    filename="multi_stock_overview.png"):
        """
        Grid overview of all stocks with key metrics.

        Args:
            verdicts: List of verdict dicts
        """
        if not verdicts:
            return ""

        ranked = sorted(verdicts, key=lambda x: x["conviction_score"],
                       reverse=True)

        n = len(ranked)
        cols = 5
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx, v in enumerate(ranked):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            symbol = v["symbol"].replace(".NS", "")
            action = v["action"]
            score = v["conviction_score"]
            price = v["current_price"]
            action_color = self.ACTION_COLORS.get(action, self.COLORS["yellow"])

            # Background color based on action
            ax.set_facecolor(self.COLORS["surface"])

            # Action badge
            ax.text(0.5, 0.85, symbol, ha="center", va="center",
                   fontsize=13, fontweight="bold",
                   color=self.COLORS["white"],
                   transform=ax.transAxes)
            ax.text(0.5, 0.60, action.replace("_", " "), ha="center",
                   va="center", fontsize=10, fontweight="bold",
                   color=action_color, transform=ax.transAxes)
            ax.text(0.5, 0.38, f"Conv: {score}/10", ha="center",
                   va="center", fontsize=9,
                   color=self.COLORS["white"],
                   transform=ax.transAxes)
            ax.text(0.5, 0.18, f"₹{price:,.0f}", ha="center",
                   va="center", fontsize=10,
                   color=self.COLORS["muted"],
                   transform=ax.transAxes)

            # Score bars (mini)
            scores = [
                v.get("fundamental_score", 0) / 10,
                v.get("valuation_score", 0) / 10,
                v.get("technical_score", 0) / 10,
            ]
            bar_colors = [self.COLORS["green"], self.COLORS["yellow"],
                         self.COLORS["purple"]]
            bar_y = 0.05
            bar_width = 0.15
            for i, (s, c) in enumerate(zip(scores, bar_colors)):
                x_start = 0.15 + i * 0.25
                ax.barh(bar_y, s * bar_width, left=x_start,
                       height=0.06, color=c, alpha=0.7,
                       transform=ax.transAxes)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            # Border color based on action
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(action_color)
                spine.set_linewidth(2)

        # Hide empty subplots
        for idx in range(n, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis("off")

        fig.suptitle("Stock Universe Overview", fontsize=16,
                    fontweight="bold", y=1.01)
        plt.tight_layout()
        fp = os.path.join(self.output_dir, filename)
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Multi-stock overview saved: {fp}")
        return fp

    # ============================================================
    #  NEW: GENERATE ALL CHARTS (batch runner)
    # ============================================================

    def generate_all_charts(self, verdicts: List[Dict] = None,
                            signals_df: pd.DataFrame = None,
                            watchlist: List[Dict] = None,
                            breadth_data: Dict = None,
                            risk_data: Dict = None,
                            sector_data: Dict[str, int] = None,
                            stock_data_cache: Dict[str, pd.DataFrame] = None):
        """
        Generate all available charts in one call.

        Args:
            verdicts: List of verdict dicts
            signals_df: DataFrame from scanner
            watchlist: List of watchlist dicts
            breadth_data: Dict from market breadth calculation
            risk_data: Dict from risk manager dashboard
            sector_data: Dict of sector counts
            stock_data_cache: Dict of symbol → DataFrame
        """
        generated = []
        print(f"\n  Generating charts...")

        # 1. Sector distribution
        if sector_data:
            fp = self.plot_sector_distribution(sector_data)
            if fp:
                generated.append(fp)

        # 2. Stock rankings
        if verdicts:
            fp = self.plot_stock_rankings(verdicts)
            if fp:
                generated.append(fp)

            # 3. Multi-stock overview
            fp = self.plot_multi_stock_overview(verdicts)
            if fp:
                generated.append(fp)

        # 4. Watchlist
        if watchlist:
            fp = self.plot_watchlist(watchlist)
            if fp:
                generated.append(fp)

        # 5. Consensus heatmap
        if signals_df is not None and not signals_df.empty:
            fp = self.plot_consensus_heatmap(signals_df)
            if fp:
                generated.append(fp)

        # 6. Market breadth
        if breadth_data:
            fp = self.plot_market_breadth(breadth_data)
            if fp:
                generated.append(fp)

        # 7. Risk dashboard
        if risk_data:
            fp = self.plot_risk_dashboard(risk_data)
            if fp:
                generated.append(fp)

        # 8. Per-stock analysis (top 5 by conviction)
        if verdicts and stock_data_cache:
            ranked = sorted(verdicts, key=lambda x: x["conviction_score"],
                          reverse=True)[:5]
            for v in ranked:
                symbol = v["symbol"]
                data = stock_data_cache.get(symbol)
                if data is not None and not data.empty:
                    fp = self.plot_stock_analysis(data, symbol, verdict=v)
                    if fp:
                        generated.append(fp)

        print(f"\n  Total charts generated: {len(generated)}")
        print(f"  Output directory: {self.output_dir}/")
        return generated

    # ============================================================
    #  HELPER: RSI Calculation
    # ============================================================

    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

        # ============================================================
    #  PHASE 1B: REGRESSION SUPERTREND OVERLAY
    #  Add after _calculate_rsi method
    # ============================================================

    def add_regression_supertrend_to_stock_chart(
            self, ax, data: pd.DataFrame, strategy_df: pd.DataFrame):
        """
        Overlay Regression SuperTrend indicators onto a stock price chart.

        Call this from plot_stock_analysis() after drawing price/candles.

        Args:
            ax:          matplotlib axes (the price panel)
            data:        Original OHLCV DataFrame
            strategy_df: DataFrame from RegressionSuperTrendStrategy
                         .generate_signals() — must contain Reg_Line,
                         Reg_Upper, Reg_Lower, Reg_Trend columns
        """
        if 'Reg_Line' not in strategy_df.columns:
            return  # No regression data

        dates = data.index
        reg_line = strategy_df['Reg_Line'].values
        reg_upper = strategy_df['Reg_Upper'].values
        reg_lower = strategy_df['Reg_Lower'].values
        reg_trend = strategy_df['Reg_Trend'].values
        signals = strategy_df['Signal'].values

        valid = ~np.isnan(reg_line)
        if valid.sum() < 2:
            return

        # Regression line — dashed, subtle
        ax.plot(dates[valid], reg_line[valid],
                color='#888888', linewidth=1, linestyle='--',
                alpha=0.6, label='Reg Line')

        # Plot trend-colored band line
        # In uptrend: plot lower band (support). In downtrend: plot upper band (resistance)
        for i in range(1, len(reg_trend)):
            if not valid[i]:
                continue
            color = '#26a69a' if reg_trend[i] == 1 else '#ef5350'
            band_val = reg_lower[i] if reg_trend[i] == 1 else reg_upper[i]
            ax.plot(dates[i], band_val,
                    color=color, marker='.', markersize=1.5, alpha=0.7)

        # Mark BUY signals
        buy_mask = (signals == Signal.BUY.value)
        if buy_mask.any():
            buy_prices = data['Low'][buy_mask] * 0.995
            ax.scatter(dates[buy_mask], buy_prices,
                       marker='^', color='#26a69a', s=100, zorder=5,
                       label='RegST BUY', edgecolors='white', linewidth=0.5)

        # Mark SELL signals
        sell_mask = (signals == Signal.SELL.value)
        if sell_mask.any():
            sell_prices = data['High'][sell_mask] * 1.005
            ax.scatter(dates[sell_mask], sell_prices,
                       marker='v', color='#ef5350', s=100, zorder=5,
                       label='RegST SELL', edgecolors='white', linewidth=0.5)

        # ============================================================
    #  PHASE 2: STRUCTURAL LEVELS OVERLAY
    # ============================================================

    def add_structural_levels_to_stock_chart(
            self, ax, data: pd.DataFrame, strategy_df: pd.DataFrame,
            label_prefix: str = ""):
        """
        Overlay swing highs/lows and signal markers from structural
        strategies (CHOCH/BOS, Liquidity Sweep) onto a stock chart.

        Args:
            ax:           matplotlib axes (price panel)
            data:         Original OHLCV DataFrame
            strategy_df:  DataFrame from CHOCHBOSRetest or
                          LiquiditySweepRetest .generate_signals()
            label_prefix: Optional prefix for legend labels
        """
        if 'Swing_High' not in strategy_df.columns:
            return

        dates = data.index
        swing_h = strategy_df['Swing_High'].values
        swing_l = strategy_df['Swing_Low'].values
        signals = strategy_df['Signal'].values

        # Draw swing high markers (subtle triangles at top)
        sh_mask = ~np.isnan(swing_h)
        if sh_mask.any():
            ax.scatter(dates[sh_mask], swing_h[sh_mask] * 1.005,
                       marker='v', color='#78909c', s=30, alpha=0.5,
                       zorder=3, label=f'{label_prefix}Swing H')

        # Draw swing low markers (subtle triangles at bottom)
        sl_mask = ~np.isnan(swing_l)
        if sl_mask.any():
            ax.scatter(dates[sl_mask], swing_l[sl_mask] * 0.995,
                       marker='^', color='#78909c', s=30, alpha=0.5,
                       zorder=3, label=f'{label_prefix}Swing L')

        # Mark BUY signals
        buy_mask = (signals == Signal.BUY.value)
        if buy_mask.any():
            ax.scatter(dates[buy_mask], data['Low'][buy_mask] * 0.993,
                       marker='^', color='#26a69a', s=120, zorder=5,
                       label=f'{label_prefix}BUY', edgecolors='white',
                       linewidth=0.8)

        # Mark SELL signals
        sell_mask = (signals == Signal.SELL.value)
        if sell_mask.any():
            ax.scatter(dates[sell_mask], data['High'][sell_mask] * 1.007,
                       marker='v', color='#ef5350', s=120, zorder=5,
                       label=f'{label_prefix}SELL', edgecolors='white',
                       linewidth=0.8)

        # ============================================================
    #  PHASE 3: EXHAUSTION + SUB-REGIME OVERLAY
    # ============================================================

    def add_exhaustion_to_stock_chart(
            self, ax, data: pd.DataFrame,
            exhaustion_df: pd.DataFrame):
        """
        Overlay exhaustion markers and sub-regime shading on stock chart.

        Args:
            ax:             matplotlib axes (price panel)
            data:           Original OHLCV DataFrame
            exhaustion_df:  DataFrame from ExhaustionDetector.analyze()
        """
        if 'Exhausting' not in exhaustion_df.columns:
            return

        dates = data.index
        exh_mask = exhaustion_df['Exhausting'].fillna(False).astype(bool)

        if exh_mask.any():
            # Highlight exhaustion bars with subtle background
            for idx in dates[exh_mask]:
                ax.axvspan(
                    idx - pd.Timedelta(hours=12),
                    idx + pd.Timedelta(hours=12),
                    alpha=0.08, color='#ff9100', zorder=0
                )

            # Add exhaustion markers above price
            exh_highs = data['High'][exh_mask] * 1.01
            ax.scatter(
                dates[exh_mask], exh_highs,
                marker='o', color='#ff9100', s=25, alpha=0.7,
                zorder=4, label='Exhaustion'
            )

    def add_sub_regime_to_stock_chart(
            self, ax, data: pd.DataFrame,
            classifier_df: pd.DataFrame):
        """
        Add subtle TREND/RANGE background shading to stock chart.

        Args:
            ax:              matplotlib axes (price panel)
            data:            Original OHLCV DataFrame
            classifier_df:   DataFrame from TrendRangeClassifier.classify()
        """
        if 'Sub_Regime' not in classifier_df.columns:
            return

        dates = data.index
        regime = classifier_df['Sub_Regime'].values

        # Find contiguous TREND/RANGE segments
        if len(regime) < 2:
            return

        # Shade RANGE segments subtly
        in_range = False
        range_start = None

        for i in range(len(regime)):
            if regime[i] == 'RANGE' and not in_range:
                in_range = True
                range_start = dates[i]
            elif regime[i] != 'RANGE' and in_range:
                in_range = False
                ax.axvspan(
                    range_start, dates[i],
                    alpha=0.04, color='#ff9100', zorder=0
                )

        # Handle case where data ends in RANGE
        if in_range and range_start is not None:
            ax.axvspan(
                range_start, dates[-1],
                alpha=0.04, color='#ff9100', zorder=0
            )

        # ============================================================
    #  PHASE 4B: ADX QUALITY OVERLAY
    # ============================================================

    def add_adx_to_stock_chart(self, ax_indicator, data: pd.DataFrame,
                                adx_period: int = 14):
        """
        Add ADX line chart to a stock analysis chart.

        Call this as a new subplot in plot_stock_analysis() to show
        ADX alongside RSI and MACD.

        Args:
            ax_indicator: matplotlib axes (new subplot for ADX)
            data:         OHLCV DataFrame
            adx_period:   ADX lookback period
        """
        # Compute ADX
        high = data["High"]
        low = data["Low"]
        close = data["Close"]
        dates = data.index

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        alpha = 1.0 / adx_period
        atr = tr.ewm(alpha=alpha, min_periods=adx_period, adjust=False).mean()
        sm_plus = plus_dm.ewm(alpha=alpha, min_periods=adx_period, adjust=False).mean()
        sm_minus = minus_dm.ewm(alpha=alpha, min_periods=adx_period, adjust=False).mean()

        plus_di = 100 * sm_plus / atr.replace(0, np.nan)
        minus_di = 100 * sm_minus / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (
            plus_di + minus_di
        ).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, min_periods=adx_period, adjust=False).mean()

        # Plot ADX
        ax_indicator.plot(dates, adx, color=self.COLORS["yellow"],
                         linewidth=1.2, label="ADX")

        # +DI and -DI
        ax_indicator.plot(dates, plus_di, color=self.COLORS["green"],
                         linewidth=0.8, alpha=0.6, label="+DI")
        ax_indicator.plot(dates, minus_di, color=self.COLORS["red"],
                         linewidth=0.8, alpha=0.6, label="-DI")

        # Threshold lines
        ax_indicator.axhline(y=15, color=self.COLORS["muted"],
                            linestyle=":", linewidth=0.8, alpha=0.5)
        ax_indicator.axhline(y=25, color=self.COLORS["green"],
                            linestyle="--", linewidth=0.8, alpha=0.5)
        ax_indicator.axhline(y=35, color=self.COLORS["orange"],
                            linestyle="--", linewidth=0.8, alpha=0.5)

        # Zone labels
        ax_indicator.text(dates[-1], 10, "NO TRADE", fontsize=7,
                         color=self.COLORS["muted"], alpha=0.5)
        ax_indicator.text(dates[-1], 30, "CLEAN", fontsize=7,
                         color=self.COLORS["green"], alpha=0.5)
        ax_indicator.text(dates[-1], 40, "CAUTION", fontsize=7,
                         color=self.COLORS["orange"], alpha=0.5)

        ax_indicator.set_ylabel("ADX")
        ax_indicator.set_ylim(0, 60)
        ax_indicator.legend(loc="upper left", fontsize=7, framealpha=0.3)
        ax_indicator.grid(True, alpha=0.3)