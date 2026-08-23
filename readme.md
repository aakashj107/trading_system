# Indian Market Trading System — v3.3

> **Version:** 3.3
> **Last Updated:** 2026-08-15
> **Features:** Regime-Adaptive + VIX Risk Gate + Consensus + Watchlist + Short-Selling + Sector-Balanced + Risk-Managed + **Phase 1-4 Modifiers (BQ, Exhaustion, Sub-Regime, ADX Gate)** + **3 Pine Strategies** + **Parallel Pipeline (B2/B3)**
> **Stock Universe:** 50 stocks (sector-balanced, 9 sectors)
> **Strategies:** 17 (9 Original + 4 Bear Short + 4 Phase 1/2/4) + 3 Pine

---

## Table of Contents

1. [System Overview](#system-overview)
2. [What's New in v3.2](#whats-new-in-v32)
3. [Project Structure](#project-structure)
4. [Architecture](#architecture)
5. [Module Breakdown](#module-breakdown)
6. [Stock Universe](#stock-universe)
7. [Strategies](#strategies)
8. [Risk Management](#risk-management)
9. [System Sections (Pipeline)](#system-sections-pipeline)
10. [Charts & Visualizations](#charts--visualizations)
11. [How to Run](#how-to-run)
12. [Pending Items](#pending-items)
13. [Changelog](#changelog)

---

## System Overview

An automated Indian stock market trading system that:

- Detects **market regime** (Bull/Bear/Sideways) using Nifty technicals
- Applies **VIX-based risk gating** to control position sizing
- Performs **multi-factor equity research** (fundamentals, management, valuation, technicals, risks) on **all stocks**
- Runs **20+ trading strategies** (17 core + 3 Pine) with regime-adaptive weighting
- Supports **short-selling** in bear markets (4 dedicated bear strategies)
- Generates **consensus signals** with regime-adjusted thresholds and Phase 1-4 modifiers
- Maintains an **intelligent watchlist** with buy zones
- Enforces **sector concentration limits** and **correlation controls**
- Simulates **paper trading** with ATR-based dynamic stops
- Supports **live trading** via Angel One integration
- Generates **10+ chart types** automatically
- Stores everything in **SQLite database**

---

## What's New in v3.2

| # | Change | Impact |
|---|---|---|
| 1 | **Phase 1A: Break Quality Filter** | Universal candle-quality filter — filters weak candles (low body ratio, small body vs ATR) with OFF/SOFT/HARD modes |
| 2 | **Phase 1B: Regression SuperTrend** | Strategy #14 — Linear regression with RMSE adaptive bands, tighter than SMA/EMA |
| 3 | **Phase 2A: CHOCH/BOS Retest** | Strategy #15 — Structural entries with 3 confirmations: Break → Retest → Confirm |
| 4 | **Phase 2B: Liquidity Sweep** | Strategy #16 — Institutional flow detection: Sweep → Retest → Breakout |
| 5 | **Phase 3A: Exhaustion Detector** | Consensus modifier — detects trend fatigue via distance contraction + RSI divergence |
| 6 | **Phase 3B: Trend/Range Sub-Regime** | Second regime layer — classifies TREND vs RANGE within Bull/Bear/Sideways |
| 7 | **Phase 4A: CCI PREP/READY** | Strategy #17 — Two-stage mean reversion: ARM on extreme (±200) → FIRE on trigger (±100) |
| 8 | **Phase 4B: ADX Quality Gate** | Consensus modifier — 5-zone ADX filter (NO_TRADE/WEAK/OK/CLEAN/LATE) with directional adjustment |
| 9 | ATR-based dynamic stop losses | Paper trading stops are now adaptive to volatility instead of fixed % |
| 10 | Regime-aware consensus thresholds | Bear regime: CONVICTION≥3, STRONG≥2 (was 4/3) — more actionable signals |
| 11 | Red flag penalization in rankings | Stocks with Negative OCF, high leverage etc. get score penalties — SBIN drops from #1 |
| 12 | 4 bear short-selling strategies | Profit from falling prices in bear regime |
| 13 | Sector-balanced universe (50 stocks) | 9 sectors: Financial 24%, Tech 15%, Auto 15%, Pharma 12%, FMCG 10%, Energy 8%, Metals 8%, Infra 6%, Other 10% |
| 14 | Sector concentration limits | Max 35% per sector, max 3 positions per sector, max 2 correlated stocks |
| 15 | Full pipeline on all stocks | Daily runner now researches all 50 stocks automatically |
| 16 | 10+ chart types | Per-stock analysis, sector distribution, rankings, watchlist, heatmap, breadth, risk dashboard |
| 17 | Correlation groups | Single source of truth in config.py — synced across risk manager and scanner |
| 18 | Time-based exits | Positions exit after 30 bars if profit < 1% |
| 19 | 3 Pine Script strategies | RTS Adaptive, Structure Entry, Combined Structure+RTS |

---

## Project Structure
trading_system/
│
├── config.py ✅ v3.2 — Sector-balanced universe (50 stocks, 9 sectors), StockInfo dataclass, correlation groups, Phase 1-4 configs
├── data_fetcher.py ✅ v1 — Yahoo Finance data with caching
├── strategies.py ✅ v3.2 — 17 strategies (9 original + 4 bear short + 4 Phase 1/2/4) + Phase 1A/3A/3B/4B modifiers
├── backtester.py ✅ v1 — Multi-stock strategy backtesting
├── paper_trader.py ✅ v3.2 — ATR stops, short positions, time exits, exit reason tracking
├── main.py ✅ v3.2 — Red flag ranking engine, menu options 1-10
│
├── scanner.py ✅ v3.2 — Regime-aware consensus, BQ filter, exhaustion, sub-regime, ADX gate, CCI PREP
├── visualizer.py ✅ v3.2 — 10+ chart types, batch generation
├── daily_runner.py ✅ v3.2 — Full pipeline on all 50 stocks, 6 run modes
├── regime_detector.py ✅ v1 — Bull/Bear/Sideways detection
├── risk_manager.py ✅ v1 — VIX gate, sector limits, correlation limits, circuit breakers
│
├── alerts.py ✅ v1 — Telegram alerts
├── portfolio_manager.py ✅ v1 — Multi-stock portfolio tracking
├── database.py ✅ v1 — SQLite storage
├── angel_one_live.py ✅ v2 — Angel One SmartAPI client (SAFE: paper stub / paper_api read-only dry-run / live requires live_confirmed)
├── execution_harness.py ✅ v1 — SAFE signal→execution wiring (consensus → OrderManager → PortfolioManager → LiveTrader[paper] → AlertSystem); no real orders
│
├── verdict.py ✅ v1 — Final verdict generator (BUY/HOLD/AVOID)
├── fundamentals.py ✅ v1 — Fundamental analysis
├── management_analyzer.py ✅ v1 — Management DNA analysis
├── valuation.py ✅ v1 — Valuation analysis
├── strategies_pine.py ✅ v1 — 3 Pine Script strategies (RTS, Structure, Combined)
├── market_intelligence.py ✅ v1 — Full intelligence layer (regime, VIX, consensus, breadth, watchlist)
├── expanded_config.py ✅ v1 — 50 stock expanded universe
│
├── data_cache/ (auto-created) — Cached market data
├── charts/ (auto-created) — Generated chart images (10+ types)
├── paper_trading_logs/ (auto-created) — Paper trading logs
├── daily_logs/ (auto-created) — Daily reports, verdicts, signals
├── reports/ (auto-created) — Full reports
└── trading_system.db (auto-created) — SQLite database


---

## Architecture
┌──────────────────────────────────────────────────────────────┐
│ DAILY RUNNER │
│ daily_runner.py (6-mode pipeline) │
└──────────────────────────┬───────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────────┐
│ REGIME DETECTOR │
│ regime_detector.py │
│ │
│ ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐ │
│ │ Nifty Data │ │ VIX Data │ │ Market Breadth │ │
│ │ SMA/RSI/ATR │ │ Thresholds │ │ % Above SMA │ │
│ └──────┬──────┘ └──────┬──────┘ └────────┬──────────┘ │
│ └────────────────┼──────────────────┘ │
│ ▼ │
│ Regime: BULL / BEAR / SIDESWAYS │
│ Strategy Weights Adjusted │
└──────────────────────────┬───────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────────┐
│ SCANNER (with Phase 1-4 Modifiers) │
│ scanner.py │
│ │
│ ┌──────────────────┐ ┌──────────────────────────────────────┐ │
│ │ DATA FETCHER │ │ STRATEGIES (17 + 3 Pine) │ │
│ │ data_fetcher.py │ │ strategies.py │ │
│ │ │ │ │ │
│ │ • Yahoo Finance │ │ 9 Original: │ │
│ │ • Cache layer │ │ • SMA_Crossover_20_50 │ │
│ │ • 50 stocks │ │ • SMA_Crossover_50_200 │ │
│ │ │ │ • EMA_Crossover_12_26 │ │
│ │ │ │ • RSI_14_30_70 │ │
│ │ │ │ • MACD_12_26_9 │ │
│ │ │ │ • Bollinger_20_2.0 │ │
│ │ │ │ • Breakout_20_Vol1.5 │ │
│ │ │ │ • MeanReversion_50_2.0std │ │
│ │ │ │ • Momentum_ROC20 │ │
│ │ │ │ │ │
│ │ │ │ 4 Bear Short: │ │
│ │ │ │ • BearTrendShort │ │
│ │ │ │ • BearMomentumShort │ │
│ │ │ │ • BearBreakdownShort │ │
│ │ │ │ • BearMeanRevShort │ │
│ │ │ │ │ │
│ │ │ │ 4 Phase 1/2/4: │ │
│ │ │ │ • Regression_SuperTrend_20_2.0 │ │
│ │ │ │ • CHOCH_BOS_Retest_P5_5 │ │
│ │ │ │ • Liquidity_Sweep_P5_5 │ │
│ │ │ │ • CCI_Prep_20_200_100 │ │
│ │ │ │ │ │
│ │ │ │ + 3 Pine Scripts (if loaded) │ │
│ └──────────────────┘ └──────────────────────────────────────┘ │
│ │
│ Phase 1A: Break Quality Filter (OFF/SOFT/HARD) │
│ Phase 3A: Exhaustion Detection (Distance + RSI) │
│ Phase 3B: Trend/Range Sub-Regime (TREND/RANGE) │
│ Phase 4B: ADX Quality Gate (5 zones) │
│ │
│ Regime-Aware Consensus: │
│ • BEAR: CONVICTION≥3, STRONG≥2 │
│ • BULL: CONVICTION≥4, STRONG≥3 │
│ • Strategy quality weighting │
│ • Conflict resolution │
└──────────────────────────┬───────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────────┐
│ CONSENSUS + INTELLIGENCE │
│ │
│ • Market Breadth (50 stocks) │
│ • Intelligent Watchlist (buy zone tracking) │
│ • Sector Exposure Analysis │
│ • Final Recommendation (WAIT/BUY/PREPARE/WATCH) │
└──────────────────────────┬───────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────────┐
│ EXECUTION LAYER │
│ │
│ ┌──────────────┐ ┌───────────────┐ ┌──────────────────┐ │
│ │ BACKTESTER │ │ PAPER TRADER │ │ ANGEL ONE LIVE │ │
│ │ backtester.py │ │ paper_trader │ │ angel_one_live.py│ │
│ │ │ │ .py │ │ │ │
│ │ Strategy │ │ • ATR stops │ │ Live market │ │
│ │ validation │ │ • Short pos │ │ connection │ │
│ │ │ │ • ₹5,00,000 │ │ │ │
│ └──────────────┘ └───────────────┘ └────────┬─────────┘ │
│ │ │
│ ┌────────────────────────────┐ │
│ │ ORDER MANAGER │ │
│ │ order_manager.py │ │
│ └─────────────┬──────────────┘ │
│ │ │
│ ┌────────────────────────────┐ │
│ │ RISK MANAGER │ │
│ │ risk_manager.py │ │
│ │ │ │
│ │ • VIX-based sizing │ │
│ │ • Sector concentration │ │
│ │ • Correlation limits │ │
│ │ • Circuit breakers │ │
│ │ • Risk scoring (0-100) │ │
│ └─────────────┬──────────────┘ │
│ │ │
│ ┌────────────────────────────┐ │
│ │ PORTFOLIO MANAGER │ │
│ │ portfolio_manager.py │ │
│ └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────────┐
│ OUTPUT & STORAGE │
│ │
│ ┌──────────────┐ ┌───────────────┐ ┌──────────────────┐ │
│ │ VISUALIZER │ │ ALERTS │ │ DATABASE │ │
│ │ visualizer.py │ │ alerts.py │ │ database.py │ │
│ │ │ │ │ │ │ │
│ │ 10+ charts: │ │ • Telegram │ │ • SQLite │ │
│ │ • Per-stock │ │ • Signals │ │ • Trade logs │ │
│ │ • Sector │ │ • Regime │ │ • Signal history │ │
│ │ • Rankings │ │ • Risk events │ │ • Performance │ │
│ │ • Heatmap │ │ │ │ │ │
│ │ • Breadth │ │ │ │ │ │
│ │ • Risk dash │ │ │ │ │ │
│ └──────────────┘ └───────────────┘ └──────────────────┘ │
└──────────────────────────────────────────────────────────────┘

---

## Module Breakdown

### Core Modules

| File | Version | Purpose |
|------|---------|---------|
| **config.py** | v3.2 | Stock universe (50 stocks, 9 sectors), StockInfo dataclass, correlation groups, risk limits, VIX thresholds, Phase 1-4 configs |
| **data_fetcher.py** | v1 | Yahoo Finance data with caching for all 50 stocks |
| **strategies.py** | v3.2 | 17 strategies: 9 original + 4 bear short + 4 Phase 1/2/4 + Phase 1A/3A/3B/4B modifiers (BQ, Exhaustion, Sub-Regime, ADX Gate, CCI PREP) |
| **backtester.py** | v1 | Multi-stock backtesting with Sharpe, win rate, drawdown, profit factor |
| **paper_trader.py** | v3.2 | ATR-based stops, short position support, time-based exits, exit reason breakdown |
| **main.py** | v3.2 | Menu system with 10 options, red flag ranking engine |
| **scanner.py** | v3.2 | Regime-aware consensus, BQ filter, exhaustion, sub-regime, ADX gate, CCI PREP, conflict resolution |
| **visualizer.py** | v3.2 | 10+ chart types, batch generation |
| **daily_runner.py** | v3.2 | Full 6-mode pipeline on all 50 stocks |
| **regime_detector.py** | v1 | Bull/Bear/Sideways using Nifty SMA50, SMA200, RSI, ATR |
| **risk_manager.py** | v1 | VIX gate, sector concentration (35%/3 pos), correlation limits (2), circuit breakers, risk scoring (0-100) |

### Extended Modules

| File | Version | Purpose |
|------|---------|---------|
| **verdict.py** | v1 | Final verdict generator — BUY/HOLD/AVOID/STRONG_AVOID with conviction score |
| **fundamentals.py** | v1 | Revenue, ROE, debt ratios, cash flow analysis |
| **management_analyzer.py** | v1 | Promoter holding, governance scoring |
| **valuation.py** | v1 | PE, PB, DCF fair value — UNDERVALUED to EXTREMELY_EXPENSIVE |
| **alerts.py** | v1 | Telegram alerts for signals, regime changes, risk events |
| **portfolio_manager.py** | v1 | Multi-stock portfolio tracking, P&L per position |
| **database.py** | v1 | SQLite — trade logs, signal history, performance metrics |
| **angel_one_live.py** | v1 | Angel One live market data and order placement |
| **order_manager.py** | v1 | Place, modify, cancel orders; fill management |
| **strategies_pine.py** | v1 | 3 Pine Script strategies: RTS Adaptive, Structure Entry, Combined Structure+RTS |
| **market_intelligence.py** | v1 | Full intelligence layer — RegimeAdaptive, VIX Gate, Consensus, Breadth, Watchlist |
| **expanded_config.py** | v1 | 50 stock expanded universe with symbol aliases |

---

## Stock Universe

### Sector Distribution (50 stocks)

| Sector | Count | % | Priority 1 Stocks |
|--------|-------|---|-------------------|
| Financial Services | 12 | 24% | HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK, BAJFINANCE, BAJAJFINSV, SHRIRAMFIN |
| Technology | 8 | 16% | TCS, INFY, HCLTECH, WIPRO, TECHM, MPHASIS, LTM |
| Auto | 8 | 16% | MARUTI, M&M, **TMCV**, **TMPV**, EICHERMOT, HEROMOTOCO, BAJAJ-AUTO |
| Pharma | 6 | 12% | SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, LUPIN |
| FMCG | 5 | 10% | HINDUNILVR, ITC, NESTLEIND, TATACONSUM, BRITANNIA |
| Energy | 4 | 8% | RELIANCE, ONGC, BPCL, NTPC |
| Metals | 4 | 8% | TATASTEEL, JSWSTEEL, HINDALCO, COALINDIA |
| Infrastructure | 3 | 6% | LT, ADANIPORTS, POWERGRID |
| Other | 5 | 10% | BHARTIARTL, TITAN, ULTRACEMCO, TRENT, ZOMATO, APOLLOHOSP |

### Priority System

| Priority | Meaning | Count | When Used |
|----------|---------|-------|-----------|
| 1 | Must scan | ~30 | `mode="core"` |
| 1+2 | Core + Extended | ~40 | `mode="active"` (default) |
| 1+2+3 | Full universe | ~50 | `mode="all"` |

### Correlation Groups

| Group | Stocks | Max Positions |
|-------|--------|---------------|
| PRIVATE_BANKS | HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK, INDUSINDBK | 2 |
| PSU_BANKS | SBIN | — |
| NBFC | BAJFINANCE, BAJAJFINSV, SHRIRAMFIN, MUTHOOTFIN | 2 |
| INSURANCE | HDFCLIFE, SBILIFE | 2 |
| IT_SERVICES | TCS, INFY, HCLTECH, WIPRO, TECHM, MPHASIS, LTM | 2 |
| PHARMA | SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, LUPIN | 2 |
| AUTO | MARUTI, M&M, TMCV, TMPV, EICHERMOT, HEROMOTOCO, BAJAJ-AUTO | 2 |
| OIL_GAS | RELIANCE, ONGC, BPCL | 2 |
| METALS | JSWSTEEL, TATASTEEL, HINDALCO, COALINDIA | 2 |
| ADANI | ADANIENT, ADANIPORTS | 2 |
---

## Strategies

### Original 9 Strategies

| Type | Category | Strategy | Description |
|------|----------|----------|-------------|
| ORIGINAL | trend | SMA_Crossover_20_50 | Fast/Slow SMA crossover |
| ORIGINAL | trend | SMA_Crossover_50_200 | Golden/Death cross |
| ORIGINAL | trend | EMA_Crossover_12_26 | Fast/Slow EMA crossover |
| ORIGINAL | trend | Breakout_20_Vol1.5 | Channel breakout with volume |
| ORIGINAL | momentum | MACD_12_26_9 | MACD line/signal crossover |
| ORIGINAL | momentum | Momentum_ROC20 | Rate of change crossover |
| ORIGINAL | mean_reversion | RSI_14_30_70 | RSI oversold/overbought |
| ORIGINAL | mean_reversion | Bollinger_20_2.0 | Bollinger Band bounce |
| ORIGINAL | mean_reversion | MeanReversion_50_2.0std | Z-score mean reversion |

### Bear Short-Selling Strategies (4)

| Type | Category | Strategy | Description |
|------|----------|----------|-------------|
| BEAR | bear_short | BearTrendShort | Short when price breaks below EMA20 + RSI < 45 + volume |
| BEAR | bear_short | BearMomentumShort | Short on MACD bearish crossover + histogram expansion + below SMA50 |
| BEAR | bear_short | BearBreakdownShort | Short on high-volume breakdowns below 20-day support |
| BEAR | bear_short | BearMeanRevShort | Short on overbought bounces in downtrend (sell the rally) |

### Phase 1/2/4 Strategies (4)

| Type | Category | Strategy | Description |
|------|----------|----------|-------------|
| PHASE 1B | trend | Regression_SuperTrend_20_2.0 | Linear regression with RMSE adaptive bands |
| PHASE 2A | structure | CHOCH_BOS_Retest_P5_5 | CHOCH/BOS with 3-confirmation entry (Break→Retest→Confirm) |
| PHASE 2B | structure | Liquidity_Sweep_P5_5 | Institutional sweep detection → Retest → Breakout |
| PHASE 4A | mean_reversion | CCI_Prep_20_200_100 | Two-stage CCI: ARM at ±200 → FIRE at ±100 |

### Pine Script Strategies (3)

| Type | Category | Strategy | Description |
|------|----------|----------|-------------|
| PINE | adaptive | RTS_Adaptive_ALMA_21_T2.0 | Reactive Trail System with ALMA trailing stops |
| PINE | structure | Structure_Reg20_Choch_ADX20 | Structure entry (WAIT+ADX+RSI+CCI) + Regression SuperTrend exit |
| PINE | structure | Combined_Structure_RTS | Structure entry + RTS adaptive trailing exit |

### Regime-Based Strategy Weights

| Category | BULL | BEAR | SIDEWAYS |
|----------|------|------|----------|
| trend | 1.5x | 0.3x | 1.0x |
| momentum | 1.2x | 0.3x | 1.0x |
| mean_reversion | 0.5x | 1.2x | 1.0x |
| structure | 1.0x | 1.0x | 1.0x |
| adaptive | 1.0x | 0.8x | 1.0x |
| bear_short | 0.0x | 1.3x | 0.5x |

### Strategy Quality Weights (for Consensus)

| Strategy | Weight | Based On |
|----------|--------|----------|
| Bollinger_20_2.0 | 1.5x | Best Sharpe in bear |
| BearTrendShort | 1.3x | Bear-specific |
| MACD_12_26_9 | 1.3x | Best momentum in bear |
| BearMomentumShort | 1.2x | Bear-specific |
| MeanReversion_50_2.0std | 1.2x | Good mean reversion |
| Structure_Reg20_Choch_ADX20 | 1.1x | Pine strategy |
| BearBreakdownShort | 1.1x | Bear-specific |
| RSI_14_30_70 | 1.0x | Standard mean reversion |
| BearMeanRevShort | 1.0x | Bear-specific |
| RTS_Adaptive_ALMA_21_T2.0 | 0.9x | Pine adaptive |
| EMA_Crossover_12_26 | 0.7x | Weak in bear |
| Combined_Structure_RTS | 0.7x | Pine structure |
| Breakout_20_Vol1.5 | 0.5x | Weak in bear |
| SMA_Crossover_20_50 | 0.4x | Weak in bear |
| Momentum_ROC20 | 0.4x | Weak in bear |
| SMA_Crossover_50_200 | 0.3x | Very few signals |

### Phase 1-4 Modifiers (Applied in Consensus)

| Modifier | Type | Description |
|----------|------|-------------|
| Break Quality Filter (BQ) | Candle filter | OFF/SOFT/HARD modes — filters weak candles by body ratio and body/ATR |
| Exhaustion Detector | Consensus modifier | Downgrades trend signals when distance contraction + RSI divergence detected |
| Trend/Range Classifier | Sub-regime detector | TREND vs RANGE based on flip count, ADX, EMA slope — adjusts category weights |
| ADX Quality Gate | Consensus modifier | 5-zone ADX filter (NO_TRADE/WEAK/OK/CLEAN/LATE) with directional adjustment |
| CCI PREP/READY | Strategy | Two-stage mean reversion: ARM on extreme (±200) → FIRE on trigger (±100) |

---

## Risk Management

### VIX Risk Gate

| VIX Level | Risk Mode | Position Size | Max Positions |
|-----------|-----------|---------------|---------------|
| < 13 | Aggressive | 15% | 5 |
| < 17 | Normal | 12% | 4 |
| < 22 | Cautious | 10% | 3 |
| < 28 | Defensive | 7% | 2 |
| < 99 | Cash Preservation | 4% | 1 |

### Position-Level Controls

| Control | Limit | Description |
|---------|-------|-------------|
| ATR-based stop loss | 2x ATR | Dynamic stop based on volatility |
| ATR-based trailing stop | 1.5x ATR | Trailing stop follows price |
| Time-based exit | 30 bars | Exit if profit < 1% after 30 bars |
| Short profit target | 8% | Auto-close short at 8% profit |
| Short margin | 20% | Margin requirement for shorts |

### Portfolio-Level Controls

| Control | Limit | Description |
|---------|-------|-------------|
| Max sector exposure | 35% | No more than 35% in one sector |
| Max sector positions | 3 | No more than 3 stocks in same sector |
| Max correlated positions | 2 | No more than 2 stocks in same correlation group |
| Daily loss limit | 3% | Circuit breaker if daily loss exceeds 3% |
| Max drawdown | 10% | Circuit breaker if drawdown exceeds 10% |
| Cool-down period | 3 days | Circuit breaker stays active for 3 days |

### Risk Scoring (0-100)

| Factor | Max Points | What |
|--------|------------|------|
| Position count | 15 | How close to max positions |
| Sector concentration | 25 | Heaviest sector % |
| Correlation | 20 | Most correlated group count |
| VIX | 20 | Current volatility level |
| Drawdown | 20 | Distance from peak capital |

Levels: LOW (0-24), MODERATE (25-49), HIGH (50-74), CRITICAL (75-100)

### Pre-Trade Checks (5 gates)

1. Circuit breaker active?
2. Max positions reached? (VIX-adjusted)
3. Sector concentration OK?
4. Correlation limits OK?
5. Position size > 0?

---

## System Sections (Pipeline)

### 6-Mode Daily Runner

| Mode | Stocks | Speed | Use Case |
|------|--------|-------|----------|
| quick | 10 | Fast | Testing changes |
| core | ~30 | Medium | Daily quick scan |
| active | ~40 | Standard | Default daily run |
| all | ~50 | Slow | Full weekend analysis |
| scan | 10 | Fast | Signal scan only |
| report | 10 | Fast | Sector distribution only |

### Full Pipeline (10 Sections)

| # | Section | File | What It Does |
|---|---------|------|--------------|
| 1 | Market Regime | regime_detector.py | Detect BULL/BEAR/SIDESWAYS from Nifty |
| 2 | VIX Risk Gate | risk_manager.py | Set position sizing from India VIX |
| 3 | Data Loading | data_fetcher.py | Pre-load all stock data (cached) |
| 4 | Equity Research | verdict.py + analyzers | Research ALL loaded stocks (fundamentals, management, valuation, technicals, risks) |
| 5 | Strategy Comparison | backtester.py | Backtest regime-appropriate strategies |
| 6 | Signal Scan + Consensus | scanner.py | Scan all stocks × all strategies, build consensus with Phase 1-4 modifiers |
| 7 | Market Intelligence | daily_runner.py | Breadth + Watchlist + Recommendation |
| 8 | Paper Trading | paper_trader.py | Simulate with ATR stops and short support |
| 9 | Charts | visualizer.py | Generate 10+ chart types |
| 10 | Final Summary | daily_runner.py | Rankings + Action + Save reports |

### Consensus Signal Strength

| Level | BULL Threshold | BEAR Threshold | Meaning |
|-------|---------------|----------------|---------|
| CONVICTION | 4+ strategies | 3+ strategies | Strongest alignment |
| STRONG | 3+ strategies | 2+ strategies | High priority |
| MODERATE | 2+ strategies | 2+ strategies | Notable |
| WEAK | 1 strategy | 1 strategy | Single signal |

### Intelligent Watchlist Status

| Status | Meaning |
|--------|---------|
| IN BUY ZONE | Price at or below buy zone — ready to buy |
| VERY CLOSE | Within 5% of buy zone |
| APPROACHING | Within 15% of buy zone |
| MONITORING | Within 20% of buy zone |
| FAR | More than 20% away |
| PASSED | Already below buy zone |
| APPROACHING (AVOID) | Near buy zone but flagged AVOID |

---

## Charts & Visualizations

### 10+ Chart Types

| Chart | Function | Description |
|-------|----------|-------------|
| Per-Stock Analysis | `plot_stock_analysis()` | Price + SMA20/50/200 + Bollinger + Volume + RSI + MACD + Verdict overlay |
| Sector Distribution | `plot_sector_distribution()` | Pie chart + horizontal bar of sector breakdown |
| Stock Rankings | `plot_stock_rankings()` | Conviction scores + stacked score breakdown |
| Multi-Stock Overview | `plot_multi_stock_overview()` | Grid of all stocks with action badges and mini bars |
| Watchlist | `plot_watchlist()` | Gap to buy zone with color-coded bars |
| Consensus Heatmap | `plot_consensus_heatmap()` | Strategy × Stock signal grid |
| Market Breadth | `plot_market_breadth()` | Gauge showing % above SMA20 + stats |
| Risk Dashboard | `plot_risk_dashboard()` | Risk gauge + VIX + Sector exposure + Risk factors |
| Equity Curve | `plot_equity_curve()` | Portfolio value over time with drawdown shading |
| Strategy Comparison | `plot_strategy_comparison()` | Return, Win Rate, Sharpe, Max DD per strategy |
| Paper Trading | `plot_paper_dashboard()` | Portfolio value, return %, positions, trade P&L |

### Batch Generation

```python
viz = Visualizer()
viz.generate_all_charts(
    verdicts=verdicts,
    signals_df=signals_df,
    watchlist=watchlist,
    breadth_data=breadth_data,
    risk_data=risk_dashboard,
    sector_data=sector_counts,
    stock_data_cache=stock_data_cache,
)

charts/
├── sector_distribution.png
├── stock_rankings.png
├── multi_stock_overview.png
├── watchlist.png
├── consensus_heatmap.png
├── market_breadth.png
├── risk_dashboard.png
├── HDFCBANK_analysis.png
├── ICICIBANK_analysis.png
├── TCS_analysis.png
├── ...
└── comparison.png

# Full pipeline on all 50 stocks
python daily_runner.py
# Enter: 3

# Full pipeline on core stocks only (~30, faster)
python daily_runner.py
# Enter: 1

# Full pipeline on quick test (10 stocks, fastest)
python daily_runner.py
# Enter: 4

python daily_runner.py
# Enter: 5

python config.py
# or
python daily_runner.py
# Enter: 6

python main.py
# Enter: 1 (single stock research)
# Enter: 3 (research + rankings with red flag penalty)

runner = DailyRunner(mode="active")
runner.schedule_daily()  # Runs at 09:30 and 15:15 IST

Run Modes

| Mode | Stocks | Speed | Use Case |
|------|--------|-------|----------|
| quick | 10 | Fast | Testing changes |
| core | ~30 | Medium | Daily quick scan |
| active | ~40 | Standard | Default daily run |
| all | ~50 | Slow | Full weekend analysis |
| scan | 10 | Fast | Signal scan only |
| report | 10 | Fast | Sector distribution only |

Pending Items

| # | Item | Priority | Status |
|---|------|----------|--------|
| 1 | ~Add tmpv and tmcv~ | HIGH | ✅ Done |
| 2 | ~Fix LTIM.NS data~ | MEDIUM | ✅ Done |
| 3 | ~ATR-based dynamic stops~ | HIGH | ✅ Done |
| 4 | ~Regime-aware consensus~ | HIGH | ✅ Done |
| 5 | ~Red flag penalization~ | MEDIUM | ✅ Done |
| 6 | ~Short-selling strategies~ | HIGH | ✅ Done |
| 7 | ~Sector-balanced universe~ | MEDIUM | ✅ Done |
| 8 | ~Sector concentration limits~ | MEDIUM | ✅ Done |
| 9 | ~Expand research to all stocks~ | MEDIUM | ✅ Done |
| 10 | ~Per-stock charts~ | LOW | ✅ Done |
| 11 | ~Integrate Pine Script strategies~ | MEDIUM | ✅ Done |
| 12 | Test Angel One live connection | MEDIUM | ✅ Done — paper_api read-only dry-run confirmed live connectivity (authenticate=True, 5/5 LTP fetched, 0 orders). Live mode still requires explicit live_confirmed=True. Tests: tests/test_angel_one_live.py |
| 13 | Validate order manager lifecycle | MEDIUM | ✅ Done — execution_harness.py drives ConsensusFilter(min_agree=3)+Donchian → OrderManager → PortfolioManager → LiveTrader(paper) → AlertSystem; 53 BUY/47 SELL fills, limits enforced, 0 open at close. Covered by tests/test_execution_harness.py |
| 14 | Test Telegram alerts end-to-end | LOW | Partial — AlertSystem wired + called in harness; silent no-op without TELEGRAM_BOT_TOKEN (safe). Needs real token to confirm delivery |
| 15 | Populate SQLite with historical data | LOW | Not started |
| 16 | Achieve positive paper trading returns | HIGH | ✅ Validated OOS: +27.0% over 2022-06→2026-08 (4y, ~6%/yr), PF 1.78, win 51.1% — 18-strat(+Donchian, min_agree=3) beats 17-strat baseline (+17.0%, PF 1.24). Long-only. See daily_logs/walkforward_OOS_2026-08-14.log |
| 17 | Go live with real capital | LOW | BLOCKED — requires #12 live auth + real-funds dry-run; edge validated but execution path only tested in paper mode |
| 18 | pytest suite + CI (A1–A4/B2/B3 regression guard) | HIGH | ✅ Done (18 tests, .github/workflows/test.yml) |


Changelog

v3.3 (2026-08-14) — Optimization & Correctness Pass
- FIX A1: risk_manager.py now imports CORRELATION_GROUPS from config.py (single source of truth) — removed stale duplicate that referenced non-existent TATAMOTORS.NS
- FIX A3: Removed dead CCIPrepReady class (duplicate of CCIPrepStrategy) from strategies.py
- FIX A4: backtester.py now handles SHORT (-2) / COVER (2) signals with correct capital accounting and LONG/SHORT direction tracking
- FIX A4 (bear strategies): BearTrendShort, BearMomentumShort, BearMeanRevShort converted to stateful SHORT/COVER pairing (COVER no longer overrides SHORT); BearBreakdownShort NaN-dropna bug fixed (Breakdown_Level initialized to 0.0 instead of NaN)
- FIX A2: paper_trader.py confirmed as a complete, importable module (long + short positions, ATR stops, circuit breaker, Kelly/volatility sizing)
- OPT B2/B3: Parallel data loading (ThreadPoolExecutor, 8 workers) in daily_runner._load_all_data; scanner.scan() now accepts data_cache and parallelizes per-stock fetch + scan — eliminates redundant network refetch in the daily pipeline
- TEST: Added network-free pytest suite (tests/) — 18 tests covering A1 risk-data unification, A3 dead-class removal, A4 backtester SHORT/COVER + bear-strategy signal pairing, A2 paper trader, B2/B3 parallel-scan correctness. Added pytest.ini + .github/workflows/test.yml (CI on Py 3.10–3.12).
- VALIDATION: Honest walk-forward (validate_walkforward.py) on 19-symbol universe, 10y data, strict IS→OOS split. 18-strat(+Donchian, min_agree=3) validated OOS: +27.0% over 2022-06→2026-08 (4y, ~6%/yr), PF 1.78, win 51.1% — beats 17-strat baseline (+17.0%, PF 1.24). Long-only (short leg proven drag). Log: daily_logs/walkforward_OOS_2026-08-14.log
- EXECUTION HARNESS: execution_harness.py wires the validated config (ConsensusFilter min_agree=3 + Donchian) → OrderManager → PortfolioManager → LiveTrader(paper) → AlertSystem, simulation-only (no real orders/credentials). Validates #13 order lifecycle: BUY/SELL fills, P&L, sector/max-position limits enforced, positions closed at end. Test: tests/test_execution_harness.py.
- ANGEL ONE #12: angel_one_live.py rewritten as a genuine SmartAPI client (generateSession / getProfile / ltpData / placeOrder). SAFE-by-default three-mode design: "paper" (print stub), "paper_api" (authenticated READ-ONLY dry-run — fetches live LTP, never places), "live" (refuses unless live_confirmed=True explicitly). 6 tests in tests/test_angel_one_live.py prove no order can fire without the two-key gate. paper_api dry-run CONFIRMED live connectivity (authenticate=True, 5/5 LTP fetched, 0 orders placed) on 2026-08-15.
v3.2 (2026-08-01)
Phase 1A: Break Quality Filter (OFF/SOFT/HARD) — universal candle filter
Phase 1B: Regression SuperTrend strategy (#14) — RMSE adaptive bands
Phase 2A: CHOCH/BOS Retest strategy (#15) — 3-confirmation structural entries
Phase 2B: Liquidity Sweep strategy (#16) — Institutional flow detection
Phase 3A: Exhaustion Detector — Trend fatigue detection via distance contraction + RSI divergence
Phase 3B: Trend/Range Sub-Regime Classifier — TREND vs RANGE within regimes
Phase 4A: CCI PREP/READY strategy (#17) — Two-stage mean reversion (ARM→FIRE)
Phase 4B: ADX Quality Gate — 5-zone ADX filter with directional adjustment
3 Pine Script strategies integrated (RTS, Structure, Combined)
Stock universe expanded to 50 stocks across 9 sectors
Correlation groups expanded (added ADANI group)
6-mode daily runner (quick/core/active/all/scan/report)
Consensus pipeline enhanced with all Phase 1-4 modifiers

v3.1 (2026-07-26)
ATR-based dynamic stop losses (2x ATR stop, 1.5x ATR trailing)
Time-based exits (30 bars, 1% minimum profit)
4 bear short-selling strategies
Regime-aware consensus thresholds (BEAR: CONVICTION≥3, STRONG≥2)
Strategy quality weighting for consensus
Conflict resolution for opposing signals
Red flag penalization in stock rankings
Sector-balanced universe (45 stocks, 8 sectors)
Sector concentration limits (35%, 3 positions)
Correlation group limits (2 correlated max)
VIX-based position sizing integrated into risk manager
Portfolio risk scoring (0-100)
Full 10-section pipeline on all stocks
10+ chart types with batch generation
Short position support in paper trader
Exit reason tracking and breakdown
Data pre-loading with caching
6 run modes (core/active/all/quick/scan/report)

v3.0 (2026-07-26)
Initial report generation
12 strategies (9 original + 3 Pine)
15 stocks researched
VIX risk gate
Market regime detection
Watchlist with buy zones
Paper trading simulation


Documentation maintained by the Indian Market Trading System. Last updated: 2026-08-01

