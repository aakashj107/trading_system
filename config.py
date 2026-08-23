"""
Indian Market Trading System — Configuration
v3.1 — Sector-balanced universe, sector tagging, concentration limits.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


@dataclass
class MarketConfig:
    """Indian market specific settings."""
    market_open: str = "09:15"
    market_close: str = "15:30"
    timezone: str = "Asia/Kolkata"
    default_exchange: str = "NSE"


@dataclass
class TradingCosts:
    """
    Realistic Indian market transaction costs.
    """
    brokerage_per_order: float = 20.0
    stt_delivery_buy: float = 0.0        # No STT on delivery BUY (NSE equity)
    stt_delivery_sell: float = 0.001     # STT 0.1% on delivery SELL
    stt_intraday_sell: float = 0.00025
    exchange_txn_charge: float = 0.0000345
    sebi_turnover_fee: float = 0.000001
    stamp_duty_buy: float = 0.00015
    gst_on_brokerage: float = 0.18


@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_risk_per_trade: float = 0.02
    max_open_positions: int = 5
    max_portfolio_risk: float = 0.10
    default_stop_loss_pct: float = 0.03
    trailing_stop: bool = True
    trailing_stop_pct: float = 0.02

    # NEW: Sector concentration limits
    max_sector_exposure_pct: float = 0.35    # Max 35% in any single sector
    max_sector_positions: int = 3            # Max 3 positions in same sector
    max_correlated_positions: int = 2        # Max 2 highly correlated stocks


@dataclass
class BacktestConfig:
    """Backtesting parameters."""
    initial_capital: float = 500000.0
    position_size_pct: float = 0.20
    start_date: str = "2022-01-01"
    end_date: str = "2025-07-18"
    benchmark: str = "^NSEI"


@dataclass
class PaperTradingConfig:
    """Paper trading parameters."""
    initial_capital: float = 500000.0
    log_file: str = "paper_trades.csv"
    portfolio_file: str = "paper_portfolio.csv"


# ============================================================
#  SECTOR DEFINITIONS
# ============================================================

class Sector(Enum):
    FINANCIAL_SERVICES = "Financial Services"
    TECHNOLOGY = "Technology"
    FMCG = "FMCG"
    PHARMA = "Pharma"
    AUTO = "Auto"
    ENERGY = "Energy"
    METALS = "Metals"
    INFRASTRUCTURE = "Infrastructure"
    CONSUMER = "Consumer"
    TELECOM = "Telecom"
    CEMENT = "Cement"
    DIVERSIFIED = "Diversified"


@dataclass
class StockInfo:
    """Stock metadata with sector tagging."""
    symbol: str
    sector: Sector
    sub_sector: str = ""
    market_cap: str = "LARGE"     # LARGE, MID, SMALL
    priority: int = 1              # 1 = must scan, 2 = optional, 3 = extended


# ============================================================
#  SECTOR-BALANCED STOCK UNIVERSE (FIXED)
# ============================================================

# Correlation groups — stocks that move together
CORRELATION_GROUPS = {
    "PRIVATE_BANKS": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"],
    "PSU_BANKS": ["SBIN.NS"],
    "NBFC": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "SHRIRAMFIN.NS", "MUTHOOTFIN.NS"],
    "INSURANCE": ["HDFCLIFE.NS", "SBILIFE.NS"],
    "IT_SERVICES": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "MPHASIS.NS"],
    "PHARMA": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
    "AUTO": ["MARUTI.NS", "M&M.NS", "TMCV.NS", "TMPV.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS"],
    "OIL_GAS": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS"],
    "METALS": ["JSWSTEEL.NS", "TATASTEEL.NS", "HINDALCO.NS", "COALINDIA.NS"],
    "ADANI": ["ADANIENT.NS", "ADANIPORTS.NS"],
}


STOCK_UNIVERSE: Dict[str, StockInfo] = {

    # ── FINANCIAL SERVICES (8 stocks — down from 12) ──────────
    "HDFCBANK.NS":    StockInfo("HDFCBANK.NS",    Sector.FINANCIAL_SERVICES, "Private Bank",    "LARGE", 1),
    "ICICIBANK.NS":   StockInfo("ICICIBANK.NS",   Sector.FINANCIAL_SERVICES, "Private Bank",    "LARGE", 1),
    "SBIN.NS":        StockInfo("SBIN.NS",        Sector.FINANCIAL_SERVICES, "PSU Bank",        "LARGE", 1),
    "KOTAKBANK.NS":   StockInfo("KOTAKBANK.NS",   Sector.FINANCIAL_SERVICES, "Private Bank",    "LARGE", 1),
    "AXISBANK.NS":    StockInfo("AXISBANK.NS",    Sector.FINANCIAL_SERVICES, "Private Bank",    "LARGE", 1),
    "BAJFINANCE.NS":  StockInfo("BAJFINANCE.NS",  Sector.FINANCIAL_SERVICES, "NBFC",            "LARGE", 1),
    "BAJAJFINSV.NS":  StockInfo("BAJAJFINSV.NS",  Sector.FINANCIAL_SERVICES, "NBFC",            "LARGE", 1),
    "SHRIRAMFIN.NS":  StockInfo("SHRIRAMFIN.NS",  Sector.FINANCIAL_SERVICES, "NBFC",            "LARGE", 1),

    # ── TECHNOLOGY (5 stocks — up from 3) ─────────────────────
    "TCS.NS":         StockInfo("TCS.NS",         Sector.TECHNOLOGY,         "IT Services",     "LARGE", 1),
    "INFY.NS":        StockInfo("INFY.NS",        Sector.TECHNOLOGY,         "IT Services",     "LARGE", 1),
    "HCLTECH.NS":     StockInfo("HCLTECH.NS",     Sector.TECHNOLOGY,         "IT Services",     "LARGE", 1),
    "WIPRO.NS":       StockInfo("WIPRO.NS",       Sector.TECHNOLOGY,         "IT Services",     "LARGE", 2),
    "TECHM.NS":       StockInfo("TECHM.NS",       Sector.TECHNOLOGY,         "IT Services",     "LARGE", 2),

    # ── FMCG / CONSUMER (4 stocks — was 0) ───────────────────
    "HINDUNILVR.NS":  StockInfo("HINDUNILVR.NS",  Sector.FMCG,               "Staples",         "LARGE", 1),
    "ITC.NS":         StockInfo("ITC.NS",         Sector.FMCG,               "Staples/Diversified","LARGE", 1),
    "NESTLEIND.NS":   StockInfo("NESTLEIND.NS",   Sector.FMCG,               "Food",            "LARGE", 2),
    "TATACONSUM.NS":  StockInfo("TATACONSUM.NS",  Sector.FMCG,               "Staples",         "LARGE", 2),

    # ── PHARMA (4 stocks — was 0) ────────────────────────────
    "SUNPHARMA.NS":   StockInfo("SUNPHARMA.NS",   Sector.PHARMA,             "Large Pharma",    "LARGE", 1),
    "DRREDDY.NS":     StockInfo("DRREDDY.NS",     Sector.PHARMA,             "Generics",        "LARGE", 1),
    "CIPLA.NS":       StockInfo("CIPLA.NS",       Sector.PHARMA,             "Generics",        "LARGE", 2),
    "DIVISLAB.NS":    StockInfo("DIVISLAB.NS",    Sector.PHARMA,             "CDMO",            "LARGE", 2),

    # ── AUTO (4 stocks — includes tmpv, tmcv) ────────────────
    "MARUTI.NS":      StockInfo("MARUTI.NS",      Sector.AUTO,               "Passenger",       "LARGE", 1),
    "M&M.NS":         StockInfo("M&M.NS",         Sector.AUTO,               "Passenger/Utility","LARGE", 1),
    "TMCV.NS":        StockInfo("TMCV.NS",        Sector.AUTO,               "Commercial",      "LARGE", 1),
    "TMPV.NS":        StockInfo("TMPV.NS",        Sector.AUTO,               "Passenger",       "LARGE", 1),

    # ── ENERGY (3 stocks — was 0 standalone) ─────────────────
    "RELIANCE.NS":    StockInfo("RELIANCE.NS",    Sector.ENERGY,             "Oil & Gas/Conglom","LARGE", 1),
    "ONGC.NS":        StockInfo("ONGC.NS",        Sector.ENERGY,             "Oil & Gas",       "LARGE", 2),
    "NTPC.NS":        StockInfo("NTPC.NS",        Sector.ENERGY,             "Power",           "LARGE", 2),

    # ── METALS (3 stocks — was 0) ────────────────────────────
    "TATASTEEL.NS":   StockInfo("TATASTEEL.NS",   Sector.METALS,             "Steel",           "LARGE", 1),
    "JSWSTEEL.NS":    StockInfo("JSWSTEEL.NS",    Sector.METALS,             "Steel",           "LARGE", 2),
    "HINDALCO.NS":    StockInfo("HINDALCO.NS",    Sector.METALS,             "Aluminium/Copper","LARGE", 2),

    # ── INFRASTRUCTURE (3 stocks — was 0) ────────────────────
    "LT.NS":          StockInfo("LT.NS",          Sector.INFRASTRUCTURE,     "EPC/Conglom",     "LARGE", 1),
    "ADANIPORTS.NS":  StockInfo("ADANIPORTS.NS",  Sector.INFRASTRUCTURE,     "Ports",           "LARGE", 2),
    "POWERGRID.NS":   StockInfo("POWERGRID.NS",   Sector.INFRASTRUCTURE,     "Power T&D",       "LARGE", 2),

    # ── OTHER / DIVERSIFIED (3 stocks) ───────────────────────
    "BHARTIARTL.NS":  StockInfo("BHARTIARTL.NS",  Sector.TELECOM,            "Telecom",         "LARGE", 1),
    "TITAN.NS":       StockInfo("TITAN.NS",       Sector.CONSUMER,           "Jewelry/Lifestyle","LARGE", 1),
    "ULTRACEMCO.NS":  StockInfo("ULTRACEMCO.NS",  Sector.CEMENT,             "Cement",          "LARGE", 2),

    # ── MOVED OUT OF CORE (were taking space) ────────────────
    # These are now in EXTENDED list below
    "INDUSINDBK.NS":  StockInfo("INDUSINDBK.NS",  Sector.FINANCIAL_SERVICES, "Private Bank",    "LARGE", 2),
    "HDFCLIFE.NS":    StockInfo("HDFCLIFE.NS",    Sector.FINANCIAL_SERVICES, "Insurance",       "LARGE", 3),
    "SBILIFE.NS":     StockInfo("SBILIFE.NS",     Sector.FINANCIAL_SERVICES, "Insurance",       "LARGE", 3),
    "MUTHOOTFIN.NS":  StockInfo("MUTHOOTFIN.NS",  Sector.FINANCIAL_SERVICES, "Gold Loan NBFC",  "LARGE", 2),
    "MPHASIS.NS":     StockInfo("MPHASIS.NS",     Sector.TECHNOLOGY,         "IT Services",     "MID",   2),

    # ── EXTENDED WATCHLIST ────────────────────────────────────
    "ADANIENT.NS":    StockInfo("ADANIENT.NS",    Sector.DIVERSIFIED,        "Conglomerate",    "LARGE", 3),
    "ASIANPAINT.NS":  StockInfo("ASIANPAINT.NS",  Sector.CONSUMER,           "Paints",          "LARGE", 2),
    "BRITANNIA.NS":   StockInfo("BRITANNIA.NS",   Sector.FMCG,               "Food",            "LARGE", 2),
    "EICHERMOT.NS":   StockInfo("EICHERMOT.NS",   Sector.AUTO,               "Two-Wheeler",     "LARGE", 2),
    "HEROMOTOCO.NS":  StockInfo("HEROMOTOCO.NS",  Sector.AUTO,               "Two-Wheeler",     "LARGE", 2),
    "BAJAJ-AUTO.NS":  StockInfo("BAJAJ-AUTO.NS",  Sector.AUTO,               "Two-Wheeler",     "LARGE", 2),
    "GRASIM.NS":      StockInfo("GRASIM.NS",      Sector.CEMENT,             "Cement/Chemicals","LARGE", 3),
    "APOLLOHOSP.NS":  StockInfo("APOLLOHOSP.NS",  Sector.PHARMA,             "Hospitals",       "LARGE", 2),
    "BPCL.NS":        StockInfo("BPCL.NS",        Sector.ENERGY,             "Oil & Gas",       "LARGE", 3),
    "COALINDIA.NS":   StockInfo("COALINDIA.NS",   Sector.ENERGY,             "Coal Mining",     "LARGE", 3),
    "UPL.NS":         StockInfo("UPL.NS",         Sector.DIVERSIFIED,        "Agrochemicals",   "LARGE", 3),
    "LUPIN.NS":       StockInfo("LUPIN.NS",       Sector.PHARMA,             "Generics",        "LARGE", 3),
}


# ============================================================
#  STOCK LIST BUILDERS
# ============================================================

def get_core_stocks() -> List[str]:
    """Priority 1 stocks — always scan these."""
    return [s for s, info in STOCK_UNIVERSE.items() if info.priority == 1]


def get_active_stocks() -> List[str]:
    """Priority 1 + 2 stocks — core + extended."""
    return [s for s, info in STOCK_UNIVERSE.items() if info.priority <= 2]


def get_all_stocks() -> List[str]:
    """All stocks in universe."""
    return list(STOCK_UNIVERSE.keys())


def get_sector_stocks(sector: Sector) -> List[str]:
    """Get all stocks in a specific sector."""
    return [s for s, info in STOCK_UNIVERSE.items() if info.sector == sector]


def get_stocks_by_priority(priority: int) -> List[str]:
    """Get stocks by priority level."""
    return [s for s, info in STOCK_UNIVERSE.items() if info.priority == priority]


def get_sector_info() -> Dict[str, Dict]:
    """Get sector distribution summary."""
    sector_counts = {}
    sector_stocks = {}
    for symbol, info in STOCK_UNIVERSE.items():
        sector_name = info.sector.value
        if sector_name not in sector_counts:
            sector_counts[sector_name] = 0
            sector_stocks[sector_name] = []
        sector_counts[sector_name] += 1
        sector_stocks[sector_name].append(symbol)
    return {
        "counts": sector_counts,
        "stocks": sector_stocks,
        "total": len(STOCK_UNIVERSE)
    }


def get_correlation_group(symbol: str) -> Optional[str]:
    """Get the correlation group for a symbol."""
    for group_name, members in CORRELATION_GROUPS.items():
        if symbol in members:
            return group_name
    return None


def get_correlated_symbols(symbol: str) -> List[str]:
    """Get symbols that are correlated with the given symbol."""
    for group_name, members in CORRELATION_GROUPS.items():
        if symbol in members:
            return [s for s in members if s != symbol]
    return []


def print_sector_distribution():
    """Print sector distribution table."""
    info = get_sector_info()
    total = info["total"]
    print(f"\n{'='*70}")
    print(f"  STOCK UNIVERSE — SECTOR DISTRIBUTION ({total} stocks)")
    print(f"{'='*70}")
    print(f"  {'Sector':<22} {'Count':<7} {'%':<7} {'Stocks'}")
    print(f"  {'─'*65}")
    for sector_name, count in sorted(info["counts"].items(), key=lambda x: -x[1]):
        pct = count / total * 100
        stocks = ", ".join(info["stocks"][sector_name][:5])
        if len(info["stocks"][sector_name]) > 5:
            stocks += f" +{len(info['stocks'][sector_name]) - 5} more"
        print(f"  {sector_name:<22} {count:<7} {pct:<5.0f}% {stocks}")
    print(f"  {'─'*65}")
    print(f"  {'TOTAL':<22} {total:<7} 100%")
    print(f"{'='*70}")


# ============================================================
#  LEGACY COMPATIBILITY — These lists feed into existing code
# ============================================================

# BUG FIX: Was missing comma between TMPV.NS and WIPRO.NS
#          causing "TMPV.NSWIPRO.NS" concatenation.
NIFTY_50_STOCKS: List[str] = get_active_stocks()

# Next 50 — now maps to priority 3 extended stocks
NIFTY_NEXT_50_STOCKS: List[str] = [
    "DABUR.NS", "PIDILITIND.NS", "HAVELLS.NS", "GODREJCP.NS", "SIEMENS.NS",
    "ABB.NS", "MARICO.NS", "COLPAL.NS", "BIOCON.NS", "LUPIN.NS",
    "TRENT.NS", "ZOMATO.NS", "PAYTM.NS", "DELHIVERY.NS", "NAUKRI.NS",
    "IRCTC.NS", "VOLTAS.NS", "BERGEPAINT.NS", "MUTHOOTFIN.NS", "MPHASIS.NS"
]

# UPDATED: Removed TATAMOTORS.NS (now TMPV.NS), added sector diversity
QUICK_TEST_STOCKS: List[str] = [
    # Financial Services (2)
    "HDFCBANK.NS", "ICICIBANK.NS",
    # Technology (1)
    "TCS.NS",
    # FMCG (1)
    "ITC.NS",
    # Pharma (1)
    "SUNPHARMA.NS",
    # Auto (1 — your new stock)
    "TMPV.NS",
    # Energy (1)
    "RELIANCE.NS",
    # Metals (1)
    "TATASTEEL.NS",
    # Infrastructure (1)
    "LT.NS",
    # Consumer (1)
    "BHARTIARTL.NS",
]

# ============================================================
#  PHASE 1A: BREAK QUALITY FILTER CONFIG
#  Universal candle-quality filter applied across all strategies.
# ============================================================

@dataclass
class BreakQualityConfig:
    """
    Filters out weak candle signals based on body ratio and body/ATR.

    OFF  : No filtering — all signals pass through unchanged.
    SOFT : Signals still count, but weak ones get half credit in consensus.
    HARD : Weak candle signals are discarded entirely.

    Body Ratio = |Close - Open| / (High - Low)
        Measures how much of the candle is directional body vs wicks.
        A ratio of 0.35 means 35% body, 65% wicks — indecisive candle.

    Body / ATR = |Close - Open| / ATR(14)
        Ensures the candle has substance relative to recent volatility.
        A value of 0.10 means the body is only 10% of average range — small.
    """
    mode: str = "SOFT"  # "OFF", "SOFT", "HARD"

    # SOFT thresholds — lenient, catches obviously weak candles
    soft_min_body_ratio: float = 0.35
    soft_min_body_atr: float = 0.10

    # HARD thresholds — strict, only allows high-conviction candles
    hard_min_body_ratio: float = 0.50
    hard_min_body_atr: float = 0.18

    atr_length: int = 14


# Default instance — change mode to "HARD" for stricter filtering
BQ_CONFIG = BreakQualityConfig()


# ============================================================
#  PHASE 1B: REGRESSION SUPERTREND CONFIG
#  Strategy #14 — Adaptive trend following via linear regression.
# ============================================================

@dataclass
class RegressionSuperTrendConfig:
    """
    Parameters for the Regression SuperTrend strategy.

    Uses linear regression + RMSE bands instead of SMA/EMA.
    The key difference: band width adapts to how well price
    is following the regression trend, not just to moving average crossovers.

    reg_length:     How many bars to fit the regression line.
                    Shorter = more responsive, longer = smoother.

    rmse_multiplier: How wide the bands are (in RMSE units).
                     2.0 = ~95% of price action stays within bands
                     during a clean trend. Higher = fewer false flips.
    """
    reg_length: int = 20
    rmse_multiplier: float = 2.0


REG_ST_CONFIG = RegressionSuperTrendConfig()

# ============================================================
#  PHASE 2A: CHOCH/BOS RETEST CONFIG
#  Structural entry framework: Break → Retest → Confirm
# ============================================================

@dataclass
class CHOCHBOSConfig:
    """
    Parameters for the CHOCH/BOS Retest strategy.

    CHOCH (Change of Character): Price breaks a swing in the OPPOSITE
    direction of current structure — signals potential trend reversal.

    BOS (Break of Structure): Price breaks a swing in the SAME
    direction as current structure — signals trend continuation.

    Entry logic (3 layers of confirmation):
    1. BREAK: price closes through a confirmed swing high/low
    2. RETEST: price returns to the neckline (break level)
    3. CONFIRM: price crosses neckline again in break direction
    """
    pivot_left: int = 5           # Bars to the left of a pivot
    pivot_right: int = 5          # Bars to the right (confirms pivot)
    max_wait_bars: int = 40       # Max bars to wait for retest after break
    adx_period: int = 14          # ADX calculation period
    adx_min: float = 20.0         # Minimum ADX to allow signals
    use_adx_filter: bool = True   # Toggle ADX filter on/off


CHOCH_BOS_CONFIG = CHOCHBOSConfig()


# ============================================================
#  PHASE 2B: LIQUIDITY SWEEP CONFIG
#  Institutional flow detection: Sweep → Retest → Breakout
# ============================================================

@dataclass
class LiquiditySweepConfig:
    """
    Parameters for the Liquidity Sweep Retest strategy.

    A liquidity sweep happens when price wicks beyond a swing
    high/low (triggering stop losses) but closes back inside.
    This suggests institutional players grabbed liquidity.

    Entry logic (3 stages):
    1. SWEEP: wick beyond swing level, close back inside
    2. RETEST: price returns to swept level with rejection
    3. BREAKOUT: price breaks the retest candle's extreme
    """
    pivot_left: int = 5           # Swing detection lookback
    pivot_right: int = 5          # Swing detection confirmation
    retest_tol_atr_mult: float = 0.15  # Tolerance for retest (× ATR)
    max_wait_bars: int = 25       # Max bars after sweep for retest+breakout
    atr_length: int = 14          # ATR period for tolerance


LIQ_SWEEP_CONFIG = LiquiditySweepConfig()

# ============================================================
#  PHASE 3A: EXHAUSTION DETECTION CONFIG
#  Consensus modifier — detects when trends are losing steam.
# ============================================================

@dataclass
class ExhaustionConfig:
    """
    Parameters for trend exhaustion detection.

    Exhaustion = a trend is running out of momentum.
    Detected by two simultaneous conditions:
    1. Distance contraction: price is getting CLOSER to its trend
       line (EMA of distance is falling) — the trend is "tiring."
    2. RSI divergence: RSI is moving AGAINST the trend direction
       (falling in uptrend, rising in downtrend) — momentum fading.

    When both fire together, trend-aligned signals get downgraded
    because the move is likely near its end.

    This is NOT a strategy — it's a quality modifier applied to
    existing signals during consensus building.
    """
    enabled: bool = True

    # Trend proxy for distance calculation
    trend_ema_length: int = 50     # EMA used as trend reference line

    # Distance measurement
    distance_ema_length: int = 10  # EMA smoothing of |close - trend_ema|
    contraction_bars: int = 3      # How many bars of contraction required

    # RSI weakness check
    rsi_period: int = 14
    rsi_weakness_bars: int = 3     # How many bars of RSI moving against trend

    # Penalty
    conviction_downgrade: int = 1  # Downgrade consensus level by N steps
    weight_penalty: float = 0.70   # Multiply signal weight by this factor


EXHAUSTION_CONFIG = ExhaustionConfig()


# ============================================================
#  PHASE 3B: TREND / RANGE SUB-REGIME CONFIG
#  Finer classification within Bull/Bear/Sideways.
# ============================================================

@dataclass
class TrendRangeConfig:
    """
    Parameters for TREND vs RANGE classification.

    This is a SECOND layer of regime detection that works alongside
    your existing Bull/Bear/Sideways. It answers a different question:

    - Bull/Bear/Sideways: "What DIRECTION is the market moving?"
    - TREND/RANGE: "What CHARACTER does the market have?"

    A Bull market can be:
    - BULL+TREND: strong uptrend → favor trend/momentum strategies
    - BULL+RANGE: choppy upward drift → favor mean reversion

    A Bear market can be:
    - BEAR+TREND: sustained selloff → favor bear short strategies
    - BEAR+RANGE: choppy decline → favor mean reversion shorts

    Classification is based on three factors:
    1. Flip count: how many trend indicator flips in lookback period
       (more flips = more ranging)
    2. ADX level: low ADX = no directional strength = ranging
    3. EMA slope: flat slope = no trend = ranging
    """
    enabled: bool = True

    # Flip counting (requires trend indicator — uses EMA crossover)
    fast_ema: int = 12
    slow_ema: int = 26
    flip_lookback: int = 60       # Bars to count flips
    flip_threshold: int = 4       # Flips >= this → RANGE candidate

    # ADX
    adx_period: int = 14
    adx_range_max: float = 20.0   # ADX <= this → RANGE candidate

    # EMA slope
    slope_ema_length: int = 50
    slope_lookback: int = 8       # Bars to measure slope over
    slope_min_pct: float = 0.05   # |slope| < this % → flat (RANGE candidate)

    # Weight adjustments when sub-regime is detected
    trend_boost: float = 1.3      # Boost trend/momentum in TREND mode
    trend_penalty: float = 0.7    # Penalize trend/momentum in RANGE mode
    mr_boost: float = 1.3         # Boost mean reversion in RANGE mode
    mr_penalty: float = 0.7       # Penalize mean reversion in TREND mode


TREND_RANGE_CONFIG = TrendRangeConfig()


# ============================================================
#  PHASE 4A: CCI PREP / READY CONFIG
#  Two-stage mean reversion: Extreme → Trigger → Entry
# ============================================================

@dataclass
class CCIPrepConfig:
    """
    Parameters for the CCI PREP/READY strategy.

    This is a two-stage mean reversion strategy:
      Stage 1 (ARM):  CCI reaches an extreme level (±200).
                      This signals capitulation — the move is overdone.
      Stage 2 (FIRE): CCI crosses back through a trigger level (±100).
                      This signals recovery — the snap-back has begun.

    Why two stages?
    - Single CCI crossover (like RSI) fires too often in noisy markets.
    - Requiring the extreme first means we only trade after genuine
      capitulation, not just normal oscillation.
    - The trigger confirms the reversal is actually starting.

    Cooldown prevents re-entry on the same move.
    """
    enabled: bool = True

    cci_length: int = 20         # CCI calculation period
    cci_source: str = "hlc3"    # Price source: (H+L+C)/3

    # Stage 1: ARM — CCI must reach this extreme
    arm_level: int = 200         # CCI must cross ±200 to ARM

    # Stage 2: FIRE — CCI must cross this trigger
    trigger_level: int = 100     # CCI crosses back through ±100 to FIRE

    # Cooldown — prevent repeated signals on same move
    cooldown_bars: int = 12      # Min bars between signals

    # State filter — only fire in the right direction
    use_state_filter: bool = True
    # LONG requires price above trend (bull bias)
    # SHORT requires price below trend (bear bias)
    trend_ema_length: int = 50


CCI_PREP_CONFIG = CCIPrepConfig()


# ============================================================
#  PHASE 4B: ADX QUALITY GATE CONFIG
#  Consensus modifier — adjusts signal confidence by ADX level.
# ============================================================

@dataclass
class ADXQualityGateConfig:
    """
    Parameters for the ADX Quality Gate.

    ADX measures trend STRENGTH (not direction). It answers:
    "Is there a real trend happening, or is the market just noisy?"

    Quality levels:
      ADX < 15:  NO_TRADE — No trend at all. All signals suppressed.
      ADX 15-20: WEAK     — Emerging trend. Signals get penalized.
      ADX 20-25: OK       — Confirmed trend. Signals pass through.
      ADX 25-35: CLEAN    — Strong trend. Signals get boosted.
      ADX > 35:  LATE     — Overextended trend. Trend signals penalized,
                            mean reversion gets boosted.

    This is a PER-STOCK modifier (each stock has its own ADX).
    """
    enabled: bool = True
    adx_period: int = 14

    # Thresholds
    no_trade_max: float = 15.0   # Below this: suppress signals
    weak_max: float = 20.0       # Below this: penalize
    ok_max: float = 25.0         # Below this: neutral
    clean_max: float = 35.0      # Below this: boost trend signals
    late_max: float = 100.0      # Above clean: late (overextended)

    # Weight multipliers per level
    no_trade_mult: float = 0.0   # Suppress entirely
    weak_mult: float = 0.6       # Heavy penalty
    ok_mult: float = 1.0         # Neutral
    clean_mult: float = 1.2      # Boost for trend signals
    late_mult: float = 0.9       # Overextended — penalize trend signals
    caution_mr_mult: float = 1.3 # Boost mean reversion in LATE zone
    caution_trend_mult: float = 0.6  # Penalize trend in LATE zone

    # ADX direction modifiers (applied to trend/momentum signals)
    adx_rising_boost: float = 1.1     # ADX rising → trend strengthening
    adx_falling_penalty: float = 0.9  # ADX falling → trend weakening


ADX_GATE_CONFIG = ADXQualityGateConfig()


# ============================================================
#  PHASE 1: UPDATED STRATEGY QUALITY WEIGHTS
#  Merge with your existing STRATEGY_QUALITY in scanner.py
# ============================================================

# Add these entries to your STRATEGY_QUALITY dict in scanner.py:
PHASE_1_STRATEGY_ADDITIONS = {
    "Regression_SuperTrend_20_2.0": {"weight": 1.2, "category": "trend"},
}

# ============================================================
#  MAIN — Print sector distribution when run directly
# ============================================================

if __name__ == "__main__":
    print_sector_distribution()

    print(f"\n  Core stocks (Priority 1): {len(get_core_stocks())}")
    print(f"  Active stocks (Priority 1+2): {len(get_active_stocks())}")
    print(f"  All stocks: {len(get_all_stocks())}")

    print(f"\n  QUICK_TEST_STOCKS: {QUICK_TEST_STOCKS}")

    # Show correlation groups
    print(f"\n  CORRELATION GROUPS:")
    for group, members in CORRELATION_GROUPS.items():
        print(f"    {group}: {len(members)} stocks")