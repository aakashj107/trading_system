"""
Indian Market Trading System — Expanded Stock Universe
Fixes TATAMOTORS symbol issue and adds 50+ stocks.
"""

# ── Fix: TATAMOTORS was renamed/split on NSE ──
# yfinance may need alternate symbols
SYMBOL_ALIASES = {
    "TATAMOTORS.NS": ["TMPV.NS", "TMCV.NS"],
    "M&M.NS": ["M&M.NS", "MM.NS"],
    "NTPC.NS": ["NTPC.NS"],
    "BAJAJ-AUTO.NS": ["BAJAJ-AUTO.NS", "BAJAJAUTO.NS"],
}

# ── Expanded Universe: 50 stocks across sectors ──
EXPANDED_UNIVERSE = [
    # Banking & Financial
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "MUTHOOTFIN.NS",
    # IT
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "MPHASIS.NS", "LTM.NS",
    # Consumer
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "TATACONSUM.NS", "DABUR.NS", "MARICO.NS", "GODREJCP.NS",
    "COLPAL.NS", "PIDILITIND.NS",
    # Auto
    "MARUTI.NS", "M&M.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS", "TATAMOTORS.NS",
    # Pharma
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    # Industrial & Infra
    "LT.NS", "ULTRACEMCO.NS", "GRASIM.NS", "ASIANPAINT.NS",
    "BERGEPAINT.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS",
    # Energy & Power
    "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "NTPC.NS", "POWERGRID.NS",
    # Telecom & Media
    "BHARTIARTL.NS",
    # Metals & Mining
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS",
    # Retail & New Age
    "TITAN.NS", "TRENT.NS", "ZOMATO.NS",
    # Healthcare
    "APOLLOHOSP.NS",
]

# ── Sector Mapping ──
EXPANDED_SECTOR_MAP = {
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking", "KOTAKBANK.NS": "Banking",
    "AXISBANK.NS": "Banking", "INDUSINDBK.NS": "Banking",
    "BAJFINANCE.NS": "NBFC", "BAJAJFINSV.NS": "NBFC",
    "HDFCLIFE.NS": "Insurance", "SBILIFE.NS": "Insurance",
    "SHRIRAMFIN.NS": "NBFC", "MUTHOOTFIN.NS": "NBFC",
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT",
    "WIPRO.NS": "IT", "TECHM.NS": "IT", "MPHASIS.NS": "IT",
    "LTIM.NS": "IT",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG",
    "NESTLEIND.NS": "FMCG", "BRITANNIA.NS": "FMCG",
    "TATACONSUM.NS": "FMCG", "DABUR.NS": "FMCG",
    "MARICO.NS": "FMCG", "GODREJCP.NS": "FMCG",
    "COLPAL.NS": "FMCG", "PIDILITIND.NS": "Chemicals",
    "MARUTI.NS": "Auto", "M&M.NS": "Auto",
    "EICHERMOT.NS": "Auto", "HEROMOTOCO.NS": "Auto",
    "BAJAJ-AUTO.NS": "Auto", "TATAMOTORS.NS": "Auto",
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma",
    "CIPLA.NS": "Pharma", "DIVISLAB.NS": "Pharma",
    "LT.NS": "Construction", "ULTRACEMCO.NS": "Cement",
    "GRASIM.NS": "Diversified", "ASIANPAINT.NS": "Paint",
    "BERGEPAINT.NS": "Paint", "SIEMENS.NS": "Industrial",
    "ABB.NS": "Industrial", "HAVELLS.NS": "Industrial",
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy",
    "BPCL.NS": "Energy", "NTPC.NS": "Power",
    "POWERGRID.NS": "Power",
    "BHARTIARTL.NS": "Telecom",
    "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals",
    "HINDALCO.NS": "Metals", "COALINDIA.NS": "Mining",
    "TITAN.NS": "Consumer", "TRENT.NS": "Retail",
    "ZOMATO.NS": "Tech", "APOLLOHOSP.NS": "Healthcare",
}