"""
Indian Market Trading System — Valuation Reality Check
P/E, EV/EBITDA, sector peers, growth premium.
"""

import yfinance as yf
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class ValuationVerdict(Enum):
    UNDERVALUED = "UNDERVALUED"
    FAIRLY_VALUED = "FAIRLY_VALUED"
    SLIGHTLY_EXPENSIVE = "SLIGHTLY_EXPENSIVE"
    EXPENSIVE = "EXPENSIVE"
    EXTREMELY_EXPENSIVE = "EXTREMELY_EXPENSIVE"


@dataclass
class PeerComparison:
    symbol: str; name: str
    pe_ratio: Optional[float] = None; pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None; roe: Optional[float] = None


@dataclass
class ValuationReport:
    symbol: str; company_name: str; sector: str; current_price: float
    verdict: ValuationVerdict = ValuationVerdict.FAIRLY_VALUED
    overall_score: float = 50
    current_pe: Optional[float] = None; current_pb: Optional[float] = None
    current_ev_ebitda: Optional[float] = None; peg_ratio: Optional[float] = None
    sector_avg_pe: Optional[float] = None; pe_premium_pct: Optional[float] = None
    pe_vs_own_history: str = ""; peer_rank: str = ""
    pe_fair_value: Optional[float] = None; pb_fair_value: Optional[float] = None
    growth_premium_justified: Optional[bool] = None
    growth_premium_notes: List[str] = field(default_factory=list)
    peers: List[PeerComparison] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


SECTOR_PEERS = {
    "Banking": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS"],
    "Technology": ["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS"],
    "Consumer Cyclical": ["MARUTI.NS","TATAMOTORS.NS","M&M.NS","EICHERMOT.NS"],
    "Consumer Defensive": ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS"],
    "Healthcare": ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS"],
    "Energy": ["RELIANCE.NS","ONGC.NS","BPCL.NS"],
    "Basic Materials": ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS"],
    "Financial Services": ["BAJFINANCE.NS","BAJAJFINSV.NS","HDFCLIFE.NS"],
    "Industrials": ["LT.NS","ULTRACEMCO.NS","GRASIM.NS"],
    "Communication Services": ["BHARTIARTL.NS"],
}


class ValuationAnalyzer:
    def analyze(self, symbol, historical_pe=None):
        print(f"\n  Analyzing valuation for {symbol}...")
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        report = ValuationReport(
            symbol=symbol,
            company_name=info.get("longName", symbol.replace(".NS","")),
            sector=info.get("sector","Unknown"),
            current_price=info.get("currentPrice",0) or info.get("regularMarketPrice",0),
        )
        report.current_pe = info.get("trailingPE")
        report.current_pb = info.get("priceToBook")
        ev = info.get("enterpriseValue"); ebitda = info.get("ebitda")
        if ev and ebitda and ebitda > 0: report.current_ev_ebitda = ev/ebitda
        report.peg_ratio = info.get("pegRatio")

        self._peer_compare(symbol, info, report)
        self._growth_check(info, report)
        self._calc_score(report)

        if report.overall_score >= 75: report.verdict = ValuationVerdict.UNDERVALUED
        elif report.overall_score >= 55: report.verdict = ValuationVerdict.FAIRLY_VALUED
        elif report.overall_score >= 40: report.verdict = ValuationVerdict.SLIGHTLY_EXPENSIVE
        elif report.overall_score >= 25: report.verdict = ValuationVerdict.EXPENSIVE
        else: report.verdict = ValuationVerdict.EXTREMELY_EXPENSIVE
        return report

    def _peer_compare(self, symbol, info, report):
        peers_syms = SECTOR_PEERS.get(report.sector, [])
        peers_syms = [p for p in peers_syms if p != symbol][:5]
        if not peers_syms: return
        peers, pes = [], []
        for ps in peers_syms:
            try:
                pi = yf.Ticker(ps).info or {}
                pe = pi.get("trailingPE")
                pb = pi.get("priceToBook")
                evv = pi.get("enterpriseValue"); eb = pi.get("ebitda")
                ev_e = evv/eb if evv and eb and eb>0 else None
                peers.append(PeerComparison(ps, pi.get("shortName",ps), pe, pb, ev_e,
                                            pi.get("returnOnEquity")))
                if pe and pe > 0: pes.append(pe)
            except: continue
        report.peers = peers
        if pes:
            report.sector_avg_pe = np.median(pes)
            if report.current_pe and report.sector_avg_pe:
                report.pe_premium_pct = (report.current_pe - report.sector_avg_pe)/report.sector_avg_pe*100
                if report.pe_premium_pct > 30:
                    report.notes.append(f"{report.pe_premium_pct:.0f}% PREMIUM to sector")
                elif report.pe_premium_pct < -30:
                    report.notes.append(f"{abs(report.pe_premium_pct):.0f}% DISCOUNT to sector")
            all_pes = [p for p in pes] + [report.current_pe] if report.current_pe else pes
            all_pes_sorted = sorted(all_pes)
            if report.current_pe in all_pes_sorted:
                rank = all_pes_sorted.index(report.current_pe)+1
                report.peer_rank = f"Rank {rank}/{len(all_pes_sorted)}"

    def _growth_check(self, info, report):
        pe = report.current_pe
        eg = info.get("earningsGrowth")
        rg = info.get("revenueGrowth")
        if report.peg_ratio:
            peg = report.peg_ratio
            if peg < 1:
                report.growth_premium_justified = True
                report.growth_premium_notes.append(f"PEG={peg:.2f} — growth is cheap")
            elif peg < 2: report.growth_premium_notes.append(f"PEG={peg:.2f} — moderate")
            else:
                report.growth_premium_justified = False
                report.growth_premium_notes.append(f"PEG={peg:.2f} — expensive vs growth")
        elif pe and eg and eg > 0:
            manual_peg = pe/(eg*100)
            report.peg_ratio = manual_peg

    def _calc_score(self, report):
        s = 50
        if report.pe_premium_pct is not None:
            if report.pe_premium_pct < -30: s += 20
            elif report.pe_premium_pct < -10: s += 10
            elif report.pe_premium_pct < 10: pass
            elif report.pe_premium_pct < 30: s -= 10
            else: s -= 20
        if report.peg_ratio is not None:
            if report.peg_ratio < 1: s += 10
            elif report.peg_ratio > 2.5: s -= 15
        if report.growth_premium_justified is True: s += 5
        elif report.growth_premium_justified is False: s -= 10
        report.overall_score = max(0, min(100, s))

    def print_report(self, report):
        icons = {"UNDERVALUED":"💰","FAIRLY_VALUED":"⚖️","SLIGHTLY_EXPENSIVE":"📈",
                 "EXPENSIVE":"🔴","EXTREMELY_EXPENSIVE":"🚨"}
        print(f"\n{'='*60}")
        print(f"  VALUATION — {report.company_name}")
        print(f"{'='*60}")
        print(f"  Price: ₹{report.current_price:,.2f}")
        print(f"  Verdict: {icons[report.verdict.value]} {report.verdict.value}")
        print(f"  Score: {report.overall_score:.0f}/100")
        print(f"  PE: {report.current_pe} | PB: {report.current_pb} | "
              f"EV/EBITDA: {report.current_ev_ebitda} | PEG: {report.peg_ratio}")
        if report.sector_avg_pe:
            print(f"  Sector Median PE: {report.sector_avg_pe:.1f}")
        if report.pe_premium_pct:
            d = "PREMIUM" if report.pe_premium_pct > 0 else "DISCOUNT"
            print(f"  vs Sector: {abs(report.pe_premium_pct):.0f}% {d}")
        if report.peer_rank: print(f"  Peer Rank: {report.peer_rank}")
        for n in report.notes: print(f"    • {n}")
        for n in report.growth_premium_notes: print(f"    • {n}")
        print(f"{'='*60}")
