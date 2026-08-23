"""
Indian Market Trading System — Fundamental Analysis
Revenue quality, margins, cash flow, debt, ROE, accounting red flags.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class HealthRating(Enum):
    STRONG = "STRONG"
    HEALTHY = "HEALTHY"
    CAUTIOUS = "CAUTIOUS"
    WEAK = "WEAK"
    DANGEROUS = "DANGEROUS"


@dataclass
class RedFlag:
    category: str
    severity: str
    description: str
    evidence: str


@dataclass
class FundamentalReport:
    symbol: str
    company_name: str
    sector: str
    health_rating: HealthRating
    overall_score: float
    revenue_quality_score: float = 0
    revenue_quality_notes: List[str] = field(default_factory=list)
    margin_score: float = 0
    margin_notes: List[str] = field(default_factory=list)
    cashflow_score: float = 0
    cashflow_notes: List[str] = field(default_factory=list)
    debt_score: float = 0
    debt_notes: List[str] = field(default_factory=list)
    roe_current: Optional[float] = None
    roe_3yr_avg: Optional[float] = None
    roe_sustainable: Optional[bool] = None
    red_flags: List[RedFlag] = field(default_factory=list)
    financials: Dict = field(default_factory=dict)


class FundamentalAnalyzer:
    def analyze(self, symbol: str) -> FundamentalReport:
        print(f"\n  Analyzing fundamentals for {symbol}...")
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        report = FundamentalReport(
            symbol=symbol,
            company_name=info.get("longName", symbol.replace(".NS", "")),
            sector=info.get("sector", "Unknown"),
            health_rating=HealthRating.HEALTHY, overall_score=0,
            revenue_quality_score=0
        )
        try:
            income = ticker.financials
            balance = ticker.balance_sheet
            cashflow = ticker.cashflow
        except Exception as e:
            print(f"  [ERROR] {e}")
            return report

        report.financials = {
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "return_on_equity": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "operating_cashflow": info.get("operatingCashflow"),
            "free_cashflow": info.get("freeCashflow"),
            "total_debt": info.get("totalDebt", 0),
            "total_cash": info.get("totalCash", 0),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margins": info.get("profitMargins"),
            "operating_margins": info.get("operatingMargins"),
            "gross_margins": info.get("grossMargins"),
            "earnings_growth": info.get("earningsGrowth"),
        }

        r_score, r_notes = self._revenue_quality(income, info)
        report.revenue_quality_score = r_score
        report.revenue_quality_notes = r_notes

        m_score, m_notes = self._margins(income, info)
        report.margin_score = m_score
        report.margin_notes = m_notes

        cf_score, cf_notes, cf_flags = self._cashflow(income, cashflow, info)
        report.cashflow_score = cf_score
        report.cashflow_notes = cf_notes
        report.red_flags.extend(cf_flags)

        d_score, d_notes, d_flags = self._debt(balance, info)
        report.debt_score = d_score
        report.debt_notes = d_notes
        report.red_flags.extend(d_flags)

        self._roe(balance, income, info, report)
        report.red_flags.extend(self._accounting_flags(income, balance, info))

        report.overall_score = (report.revenue_quality_score + report.margin_score +
                                report.cashflow_score + report.debt_score)
        if report.overall_score >= 80:
            report.health_rating = HealthRating.STRONG
        elif report.overall_score >= 60:
            report.health_rating = HealthRating.HEALTHY
        elif report.overall_score >= 40:
            report.health_rating = HealthRating.CAUTIOUS
        elif report.overall_score >= 20:
            report.health_rating = HealthRating.WEAK
        else:
            report.health_rating = HealthRating.DANGEROUS

        high_flags = [f for f in report.red_flags if f.severity == "HIGH"]
        if len(high_flags) >= 2:
            if report.health_rating == HealthRating.STRONG:
                report.health_rating = HealthRating.HEALTHY
            elif report.health_rating == HealthRating.HEALTHY:
                report.health_rating = HealthRating.CAUTIOUS
        return report

    def _revenue_quality(self, income, info):
        score, notes = 0, []
        rg = info.get("revenueGrowth")
        if rg is not None:
            if rg > 0.15:
                score += 8; notes.append(f"Strong revenue growth: {rg*100:.1f}%")
            elif rg > 0.05:
                score += 5; notes.append(f"Moderate growth: {rg*100:.1f}%")
            elif rg > 0:
                score += 2; notes.append(f"Low growth: {rg*100:.1f}%")
            else:
                notes.append(f"NEGATIVE growth: {rg*100:.1f}%")

        eg = info.get("earningsGrowth")
        if eg is not None:
            if eg > 0 and rg and eg > rg:
                score += 5; notes.append("Earnings growing faster than revenue")
            elif eg > 0:
                score += 3
            else:
                notes.append(f"Earnings declining: {eg*100:.1f}%")

        if income is not None and not income.empty:
            for label in ["Total Revenue", "Revenue"]:
                if label in income.index:
                    vals = income.loc[label].dropna().values
                    if len(vals) >= 3:
                        gs = []
                        for i in range(len(vals)-1):
                            if vals[i+1] != 0:
                                gs.append((vals[i]-vals[i+1])/abs(vals[i+1]))
                        if gs:
                            pos = sum(1 for g in gs if g > 0)
                            if pos/len(gs) >= 0.75:
                                score += 7; notes.append(f"Consistent revenue: {pos}/{len(gs)} years")
                            elif pos/len(gs) >= 0.5:
                                score += 4
                            else:
                                score += 1; notes.append("Lumpy revenue")
                    break
        return min(score, 25), notes

    def _margins(self, income, info):
        score, notes = 0, []
        om = info.get("operatingMargins")
        pm = info.get("profitMargins")
        if om is not None:
            if om > 0.25: score += 6; notes.append(f"Excellent OP margin: {om*100:.1f}%")
            elif om > 0.15: score += 4; notes.append(f"Good OP margin: {om*100:.1f}%")
            elif om > 0.08: score += 2; notes.append(f"Average OP margin: {om*100:.1f}%")
            elif om > 0: score += 1
            else: notes.append(f"NEGATIVE OP margin: {om*100:.1f}%")
        if pm is not None:
            if pm > 0.20: score += 5; notes.append(f"High net margin: {pm*100:.1f}%")
            elif pm > 0.10: score += 3; notes.append(f"Healthy net margin: {pm*100:.1f}%")
            elif pm > 0.05: score += 2
            elif pm > 0: score += 1
            else: notes.append(f"NEGATIVE net margin: {pm*100:.1f}%")
        return min(score, 25), notes

    def _cashflow(self, income, cashflow, info):
        score, notes, flags = 0, [], []
        ocf = info.get("operatingCashflow", 0) or 0
        fcf = info.get("freeCashflow", 0) or 0
        rev = info.get("totalRevenue", 1) or 1
        if ocf > 0:
            m = ocf/rev
            if m > 0.15: score += 7; notes.append(f"Strong OCF margin: {m*100:.1f}%")
            elif m > 0.08: score += 5
            elif m > 0: score += 3
        elif ocf < 0:
            flags.append(RedFlag("CASH_FLOW","HIGH","Negative OCF",f"₹{ocf:,.0f}"))
        if fcf > 0:
            score += 6; notes.append(f"Positive FCF: ₹{fcf:,.0f}")
        elif fcf < 0:
            flags.append(RedFlag("CASH_FLOW","MEDIUM","Negative FCF",f"₹{fcf:,.0f}"))

        if cashflow is not None and not cashflow.empty and income is not None:
            try:
                cf_row = None
                for l in ["Operating Cash Flow","Total Cash From Operating Activities"]:
                    if l in cashflow.index: cf_row = cashflow.loc[l]; break
                ni_row = None
                for l in ["Net Income","Net Income Common Stockholders"]:
                    if l in income.index: ni_row = income.loc[l]; break
                if cf_row is not None and ni_row is not None:
                    cv, nv = cf_row.dropna().values, ni_row.dropna().values
                    ml = min(len(cv),len(nv))
                    ratios = [cv[i]/nv[i] for i in range(ml) if nv[i] and nv[i]>0]
                    if ratios:
                        avg = np.mean(ratios)
                        if avg >= 1.2: score += 7; notes.append(f"Cash conversion: {avg:.2f}x (excellent)")
                        elif avg >= 0.8: score += 5; notes.append(f"Cash conversion: {avg:.2f}x (adequate)")
                        elif avg >= 0.5: score += 2; notes.append(f"Cash conversion: {avg:.2f}x (weak)")
                        else:
                            notes.append(f"Cash conversion: {avg:.2f}x (dangerous)")
                            flags.append(RedFlag("CASH_FLOW","HIGH","NI not backed by OCF",f"Ratio:{avg:.2f}x"))
            except Exception: pass
        return min(score, 25), notes, flags

    def _debt(self, balance, info):
        score, notes, flags = 0, [], []
        dte = info.get("debtToEquity")
        cr = info.get("currentRatio")
        td = info.get("totalDebt", 0) or 0
        tc = info.get("totalCash", 0) or 0
        if dte is not None:
            d = dte/100
            if d < 0.3: score += 8; notes.append(f"Very low D/E: {d:.2f}")
            elif d < 0.7: score += 6; notes.append(f"Moderate D/E: {d:.2f}")
            elif d < 1.5: score += 3; notes.append(f"High D/E: {d:.2f}")
            elif d < 3: score += 1
            else:
                flags.append(RedFlag("DEBT","HIGH","Dangerous leverage",f"D/E:{d:.2f}"))
        if td > 0:
            nd = td - tc
            if nd < 0: score += 5; notes.append(f"Net CASH positive: ₹{tc-td:,.0f}")
            else: score += 2
        else:
            score += 5; notes.append("DEBT FREE")
        if cr is not None:
            if cr > 2: score += 5; notes.append(f"Excellent liquidity: CR={cr:.2f}")
            elif cr > 1.2: score += 4; notes.append(f"Adequate liquidity: CR={cr:.2f}")
            elif cr > 0.8: score += 2
            else: flags.append(RedFlag("LIQUIDITY","HIGH","Poor liquidity",f"CR:{cr:.2f}"))
        return min(score, 25), notes, flags

    def _roe(self, balance, income, info, report):
        roe = info.get("returnOnEquity")
        report.roe_current = roe
        if roe is None: return
        leverage = (info.get("totalAssets",1) or 1)/(info.get("totalStockholderEquity",1) or 1)
        if leverage > 3:
            report.debt_notes.append(f"ROE inflated by leverage: {leverage:.2f}x")
            report.roe_sustainable = False
        else:
            report.roe_sustainable = True

    def _accounting_flags(self, income, balance, info):
        flags = []
        if balance is not None and not balance.empty:
            try:
                for l in ["Retained Earnings","Retained Earnings Total Equity"]:
                    if l in balance.index:
                        re = balance.loc[l].dropna()
                        if len(re) > 0 and re.iloc[0] < 0:
                            flags.append(RedFlag("ACCOUNTING","HIGH",
                                "NEGATIVE retained earnings",f"₹{re.iloc[0]:,.0f}"))
                        break
            except Exception: pass
        return flags

    def print_report(self, report):
        icons = {"STRONG":"★★★★★","HEALTHY":"★★★★☆","CAUTIOUS":"★★★☆☆",
                 "WEAK":"★★☆☆☆","DANGEROUS":"★☆☆☆☆"}
        print(f"\n{'='*60}")
        print(f"  FUNDAMENTAL ANALYSIS — {report.company_name} ({report.symbol})")
        print(f"{'='*60}")
        print(f"  Rating: {icons[report.health_rating.value]} {report.health_rating.value}")
        print(f"  Score:  {report.overall_score:.0f}/100")
        for name, s in [("Revenue",report.revenue_quality_score),("Margins",report.margin_score),
                         ("CashFlow",report.cashflow_score),("Debt",report.debt_score)]:
            bar = "█"*int(s/25*10)+"░"*(10-int(s/25*10))
            print(f"  {name:<12} [{bar}] {s:.0f}/25")
        for title, notes in [("REVENUE",report.revenue_quality_notes),("MARGINS",report.margin_notes),
                              ("CASH FLOW",report.cashflow_notes),("DEBT",report.debt_notes)]:
            if notes:
                print(f"\n  {title}:")
                for n in notes: print(f"    • {n}")
        if report.red_flags:
            print(f"\n  RED FLAGS ({len(report.red_flags)}):")
            for f in report.red_flags:
                print(f"    🔴 [{f.severity}] {f.description} — {f.evidence}")
        else:
            print(f"\n  ✅ No red flags")
        print(f"{'='*60}")
