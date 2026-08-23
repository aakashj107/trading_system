"""
Indian Market Trading System — Final Verdict Generator
BUY/HOLD/AVOID, conviction score, interesting price, one-line summary.
"""

import yfinance as yf
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from fundamentals import FundamentalAnalyzer, HealthRating
from management_analyzer import ManagementAnalyzer
from valuation import ValuationAnalyzer, ValuationVerdict
from data_fetcher import YFinanceDataFetcher


class Action(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID"
    STRONG_AVOID = "STRONG_AVOID"


@dataclass
class RiskFactor:
    category: str; risk: str; impact: str; probability: str


@dataclass
class FinalVerdict:
    symbol: str; company_name: str; sector: str; date: str
    action: Action; conviction_score: float; one_line_summary: str
    current_price: float; interesting_price: float
    stop_loss_price: float; target_price: float
    fundamental_score: float; management_score: float
    valuation_score: float; technical_score: float; risk_score: float
    risk_factors: List[RiskFactor] = field(default_factory=list)
    fundamental_summary: str = ""; management_summary: str = ""
    valuation_summary: str = ""; technical_summary: str = ""
    key_bull_points: List[str] = field(default_factory=list)
    key_bear_points: List[str] = field(default_factory=list)


class VerdictEngine:
    WEIGHTS = {"fundamental":0.25, "management":0.15, "valuation":0.25,
               "technical":0.20, "risk":0.15}

    def __init__(self):
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.management_analyzer = ManagementAnalyzer()
        self.valuation_analyzer = ValuationAnalyzer()
        self.fetcher = YFinanceDataFetcher()

    def generate_verdict(self, symbol):
        print(f"\n{'='*60}")
        print(f"  GENERATING VERDICT FOR {symbol}")
        print(f"{'='*60}")
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        company_name = info.get("longName", symbol.replace(".NS",""))
        sector = info.get("sector","Unknown")
        current_price = info.get("currentPrice",0) or info.get("regularMarketPrice",0)

        print(f"\n  [1/5] Fundamentals...")
        fr = self.fundamental_analyzer.analyze(symbol)
        fs = fr.overall_score / 10

        print(f"\n  [2/5] Management...")
        mr = self.management_analyzer.analyze(symbol)
        ms = mr.management_score / 10

        print(f"\n  [3/5] Valuation...")
        vr = self.valuation_analyzer.analyze(symbol)
        vs = vr.overall_score / 10

        print(f"\n  [4/5] Technicals...")
        ts, tech_sum = self._technicals(symbol)

        print(f"\n  [5/5] Risks...")
        rs, risks = self._risks(symbol, info, fr, vr)

        w = (fs*self.WEIGHTS["fundamental"] + ms*self.WEIGHTS["management"] +
             vs*self.WEIGHTS["valuation"] + ts*self.WEIGHTS["technical"] +
             rs*self.WEIGHTS["risk"])
        conviction = max(1, min(10, round(w, 1)))

        action = self._action(conviction, fs, ms, vs)
        interesting = self._interesting_price(current_price, vr)
        sl = round(current_price * 0.92, 2)
        target = round(current_price * 1.20, 2)

        bull = self._bull(fr, mr, vr)
        bear = self._bear(fr, risks)

        one_liner = self._one_liner(company_name, action, conviction,
                                     current_price, interesting)

        return FinalVerdict(
            symbol=symbol, company_name=company_name, sector=sector,
            date=datetime.now().strftime("%Y-%m-%d"),
            action=action, conviction_score=conviction,
            one_line_summary=one_liner,
            current_price=current_price, interesting_price=interesting,
            stop_loss_price=sl, target_price=target,
            fundamental_score=round(fs,1), management_score=round(ms,1),
            valuation_score=round(vs,1), technical_score=round(ts,1),
            risk_score=round(rs,1),
            risk_factors=risks, fundamental_summary=fr.health_rating.value,
            management_summary=mr.summary, valuation_summary=vr.verdict.value,
            technical_summary=tech_sum, key_bull_points=bull, key_bear_points=bear
        )

    def _technicals(self, symbol):
        data = self.fetcher.get_historical_data(symbol, period="1y")
        if data.empty: return 5.0, "No data"
        close = data["Close"]
        s, notes = 5.0, []
        sma50 = close.rolling(50).mean().iloc[-1]
        cur = close.iloc[-1]
        if len(close) >= 200:
            sma200 = close.rolling(200).mean().iloc[-1]
            if cur > sma200: s += 1.5; notes.append("Above 200 SMA")
            else: s -= 1.5; notes.append("Below 200 SMA")
            if sma50 > sma200: s += 1.0
            else: s -= 1.0
        if cur > sma50: s += 0.5
        else: s -= 0.5
        return max(0,min(10,s)), " | ".join(notes[:3])

    def _risks(self, symbol, info, fr, vr):
        s, risks = 7.0, []
        risks.append(RiskFactor("SECTOR",
            f"General {info.get('sector','market')} risk — cyclicality, regulation",
            "MEDIUM","MEDIUM"))
        if fr.red_flags:
            hf = [f for f in fr.red_flags if f.severity == "HIGH"]
            if hf: s -= 2.0; risks.append(RiskFactor("COMPANY", hf[0].description, "HIGH","MEDIUM"))
        dte = info.get("debtToEquity")
        if dte and dte > 200:
            s -= 1.5; risks.append(RiskFactor("COMPANY",f"High leverage D/E={dte/100:.1f}","HIGH","MEDIUM"))
        if vr.verdict in [ValuationVerdict.EXPENSIVE, ValuationVerdict.EXTREMELY_EXPENSIVE]:
            s -= 1.5; risks.append(RiskFactor("COMPANY","Expensive valuation — limited margin of safety","HIGH","MEDIUM"))
        risks.append(RiskFactor("MACRO","FII outflows, INR depreciation, RBI policy, geopolitical","HIGH","LOW"))
        return max(0,min(10,s)), risks[:3]

    def _action(self, c, fs, ms, vs):
        if fs < 3: return Action.STRONG_AVOID
        if ms < 2: return Action.AVOID
        if vs < 2: return Action.AVOID
        if c >= 7.5: return Action.STRONG_BUY
        if c >= 6.0: return Action.BUY
        if c >= 4.0: return Action.HOLD
        if c >= 2.5: return Action.AVOID
        return Action.STRONG_AVOID

    def _interesting_price(self, cur, vr):
        prices = [cur * 0.85]
        if vr.pe_fair_value and vr.pe_fair_value > 0: prices.append(vr.pe_fair_value * 0.90)
        if vr.verdict in [ValuationVerdict.EXPENSIVE, ValuationVerdict.EXTREMELY_EXPENSIVE]:
            prices.append(cur * 0.75)
        return round(np.median(prices), 2)

    def _bull(self, fr, mr, vr):
        pts = []
        if fr.health_rating in [HealthRating.STRONG, HealthRating.HEALTHY]:
            pts.append(f"Fundamentals: {fr.health_rating.value} ({fr.overall_score:.0f}/100)")
        if fr.roe_current and fr.roe_current > 0.15: pts.append(f"High ROE: {fr.roe_current*100:.1f}%")
        if fr.cashflow_score > 15: pts.append("Strong cash flow")
        if fr.debt_score > 15: pts.append("Conservative debt")
        if vr.verdict in [ValuationVerdict.UNDERVALUED, ValuationVerdict.FAIRLY_VALUED]:
            pts.append(f"Valuation: {vr.verdict.value}")
        if mr.overall_rating == "STRONG": pts.append("Strong management")
        if not fr.red_flags: pts.append("No red flags")
        return pts[:5]

    def _bear(self, fr, risks):
        pts = []
        for f in fr.red_flags:
            if f.severity == "HIGH": pts.append(f"RED FLAG: {f.description}")
        if risks: pts.append(f"KEY RISK: {risks[0].risk}")
        return pts[:5]

    def _one_liner(self, name, action, c, cur, ip):
        at = {Action.STRONG_BUY:"Strong conviction buy", Action.BUY:"Buy with discipline",
              Action.HOLD:"Hold if own", Action.AVOID:"Avoid — risk/reward unfavorable",
              Action.STRONG_AVOID:"Stay far away"}
        line = f"{name}: {at[action]}, conviction {c}/10 at ₹{cur:,.0f}"
        if action in [Action.STRONG_BUY, Action.BUY]:
            line += f" — interesting below ₹{ip:,.0f}"
        elif action in [Action.AVOID, Action.STRONG_AVOID]:
            line += f" — wait for ₹{ip:,.0f}"
        return line

    def print_verdict(self, v):
        ai = {Action.STRONG_BUY:"🟢🟢 STRONG BUY", Action.BUY:"🟢 BUY",
              Action.HOLD:"🟡 HOLD", Action.AVOID:"🔴 AVOID",
              Action.STRONG_AVOID:"🔴🔴 STRONG AVOID"}
        print(f"\n{'█'*60}")
        print(f"  FINAL VERDICT")
        print(f"{'█'*60}")
        print(f"  {v.company_name} ({v.symbol}) — {v.sector}")
        print(f"  Date: {v.date}")
        print(f"\n  ┌{'─'*50}┐")
        print(f"  │  {ai[v.action]:^46}  │")
        print(f"  │  Conviction: {v.conviction_score}/10{'':<33}│")
        print(f"  └{'─'*50}┘")
        print(f"\n  💬 {v.one_line_summary}")
        print(f"\n  Current:    ₹{v.current_price:>10,.2f}")
        print(f"  Interesting:₹{v.interesting_price:>10,.2f}")
        print(f"  Stop Loss:  ₹{v.stop_loss_price:>10,.2f}")
        print(f"  Target:     ₹{v.target_price:>10,.2f}")
        print(f"\n  SCORES:")
        for name, sc, summ in [
            ("Fundamentals",v.fundamental_score,v.fundamental_summary),
            ("Management",v.management_score,v.management_summary),
            ("Valuation",v.valuation_score,v.valuation_summary),
            ("Technicals",v.technical_score,v.technical_summary),
            ("Risk",v.risk_score,f"{len(v.risk_factors)} risks")]:
            bar = "█"*int(sc)+"░"*(10-int(sc))
            print(f"    {name:<14} [{bar}] {sc}/10  {summ}")
        if v.key_bull_points:
            print(f"\n  🐂 BULL:")
            for p in v.key_bull_points: print(f"    + {p}")
        if v.key_bear_points:
            print(f"\n  🐻 BEAR:")
            for p in v.key_bear_points: print(f"    - {p}")
        if v.risk_factors:
            print(f"\n  ⚠️ TOP 3 RISKS:")
            for i,r in enumerate(v.risk_factors,1):
                print(f"    {i}. [{r.category}] {r.risk} ({r.impact}/{r.probability})")
        print(f"{'█'*60}")
