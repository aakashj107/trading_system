"""
Indian Market Trading System — Management DNA Analyzer
Promoter holding, pledge, concall tone, FII/DII.
"""

import yfinance as yf
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PromoterAlert:
    alert_type: str
    severity: str
    description: str
    current_value: str
    previous_value: str


@dataclass
class ManagementReport:
    symbol: str
    company_name: str
    promoter_holding_pct: Optional[float] = None
    promoter_trend: str = ""
    promoter_alerts: List[PromoterAlert] = field(default_factory=list)
    fii_pct: Optional[float] = None
    dii_pct: Optional[float] = None
    management_score: float = 0
    confidence_level: str = ""
    quality_notes: List[str] = field(default_factory=list)
    concall_sentiment: str = ""
    concall_red_flags: List[str] = field(default_factory=list)
    overall_rating: str = ""
    summary: str = ""


class ManagementAnalyzer:
    def analyze(self, symbol, concall_transcript=None, previous_promoter_pct=None):
        print(f"\n  Analyzing management for {symbol}...")
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        report = ManagementReport(
            symbol=symbol,
            company_name=info.get("longName", symbol.replace(".NS", ""))
        )
        held_insiders = info.get("heldPercentInsiders", 0) or 0
        held_institutions = info.get("heldPercentInstitutions", 0) or 0
        if held_insiders > 0:
            report.promoter_holding_pct = held_insiders * 100
        if held_institutions > 0:
            report.fii_pct = held_institutions * 60
            report.dii_pct = held_institutions * 40

        if report.promoter_holding_pct:
            pct = report.promoter_holding_pct
            if pct >= 60: report.quality_notes.append(f"Very high promoter: {pct:.1f}%")
            elif pct >= 50: report.quality_notes.append(f"Healthy promoter: {pct:.1f}%")
            elif pct >= 35: report.quality_notes.append(f"Moderate promoter: {pct:.1f}%")
            elif pct >= 25: report.quality_notes.append(f"Low promoter: {pct:.1f}%")
            else:
                report.quality_notes.append(f"VERY LOW promoter: {pct:.1f}% — RED FLAG")
                report.promoter_alerts.append(PromoterAlert(
                    "LOW_STAKE","HIGH","Critically low promoter holding",
                    f"{pct:.1f}%","N/A"))

        if previous_promoter_pct and report.promoter_holding_pct:
            change = report.promoter_holding_pct - previous_promoter_pct
            if change < -2:
                report.promoter_trend = "DECLINING"
                report.quality_notes.append(f"Stake REDUCED: {previous_promoter_pct:.1f}% → {report.promoter_holding_pct:.1f}%")
                report.promoter_alerts.append(PromoterAlert(
                    "STAKE_REDUCTION","HIGH",f"Stake dropped {abs(change):.1f}pp",
                    f"{report.promoter_holding_pct:.1f}%",f"{previous_promoter_pct:.1f}%"))
            elif change > 1:
                report.promoter_trend = "INCREASING"
                report.quality_notes.append(f"Stake INCREASING: positive signal")
            else:
                report.promoter_trend = "STABLE"

        if concall_transcript:
            self._analyze_concall(concall_transcript, report)

        self._calc_score(report)
        report.overall_rating = "STRONG" if report.management_score >= 70 else (
            "AVERAGE" if report.management_score >= 45 else "CONCERNING")
        report.summary = self._summary(report)
        return report

    def _analyze_concall(self, text, report):
        t = text.lower()
        conf = ["confident","optimistic","robust","excellent","record","outperform",
                "strong demand","growth trajectory","expanding","bullish","accelerating"]
        defs = ["challenging","headwinds","uncertain","cautious","pressured","sluggish",
                "concerned","competitive pressure","margin pressure","decline","restructuring"]
        reds = ["accounting change","restatement","related party","regulatory investigation",
                "impairment","goodwill write-down","going concern","qualified opinion",
                "material weakness","delayed filing"]
        cc = sum(t.count(w) for w in conf)
        dc = sum(t.count(w) for w in defs)
        if cc > dc * 1.5: report.concall_sentiment = "CONFIDENT"
        elif dc > cc * 1.5: report.concall_sentiment = "DEFENSIVE"
        else: report.concall_sentiment = "NEUTRAL"
        for p in reds:
            if p in t: report.concall_red_flags.append(f"Mentions: '{p}'")

    def _calc_score(self, report):
        s = 50
        if report.promoter_holding_pct:
            p = report.promoter_holding_pct
            if p >= 60: s += 15
            elif p >= 50: s += 10
            elif p >= 35: s += 5
            elif p < 25: s -= 15
        if report.promoter_trend == "INCREASING": s += 10
        elif report.promoter_trend == "DECLINING": s -= 15
        s -= sum(10 for a in report.promoter_alerts if a.severity == "HIGH")
        if report.concall_sentiment == "CONFIDENT": s += 10
        elif report.concall_sentiment == "DEFENSIVE": s -= 10
        s -= len(report.concall_red_flags) * 5
        report.management_score = max(0, min(100, s))

    def _summary(self, report):
        parts = []
        if report.promoter_holding_pct: parts.append(f"Promoter {report.promoter_holding_pct:.0f}%")
        if report.promoter_trend:
            parts.append(f"stake {'rising' if report.promoter_trend=='INCREASING' else 'FALLING'}")
        if report.concall_sentiment: parts.append(f"concall {report.concall_sentiment.lower()}")
        return " | ".join(parts) if parts else "Insufficient data"

    def print_report(self, report):
        print(f"\n{'='*60}")
        print(f"  MANAGEMENT DNA — {report.company_name}")
        print(f"{'='*60}")
        print(f"  Rating: {report.overall_rating} | Score: {report.management_score:.0f}/100")
        print(f"  Summary: {report.summary}")
        if report.promoter_holding_pct:
            print(f"  Promoter: {report.promoter_holding_pct:.1f}%")
        if report.fii_pct: print(f"  FII (est): ~{report.fii_pct:.1f}%")
        if report.dii_pct: print(f"  DII (est): ~{report.dii_pct:.1f}%")
        for n in report.quality_notes: print(f"    • {n}")
        if report.promoter_alerts:
            for a in report.promoter_alerts:
                print(f"    🔴 [{a.alert_type}] {a.description}")
        if report.concall_red_flags:
            for f in report.concall_red_flags: print(f"    ⚠️ {f}")
        print(f"{'='*60}")
