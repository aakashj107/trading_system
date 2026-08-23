"""
Risk Manager — Circuit breakers, drawdown limits, correlation groups,
and India VIX risk sizing.

v3.3 — VIX sizing is now delegated to market_intelligence.IndiaVIXRiskGate
(the same engine run_all.py uses), so DailyRunner and run_all.py stay in sync.
The class still exposes the DailyRunner-facing API:
    .regime
    .update_vix(vix_value)
    .vix_regime / .vix_position_pct / .vix_max_positions / .vix_risk_mode / .vix_message
    .VIX_THRESHOLDS        (class attribute, used for the dashboard table)
    .check_pre_trade(symbol, side, positions, price, atr) -> (allowed, reason, detail)
"""
from typing import Dict, List

from config import CORRELATION_GROUPS  # Single source of truth


class RiskManager:
    # ── VIX dashboard table (threshold, regime label, position%, max positions) ──
    VIX_THRESHOLDS = [
        (13, "AGGRESSIVE", 0.15, 5),
        (17, "NORMAL",     0.12, 4),
        (22, "CAUTIOUS",   0.10, 3),
        (28, "DEFENSIVE",  0.07, 2),
        (99, "CASH_PRES",  0.04, 1),
    ]

    def __init__(self, total_capital=500000.0, max_daily_loss_pct=0.03,
                 max_drawdown_pct=0.10, max_correlated=2, cool_down_days=3,
                 risk_config=None):
        self.total_capital = total_capital
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_correlated = max_correlated
        self.cool_down_days = cool_down_days
        self.risk_config = risk_config
        self.peak_capital = total_capital
        self.circuit_breaker_active = False
        self.alerts: List[str] = []

        # ── VIX-backed position sizing (delegated to the real gate) ──
        self.regime = "UNKNOWN"
        self.vix_value = 15.0
        self.vix_regime = "NORMAL"
        self.vix_position_pct = 0.12
        self.vix_max_positions = 4
        self.vix_risk_mode = "Normal"
        self.vix_message = "Normal volatility — standard sizing"

        try:
            from market_intelligence import IndiaVIXRiskGate
            self._vix_gate = IndiaVIXRiskGate()
            # Apply a sane default so the dashboard is populated before update_vix()
            self.update_vix(self.vix_value)
        except Exception:
            self._vix_gate = None

    # ==========================================================
    #  VIX RISK GATE (delegated)
    # ==========================================================
    def update_vix(self, vix_value: float):
        """Recompute VIX-driven position sizing from a VIX level."""
        self.vix_value = vix_value
        if self._vix_gate is None:
            return
        try:
            rp = self._vix_gate.get_risk_profile(vix_value)
            self.vix_regime = rp.vix_regime
            self.vix_position_pct = rp.position_pct
            self.vix_max_positions = rp.max_positions
            self.vix_risk_mode = rp.risk_label
            self.vix_message = rp.message
        except Exception:
            pass

    # ==========================================================
    #  PRE-TRADE CHECK
    #  DailyRunner calls:  allowed, reason, _ = check_pre_trade(
    #      symbol, side, positions, price, atr)
    # ==========================================================
    def check_pre_trade(self, symbol, side, positions, price=None, atr=None):
        detail = None
        if self.circuit_breaker_active:
            return False, "Circuit breaker active", detail
        if side in ("BUY", "SHORT"):
            for group, members in CORRELATION_GROUPS.items():
                if symbol in members:
                    existing = sum(1 for s in positions if s in members)
                    if existing >= self.max_correlated:
                        return (False,
                                f"Too many {group} positions ({existing})",
                                {"group": group, "existing": existing})
        return True, "OK", detail

    def get_dashboard(self, positions):
        return {"circuit_breaker": self.circuit_breaker_active,
                "open_positions": len(positions), "alerts": self.alerts}
