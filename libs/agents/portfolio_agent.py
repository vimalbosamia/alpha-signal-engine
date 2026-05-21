"""Agent 10: Portfolio Intelligence Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class PortfolioIntelligenceAgent(TradingAgent):
    """Manages portfolio-level risk: correlation, exposure, VaR, concentration."""

    @property
    def name(self) -> str:
        return "portfolio_intelligence"

    @property
    def weight(self) -> float:
        return 1.3

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        open_positions = int(self._safe_get(context, "open_positions", 0))
        max_positions = int(self._safe_get(context, "max_positions", 10))
        sector_exposure: dict[str, float] = self._safe_get(context, "sector_exposure", {}) or {}
        correlation_risk = float(self._safe_get(context, "correlation_risk", 0.0))
        portfolio_beta = float(self._safe_get(context, "portfolio_beta", 1.0))
        unrealized_pnl_pct = float(self._safe_get(context, "unrealized_pnl_pct", 0.0))
        directional_bias_pct = float(self._safe_get(context, "directional_bias_pct", 0.0))

        score = 0.0
        reasons: list[str] = []
        risk_level = 0.3

        # Position limit check
        if open_positions >= max_positions:
            score -= 0.5
            reasons.append(f"Max positions reached ({open_positions}/{max_positions})")
            risk_level = 0.9
        elif open_positions >= max_positions * 0.8:
            score -= 0.2
            reasons.append(f"Near position limit ({open_positions}/{max_positions})")
            risk_level = 0.6

        # Concentration risk
        if sector_exposure:
            max_sector_pct = max(sector_exposure.values()) if sector_exposure else 0
            if max_sector_pct > 50:
                score -= 0.3
                reasons.append(f"Sector concentration ({max_sector_pct:.0f}%)")
                risk_level = max(risk_level, 0.7)

        # Correlation risk
        if correlation_risk > 0.7:
            score -= 0.3
            reasons.append(f"High correlation risk ({correlation_risk:.2f})")
            risk_level = max(risk_level, 0.7)

        # Directional exposure
        if abs(directional_bias_pct) > 70:
            score -= 0.2
            direction = "long" if directional_bias_pct > 0 else "short"
            reasons.append(f"Heavy {direction} bias ({abs(directional_bias_pct):.0f}%)")
            risk_level = max(risk_level, 0.6)

        # Portfolio beta
        if portfolio_beta > 1.5:
            score -= 0.1
            reasons.append(f"High beta ({portfolio_beta:.2f})")
        elif portfolio_beta < 0.5:
            score += 0.1
            reasons.append(f"Defensive beta ({portfolio_beta:.2f})")

        # Unrealized PnL risk
        if unrealized_pnl_pct < -5:
            score -= 0.2
            reasons.append(f"Underwater portfolio ({unrealized_pnl_pct:.1f}%)")
            risk_level = max(risk_level, 0.7)

        score = self._clamp(score)
        should_trade = open_positions < max_positions and risk_level < 0.8

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=0.75,
            reasoning="; ".join(reasons) or "Portfolio risk within bounds",
            signals={
                "open_positions": open_positions,
                "correlation_risk": correlation_risk,
                "portfolio_beta": portfolio_beta,
            },
            should_trade=should_trade,
            risk_level=risk_level,
        )
