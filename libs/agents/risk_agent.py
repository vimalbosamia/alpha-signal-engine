"""Agent 9: Risk Management Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class RiskManagementAgent(TradingAgent):
    """Evaluates risk: R:R ratio, position sizing, drawdown, leverage."""

    @property
    def name(self) -> str:
        return "risk_management"

    @property
    def weight(self) -> float:
        return 1.5  # Risk is critical — higher weight

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        rr_ratio = float(self._safe_get(context, "risk_reward_ratio", 0.0))
        max_drawdown_pct = float(self._safe_get(context, "max_drawdown_pct", 0.0))
        leverage = float(self._safe_get(context, "leverage", 1.0))
        portfolio_exposure_pct = float(self._safe_get(context, "portfolio_exposure_pct", 0.0))
        consecutive_losses = int(self._safe_get(context, "consecutive_losses", 0))
        stop_dist_pct = float(self._safe_get(context, "stop_dist_pct", 0.0))

        score = 0.0
        reasons: list[str] = []
        risk_level = 0.3

        # Risk-reward evaluation
        if rr_ratio >= 3.0:
            score += 0.4
            reasons.append(f"Excellent R:R ({rr_ratio:.1f})")
        elif rr_ratio >= 2.0:
            score += 0.2
            reasons.append(f"Good R:R ({rr_ratio:.1f})")
        elif rr_ratio >= 1.0:
            reasons.append(f"Acceptable R:R ({rr_ratio:.1f})")
        else:
            score -= 0.4
            reasons.append(f"Poor R:R ({rr_ratio:.1f})")
            risk_level = 0.7

        # Drawdown check
        if max_drawdown_pct > 10:
            score -= 0.3
            risk_level = max(risk_level, 0.8)
            reasons.append(f"High drawdown ({max_drawdown_pct:.1f}%)")
        elif max_drawdown_pct > 5:
            score -= 0.1
            risk_level = max(risk_level, 0.6)
            reasons.append(f"Elevated drawdown ({max_drawdown_pct:.1f}%)")

        # Leverage risk
        if leverage > 10:
            score -= 0.3
            risk_level = max(risk_level, 0.9)
            reasons.append(f"Extreme leverage ({leverage}x)")
        elif leverage > 5:
            score -= 0.1
            risk_level = max(risk_level, 0.6)
            reasons.append(f"High leverage ({leverage}x)")

        # Portfolio exposure
        if portfolio_exposure_pct > 80:
            score -= 0.3
            risk_level = max(risk_level, 0.8)
            reasons.append(f"Overexposed ({portfolio_exposure_pct:.0f}%)")

        # Consecutive losses — tilt protection
        if consecutive_losses >= 5:
            score -= 0.3
            reasons.append(f"Losing streak ({consecutive_losses})")
        elif consecutive_losses >= 3:
            score -= 0.1
            reasons.append(f"Recent losses ({consecutive_losses})")

        score = self._clamp(score)
        should_trade = risk_level < 0.8 and rr_ratio >= 1.0

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=0.8,  # Risk assessment is relatively objective
            reasoning="; ".join(reasons) or "Risk within bounds",
            signals={
                "rr_ratio": rr_ratio,
                "drawdown": max_drawdown_pct,
                "leverage": leverage,
                "exposure": portfolio_exposure_pct,
            },
            should_trade=should_trade,
            risk_level=risk_level,
        )
