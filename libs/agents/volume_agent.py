"""Agent 4: Volume & Liquidity Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class VolumeLiquidityAgent(TradingAgent):
    """Analyzes volume profile, liquidity conditions, and orderflow bias."""

    @property
    def name(self) -> str:
        return "volume_liquidity"

    @property
    def weight(self) -> float:
        return 1.0

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        volume_ratio = float(self._safe_get(context, "volume_ratio", 1.0))
        spread_pct = float(self._safe_get(context, "spread_pct", 0.0))
        buy_volume_pct = float(self._safe_get(context, "buy_volume_pct", 50.0))
        open_interest_change = float(self._safe_get(context, "open_interest_change", 0.0))

        score = 0.0
        reasons: list[str] = []

        # Volume confirmation
        if volume_ratio > 1.5:
            score += 0.3
            reasons.append(f"High volume ({volume_ratio:.1f}x avg)")
        elif volume_ratio < 0.5:
            score -= 0.2
            reasons.append(f"Low volume ({volume_ratio:.1f}x avg)")

        # Orderflow bias
        if buy_volume_pct > 60:
            score += 0.2
            reasons.append(f"Buy pressure dominant ({buy_volume_pct:.0f}%)")
        elif buy_volume_pct < 40:
            score -= 0.2
            reasons.append(f"Sell pressure dominant ({buy_volume_pct:.0f}%)")

        # Open interest
        if open_interest_change > 5:
            score += 0.1
            reasons.append("Rising open interest")
        elif open_interest_change < -5:
            score -= 0.1
            reasons.append("Declining open interest")

        # Spread/liquidity risk
        risk_level = 0.3
        if spread_pct > 0.5:
            risk_level = 0.7
            reasons.append(f"Wide spread ({spread_pct:.2f}%)")
        elif spread_pct > 0.2:
            risk_level = 0.5

        score = self._clamp(score)
        should_trade = volume_ratio > 0.3 and spread_pct < 1.0

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=min(0.8, 0.4 + volume_ratio * 0.1),
            reasoning="; ".join(reasons) or "Normal volume conditions",
            signals={
                "volume_ratio": volume_ratio,
                "spread_pct": spread_pct,
                "buy_volume_pct": buy_volume_pct,
            },
            should_trade=should_trade,
            risk_level=risk_level,
        )
