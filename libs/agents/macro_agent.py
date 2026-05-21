"""Agent 5: Macro/News Intelligence Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class MacroNewsAgent(TradingAgent):
    """Evaluates macro conditions: DXY, yields, BTC dominance, events."""

    @property
    def name(self) -> str:
        return "macro_news"

    @property
    def weight(self) -> float:
        return 0.8

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        dxy_trend = str(self._safe_get(context, "dxy_trend", "flat"))
        yield_trend = str(self._safe_get(context, "yield_trend", "flat"))
        btc_dominance_change = float(self._safe_get(context, "btc_dominance_change", 0.0))
        fomc_within_24h = bool(self._safe_get(context, "fomc_within_24h", False))
        cpi_within_24h = bool(self._safe_get(context, "cpi_within_24h", False))
        funding_rate = float(self._safe_get(context, "funding_rate", 0.0))
        asset_class = str(self._safe_get(context, "asset_class", "crypto"))

        score = 0.0
        reasons: list[str] = []
        risk_level = 0.3

        # Event risk
        if fomc_within_24h or cpi_within_24h:
            risk_level = 0.8
            reasons.append("High-impact macro event within 24h")

        # DXY (strong dollar typically bearish for risk assets)
        if dxy_trend == "rising":
            score -= 0.2
            reasons.append("DXY rising — risk-off")
        elif dxy_trend == "falling":
            score += 0.2
            reasons.append("DXY falling — risk-on")

        # Yields (rising yields pressure growth/risk assets)
        if yield_trend == "rising":
            score -= 0.15
            reasons.append("Yields rising")
        elif yield_trend == "falling":
            score += 0.15
            reasons.append("Yields falling")

        # BTC dominance (rising = bad for alts, falling = alt season)
        if asset_class == "crypto":
            if btc_dominance_change > 1.0:
                score -= 0.15
                reasons.append("BTC dominance rising — weak alts")
            elif btc_dominance_change < -1.0:
                score += 0.15
                reasons.append("BTC dominance falling — alt strength")

            # Funding rate extremes
            if funding_rate > 0.05:
                score -= 0.1
                reasons.append("Extreme positive funding — crowded longs")
            elif funding_rate < -0.05:
                score += 0.1
                reasons.append("Negative funding — potential short squeeze")

        score = self._clamp(score)
        confidence = 0.4 if not reasons else 0.6
        should_trade = not (fomc_within_24h or cpi_within_24h)

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=confidence,
            reasoning="; ".join(reasons) or "No significant macro factors",
            signals={
                "dxy_trend": dxy_trend,
                "yield_trend": yield_trend,
                "funding_rate": funding_rate,
                "event_risk": fomc_within_24h or cpi_within_24h,
            },
            should_trade=should_trade,
            risk_level=risk_level,
        )
