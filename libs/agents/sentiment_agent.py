"""Agent 6: Sentiment Intelligence Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class SentimentAgent(TradingAgent):
    """Evaluates market sentiment from fear/greed, social signals, and positioning."""

    @property
    def name(self) -> str:
        return "sentiment"

    @property
    def weight(self) -> float:
        return 0.7

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        fear_greed = float(self._safe_get(context, "fear_greed_index", 50.0))
        long_short_ratio = float(self._safe_get(context, "long_short_ratio", 1.0))
        social_sentiment = float(self._safe_get(context, "social_sentiment", 0.0))

        score = 0.0
        reasons: list[str] = []

        # Fear/Greed as contrarian indicator
        if fear_greed < 20:
            score += 0.3
            reasons.append(f"Extreme fear ({fear_greed:.0f}) — contrarian bullish")
        elif fear_greed < 35:
            score += 0.15
            reasons.append(f"Fear ({fear_greed:.0f})")
        elif fear_greed > 80:
            score -= 0.3
            reasons.append(f"Extreme greed ({fear_greed:.0f}) — contrarian bearish")
        elif fear_greed > 65:
            score -= 0.15
            reasons.append(f"Greed ({fear_greed:.0f})")

        # Long/short ratio (contrarian)
        if long_short_ratio > 2.0:
            score -= 0.2
            reasons.append(f"Crowded longs ({long_short_ratio:.1f})")
        elif long_short_ratio < 0.5:
            score += 0.2
            reasons.append(f"Crowded shorts ({long_short_ratio:.1f})")

        # Social sentiment
        if social_sentiment > 0.5:
            score += 0.1
            reasons.append("Positive social sentiment")
        elif social_sentiment < -0.5:
            score -= 0.1
            reasons.append("Negative social sentiment")

        score = self._clamp(score)
        confidence = 0.4 + min(0.3, abs(fear_greed - 50) / 100)

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=round(confidence, 3),
            reasoning="; ".join(reasons) or "Neutral sentiment",
            signals={
                "fear_greed": fear_greed,
                "long_short_ratio": long_short_ratio,
                "social_sentiment": social_sentiment,
            },
            should_trade=True,
            risk_level=0.6 if fear_greed > 75 or fear_greed < 25 else 0.3,
        )
