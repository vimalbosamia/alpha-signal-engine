"""Agent 3: Pattern Recognition Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent

_BULLISH_PATTERNS = {
    "hammer", "morning_star", "bullish_engulfing", "piercing_line",
    "three_white_soldiers", "dragonfly_doji", "inverted_hammer",
}

_BEARISH_PATTERNS = {
    "shooting_star", "evening_star", "bearish_engulfing", "dark_cloud_cover",
    "three_black_crows", "gravestone_doji", "hanging_man",
}

_NEUTRAL_PATTERNS = {
    "doji", "spinning_top", "inside_bar", "outside_bar", "marubozu",
}


class PatternRecognitionAgent(TradingAgent):
    """Evaluates candlestick patterns and their historical reliability."""

    @property
    def name(self) -> str:
        return "pattern_recognition"

    @property
    def weight(self) -> float:
        return 1.0

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        patterns: list[str] = self._safe_get(context, "patterns", []) or []
        pattern_confidences: dict[str, float] = self._safe_get(
            context, "pattern_confidences", {}
        ) or {}

        if not patterns:
            return AgentOutput(
                agent_name=self.name,
                score=0.0,
                confidence=0.2,
                reasoning="No candlestick patterns detected",
                should_trade=False,
                risk_level=0.5,
            )

        bullish_score = 0.0
        bearish_score = 0.0
        reasons: list[str] = []

        for p in patterns:
            p_lower = p.lower().replace(" ", "_")
            conf = pattern_confidences.get(p, 0.7)

            if p_lower in _BULLISH_PATTERNS:
                bullish_score += conf * 0.3
                reasons.append(f"Bullish: {p}")
            elif p_lower in _BEARISH_PATTERNS:
                bearish_score += conf * 0.3
                reasons.append(f"Bearish: {p}")
            else:
                reasons.append(f"Neutral: {p}")

        score = self._clamp(bullish_score - bearish_score)
        total_patterns = len(patterns)
        confidence = min(0.9, 0.3 + total_patterns * 0.15)

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=round(confidence, 3),
            reasoning="; ".join(reasons),
            signals={"patterns": patterns, "bullish_score": bullish_score, "bearish_score": bearish_score},
            should_trade=abs(score) > 0.15,
            risk_level=0.4,
        )
