"""Agent 2: Technical Analysis Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class TechnicalAnalysisAgent(TradingAgent):
    """Analyzes RSI, MACD, EMA structure, Bollinger Bands, and market structure."""

    @property
    def name(self) -> str:
        return "technical_analysis"

    @property
    def weight(self) -> float:
        return 1.3

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        rsi = float(self._safe_get(context, "rsi", 50.0))
        macd_hist = float(self._safe_get(context, "macd_histogram", 0.0))
        ema_alignment = float(self._safe_get(context, "ema_alignment", 1.0))
        bb_position = float(self._safe_get(context, "bb_position", 0.5))

        signals: dict[str, Any] = {}
        score = 0.0
        reasons: list[str] = []

        # RSI analysis
        if rsi < 30:
            score += 0.4
            reasons.append(f"RSI oversold ({rsi:.0f})")
            signals["rsi_signal"] = "oversold"
        elif rsi > 70:
            score -= 0.4
            reasons.append(f"RSI overbought ({rsi:.0f})")
            signals["rsi_signal"] = "overbought"
        else:
            signals["rsi_signal"] = "neutral"

        # MACD histogram
        if macd_hist > 0:
            score += min(0.3, macd_hist * 10)
            reasons.append("MACD bullish")
        elif macd_hist < 0:
            score -= min(0.3, abs(macd_hist) * 10)
            reasons.append("MACD bearish")

        # EMA alignment (fast/slow ratio)
        if ema_alignment > 1.005:
            score += 0.2
            reasons.append("EMAs bullish aligned")
        elif ema_alignment < 0.995:
            score -= 0.2
            reasons.append("EMAs bearish aligned")

        # Bollinger position
        if bb_position < 0.1:
            score += 0.2
            reasons.append("Near lower Bollinger Band")
        elif bb_position > 0.9:
            score -= 0.2
            reasons.append("Near upper Bollinger Band")

        score = self._clamp(score)

        # Confidence from signal agreement
        bullish_count = sum(1 for r in reasons if "bullish" in r or "oversold" in r or "lower" in r)
        bearish_count = sum(1 for r in reasons if "bearish" in r or "overbought" in r or "upper" in r)
        total_signals = bullish_count + bearish_count
        agreement = max(bullish_count, bearish_count) / max(total_signals, 1)
        confidence = 0.3 + agreement * 0.5

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=round(confidence, 3),
            reasoning="; ".join(reasons) or "No strong technical signals",
            signals=signals,
            should_trade=abs(score) > 0.2,
            risk_level=0.3 if abs(score) > 0.5 else 0.5,
        )
