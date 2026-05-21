"""
ReversalBot — targets candle reversal setups at extremes.

Accepts signals when:
  - Strategy is a reversal variant
  - R:R >= 2.0 (reversals need room)
  - Confidence >= 0.30
  - At least 1 reversal pattern detected
"""
from __future__ import annotations

from libs.core.models.domain import SignalOutput
from libs.paper_trading.bot_agent import BotAgent

_STRATEGIES: frozenset[str] = frozenset({
    "hammer_reversal", "shooting_star_reversal", "engulfing_reversal",
    "morning_star_reversal", "evening_star_reversal", "doji_reversal",
    "double_bottom", "double_top", "rsi_divergence",
})

_REVERSAL_PATTERNS: frozenset[str] = frozenset({
    "hammer", "inverted_hammer", "shooting_star", "hanging_man",
    "bullish_engulfing", "bearish_engulfing", "morning_star", "evening_star",
    "doji", "dragonfly_doji", "gravestone_doji", "piercing_line",
    "dark_cloud_cover", "tweezer_top", "tweezer_bottom",
})

_MIN_CONFIDENCE: float = 0.30
_MIN_RR: float = 2.0


class ReversalBot(BotAgent):
    """Targets high-probability candle reversal setups."""

    NAME = "ReversalBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        if signal.strategy_name not in _STRATEGIES:
            return False
        if signal.confidence < _MIN_CONFIDENCE:
            return False
        if signal.estimated_risk_reward < _MIN_RR:
            return False
        # Require at least 1 reversal pattern
        if not any(p in _REVERSAL_PATTERNS for p in signal.patterns_detected):
            return False
        return True
