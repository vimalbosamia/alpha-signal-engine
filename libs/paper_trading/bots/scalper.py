"""
ScalperBot — exploits fast micro-moves on the shortest timeframes.

Accepts signals when:
  - Timeframe is M5 or M15 (short-term only)
  - R:R >= 1.5
  - Confidence >= 0.25

Target: tp1 always (quick in, quick out).
"""
from __future__ import annotations

from libs.core.models.domain import SignalOutput, Timeframe
from libs.paper_trading.bot_agent import BotAgent

_ALLOWED_TIMEFRAMES: frozenset[Timeframe] = frozenset({
    Timeframe.FIVE_MIN,
    Timeframe.FIFTEEN_MIN,
})

_MIN_RR: float = 1.5
_MIN_CONFIDENCE: float = 0.25


class ScalperBot(BotAgent):
    """High-frequency scalper limited to M5/M15 with minimum R:R filter."""

    NAME = "ScalperBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        if signal.timeframe not in _ALLOWED_TIMEFRAMES:
            return False
        if signal.estimated_risk_reward < _MIN_RR:
            return False
        if signal.confidence < _MIN_CONFIDENCE:
            return False
        return True

    def _target_exit(self) -> str:
        return "tp1"
