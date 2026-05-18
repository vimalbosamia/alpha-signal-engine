"""
ScalperBot — exploits fast micro-moves on the shortest timeframes.

Accepts signals when:
  - Timeframe is M1 or M5 (short-term only)
  - Estimated R:R >= 1.5 (minimum payoff to justify spread/fees)
  - All strategies accepted (no strategy filter)

Target: tp1 always (quick in, quick out).
"""
from __future__ import annotations

from libs.core.models.domain import SignalOutput, Timeframe
from libs.paper_trading.bot_agent import BotAgent

_ALLOWED_TIMEFRAMES: frozenset[Timeframe] = frozenset({
    Timeframe.ONE_MIN,
    Timeframe.FIVE_MIN,
    Timeframe.FIFTEEN_MIN,
})

_MIN_RR: float = 1.0


class ScalperBot(BotAgent):
    """High-frequency scalper limited to M1/M5 with minimum R:R filter."""

    NAME = "ScalperBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        return signal.timeframe in _ALLOWED_TIMEFRAMES and signal.estimated_risk_reward >= _MIN_RR

    def _target_exit(self) -> str:
        return "tp1"
