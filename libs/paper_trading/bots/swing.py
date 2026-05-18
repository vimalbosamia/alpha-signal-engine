"""
SwingBot — holds multi-session moves on higher timeframes.

Accepts signals when:
  - Timeframe is H1 or H4
  - Estimated R:R >= 2.5 (only high-quality setups justify overnight hold)
  - All strategies accepted (no strategy filter)

Target: tp2 (held for larger moves).
"""
from __future__ import annotations

from libs.core.models.domain import SignalOutput, Timeframe
from libs.paper_trading.bot_agent import BotAgent

_ALLOWED_TIMEFRAMES: frozenset[Timeframe] = frozenset({
    Timeframe.THIRTY_MIN,
    Timeframe.ONE_HOUR,
    Timeframe.FOUR_HOUR,
    Timeframe.ONE_DAY,
})

_MIN_RR: float = 1.5


class SwingBot(BotAgent):
    """Swing trader targeting extended moves on H1/H4 with high R:R."""

    NAME = "SwingBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        # Aggressive mode: take signals with decent R:R on any timeframe
        return signal.estimated_risk_reward >= _MIN_RR

    def _target_exit(self) -> str:
        return "tp2"
