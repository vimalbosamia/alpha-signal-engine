"""
ReversalBot — fades exhausted moves at key reversal patterns.

Accepts signals when ALL of:
  - Strategy is in the known reversal set
  - Signal has at least one recognised reversal pattern
  - Market regime is NOT RANGING_HIGH_VOL (choppy)
  - Timeframe is M15 or H1
"""
from __future__ import annotations

from libs.core.models.domain import MarketRegime, SignalOutput, Timeframe
from libs.paper_trading.bot_agent import BotAgent

_STRATEGIES: frozenset[str] = frozenset({
    "hammer_reversal", "shooting_star_reversal",
})

_REVERSAL_PATTERNS: frozenset[str] = frozenset({
    "hammer",
    "inverted_hammer",
    "shooting_star",
    "hanging_man",
    "bullish_engulfing",
    "bearish_engulfing",
    "morning_star",
    "evening_star",
    "pin_bar",
    "rejection_candle",
    "tweezer_top",
    "tweezer_bottom",
})

_SKIP_REGIMES: frozenset[MarketRegime] = frozenset({MarketRegime.RANGING_HIGH_VOL})

_ALLOWED_TIMEFRAMES: frozenset[Timeframe] = frozenset({
    Timeframe.FIVE_MIN,
    Timeframe.FIFTEEN_MIN,
    Timeframe.THIRTY_MIN,
    Timeframe.ONE_HOUR,
    Timeframe.FOUR_HOUR,
})


class ReversalBot(BotAgent):
    """Targets high-probability candle reversal setups."""

    NAME = "ReversalBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        return signal.strategy_name in _STRATEGIES
