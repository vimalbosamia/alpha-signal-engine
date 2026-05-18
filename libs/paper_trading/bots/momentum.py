"""
MomentumBot — chases trending and breakout moves.

Accepts signals when:
  - Strategy is in the known momentum set, OR confidence >= 0.7
  - Market regime is trending/breakout (not ranging or climactic)
  - Timeframe is M15, M30, or H1
"""
from __future__ import annotations

from libs.core.models.domain import MarketRegime, SignalOutput, Timeframe
from libs.paper_trading.bot_agent import BotAgent

_STRATEGIES: frozenset[str] = frozenset({"ema_crossover", "resistance_breakout"})

_ALLOWED_REGIMES: frozenset[MarketRegime] = frozenset({
    MarketRegime.TRENDING_UP,
    MarketRegime.TRENDING_DOWN,
    MarketRegime.BREAKOUT,
})

_ALLOWED_TIMEFRAMES: frozenset[Timeframe] = frozenset({
    Timeframe.FIFTEEN_MIN,
    Timeframe.THIRTY_MIN,
    Timeframe.ONE_HOUR,
})

_HIGH_CONFIDENCE_THRESHOLD: float = 0.70


class MomentumBot(BotAgent):
    """Rides momentum signals in trending and breakout regimes."""

    NAME = "MomentumBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        if signal.timeframe not in _ALLOWED_TIMEFRAMES:
            return False

        if signal.market_regime not in _ALLOWED_REGIMES:
            return False

        strategy_match = signal.strategy_name in _STRATEGIES
        high_confidence = signal.confidence >= _HIGH_CONFIDENCE_THRESHOLD
        return strategy_match or high_confidence
