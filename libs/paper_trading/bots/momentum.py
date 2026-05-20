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

_STRATEGIES: frozenset[str] = frozenset({
    "ema_crossover", "macd_crossover", "sma_crossover", "trend_following",
    "momentum_continuation", "mtf_alignment",
})

_ALLOWED_REGIMES: frozenset[MarketRegime] = frozenset({
    MarketRegime.TRENDING_UP,
    MarketRegime.TRENDING_DOWN,
    MarketRegime.BREAKOUT,
})

_ALLOWED_TIMEFRAMES: frozenset[Timeframe] = frozenset({
    Timeframe.FIVE_MIN,
    Timeframe.FIFTEEN_MIN,
    Timeframe.THIRTY_MIN,
    Timeframe.ONE_HOUR,
    Timeframe.FOUR_HOUR,
})

_HIGH_CONFIDENCE_THRESHOLD: float = 0.50


class MomentumBot(BotAgent):
    """Rides momentum signals in trending and breakout regimes."""

    NAME = "MomentumBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        return signal.strategy_name in _STRATEGIES
