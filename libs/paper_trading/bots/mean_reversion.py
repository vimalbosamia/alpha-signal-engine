"""
MeanReversionBot — exploits mean reversion in range-bound markets.

Accepts signals when:
  - Strategy is a pullback continuation variant
  - Market regime is ranging (low or high vol)
  - Timeframe is M15 or M30
"""
from __future__ import annotations

from libs.core.models.domain import MarketRegime, SignalOutput, Timeframe
from libs.paper_trading.bot_agent import BotAgent

_STRATEGIES: frozenset[str] = frozenset({
    "pullback_continuation", "pullback_bear_continuation", "rsi_mean_reversion",
})

_ALLOWED_REGIMES: frozenset[MarketRegime] = frozenset({
    MarketRegime.RANGING_LOW_VOL,
    MarketRegime.RANGING_HIGH_VOL,
})

_ALLOWED_TIMEFRAMES: frozenset[Timeframe] = frozenset({
    Timeframe.FIVE_MIN,
    Timeframe.FIFTEEN_MIN,
    Timeframe.THIRTY_MIN,
    Timeframe.ONE_HOUR,
})


class MeanReversionBot(BotAgent):
    """Buys/sells pullbacks within established ranges."""

    NAME = "MeanReversionBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        return signal.market_regime in _ALLOWED_REGIMES and signal.strategy_name in _STRATEGIES
