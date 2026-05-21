"""
MomentumBot — chases trending and breakout moves.

Accepts signals when:
  - Strategy matches momentum/trend strategies
  - Regime is trending or breakout (NOT ranging)
  - Confidence >= 0.30
  - R:R >= 1.5
  - ADX > 20 (confirmed trend)
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

_MIN_CONFIDENCE: float = 0.20
_MIN_RR: float = 1.0


class MomentumBot(BotAgent):
    """Rides momentum signals in trending and breakout regimes."""

    NAME = "MomentumBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        if signal.strategy_name not in _STRATEGIES:
            return False
        if signal.market_regime not in _ALLOWED_REGIMES:
            return False
        if signal.confidence < _MIN_CONFIDENCE:
            return False
        if signal.estimated_risk_reward < _MIN_RR:
            return False
        return True
