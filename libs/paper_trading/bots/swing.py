"""
SwingBot — holds multi-session moves on breakout setups.

Accepts signals when:
  - Strategy is a breakout/retest variant
  - R:R >= 2.0 (high-quality setups only)
  - Confidence >= 0.35
  - Regime is trending or breakout (not ranging)

Target: tp2 (held for larger moves).
"""
from __future__ import annotations

from libs.core.models.domain import MarketRegime, SignalOutput
from libs.paper_trading.bot_agent import BotAgent

_STRATEGIES: frozenset[str] = frozenset({
    "resistance_breakout", "support_breakdown", "volume_breakout",
    "atr_breakout", "range_breakout", "break_and_retest",
    "volatility_squeeze", "opening_range_breakout", "fibonacci_bounce",
})

_ALLOWED_REGIMES: frozenset[MarketRegime] = frozenset({
    MarketRegime.TRENDING_UP,
    MarketRegime.TRENDING_DOWN,
    MarketRegime.BREAKOUT,
})

_MIN_CONFIDENCE: float = 0.35
_MIN_RR: float = 2.0


class SwingBot(BotAgent):
    """Swing trader targeting extended moves with high R:R."""

    NAME = "SwingBot"

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

    def _target_exit(self) -> str:
        return "tp2"
