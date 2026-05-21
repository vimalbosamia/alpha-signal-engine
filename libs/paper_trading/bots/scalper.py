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

# Strategies appropriate for scalping — fast in/out micro-moves
_SCALP_STRATEGIES: frozenset[str] = frozenset({
    "candle_direction_flip", "candle_momentum",
    "ema_crossover", "macd_crossover",
    "bollinger_mean_reversion", "rsi_mean_reversion",
    "pullback_continuation", "pullback_bear_continuation",
    "vwap_reclaim", "gap_fill",
})

_MIN_RR: float = 1.5
_MIN_CONFIDENCE: float = 0.25


class ScalperBot(BotAgent):
    """High-frequency scalper limited to fast strategies on M5/M15."""

    NAME = "ScalperBot"

    # Candle-level strategies bypass timeframe filter (they work on any TF)
    _CANDLE_STRATEGIES: frozenset[str] = frozenset({
        "candle_direction_flip", "candle_momentum",
    })

    def should_take_signal(self, signal: SignalOutput) -> bool:
        # Must be a scalp-appropriate strategy
        if signal.strategy_name not in _SCALP_STRATEGIES:
            return False
        # Candle strategies: accept from any timeframe, lower confidence OK
        if signal.strategy_name in self._CANDLE_STRATEGIES:
            return signal.estimated_risk_reward >= 1.0
        # Other strategies: timeframe + confidence filter
        if signal.timeframe not in _ALLOWED_TIMEFRAMES:
            return False
        if signal.estimated_risk_reward < _MIN_RR:
            return False
        if signal.confidence < _MIN_CONFIDENCE:
            return False
        return True

    def _target_exit(self) -> str:
        return "tp1"
