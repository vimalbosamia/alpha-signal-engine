"""
Regime-Strategy Activation Matrix.

Provides an explicit mapping of which trading strategies are permitted
in each market regime. Each check returns an immutable StrategyActivation
describing whether the strategy is allowed and any confidence modifier.

Design rules:
  - All results are immutable (frozen dataclass)
  - No magic numbers — all sets are named class constants in MATRIX
  - Never raises — returns a safe StrategyActivation on any input
  - Confidence modifiers are bounded to [-0.10, +0.10]
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyActivation:
    """Immutable result of checking a strategy against a regime."""

    strategy_name: str
    is_allowed: bool
    confidence_modifier: float   # -0.10 to +0.10
    reason: str


# ── Engine ────────────────────────────────────────────────────────────────────

class RegimeStrategyMatrix:
    """Explicit mapping of which strategies are allowed in which market regimes."""

    # ── Confidence boost constants ─────────────────────────────────────────────
    _TREND_BOOST_AMOUNT: float = 0.05
    _TREND_BOOST_STRATEGIES: frozenset[str] = frozenset(
        {"trend_following", "momentum_continuation"}
    )

    # ── Matrix: regime → set of allowed strategy name prefixes ────────────────
    MATRIX: dict[str, set[str]] = {
        "trending_up": {
            "ema_crossover",
            "sma_crossover",
            "macd_crossover",
            "trend_following",
            "momentum_continuation",
            "pullback_continuation",
            "resistance_breakout",
            "break_and_retest",
            "volume_breakout",
            "mtf_alignment",
            "vwap_reclaim",
        },
        "trending_down": {
            "ema_crossover",
            "sma_crossover",
            "macd_crossover",
            "trend_following",
            "momentum_continuation",
            "pullback_bear_continuation",
            "support_breakdown",
            "break_and_retest",
            "volume_breakout",
            "mtf_alignment",
            "shooting_star_reversal",
        },
        "ranging_low_vol": {
            "bollinger_mean_reversion",
            "range_fade",
            "rsi_mean_reversion",
            "vwap_reclaim",
            "fibonacci_bounce",
            "gap_fill",
        },
        "ranging_high_vol": {
            "range_fade",
            "rsi_mean_reversion",
        },  # very few — choppy
        "breakout": {
            "resistance_breakout",
            "support_breakdown",
            "volume_breakout",
            "atr_breakout",
            "range_breakout",
            "volatility_squeeze",
            "opening_range_breakout",
        },
        "climactic": set(),   # NO strategies — too dangerous
        "unknown": {
            "hammer_reversal",
            "shooting_star_reversal",
        },  # only reversals
    }

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, strategy_name: str, regime: str) -> StrategyActivation:
        """
        Check if *strategy_name* is allowed in *regime*.

        Returns a StrategyActivation describing the outcome.  Applies a small
        confidence boost for ideal trend-following matches in trending regimes.
        """
        if regime == "climactic":
            return StrategyActivation(
                strategy_name=strategy_name,
                is_allowed=False,
                confidence_modifier=0.0,
                reason="No strategies in climactic regime",
            )

        allowed = self.MATRIX.get(regime, set())
        is_allowed = strategy_name in allowed

        if not is_allowed:
            return StrategyActivation(
                strategy_name=strategy_name,
                is_allowed=False,
                confidence_modifier=0.0,
                reason=f"{strategy_name} not allowed in {regime}",
            )

        # Boost confidence for ideal regime matches
        boost = (
            self._TREND_BOOST_AMOUNT
            if strategy_name in self._TREND_BOOST_STRATEGIES and "trending" in regime
            else 0.0
        )

        return StrategyActivation(
            strategy_name=strategy_name,
            is_allowed=True,
            confidence_modifier=boost,
            reason=f"{strategy_name} allowed in {regime}",
        )

    def get_allowed(self, regime: str) -> set[str]:
        """Return the set of allowed strategy names for *regime*."""
        return self.MATRIX.get(regime, set())
