"""
Strategy Registry — maps every strategy to its confluence layer and category.

Each strategy has exactly ONE layer role (trigger vs confirmation) and ONE category.
This prevents correlated stacking: max 1 strategy per category counts toward scoring.
"""
from __future__ import annotations

from enum import Enum


class StrategyLayer(str, Enum):
    """Which confluence layer a strategy belongs to."""
    TRIGGER = "trigger"
    CONFIRMATION = "confirmation"


class StrategyCategory(str, Enum):
    """Logical grouping — max 1 per category counted in confluence."""
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    REVERSAL = "reversal"
    TREND = "trend"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    PULLBACK = "pullback"
    SCALP = "scalp"
    LEVEL = "level"
    CONFIRMATION = "confirmation"
    VOLATILITY = "volatility"


# ── Strategy → (layer, category) mapping ─────────────────────────────────────

_REGISTRY: dict[str, tuple[StrategyLayer, StrategyCategory]] = {
    # Breakout triggers
    "resistance_breakout":      (StrategyLayer.TRIGGER, StrategyCategory.BREAKOUT),
    "volume_breakout":          (StrategyLayer.TRIGGER, StrategyCategory.BREAKOUT),
    "atr_breakout":             (StrategyLayer.TRIGGER, StrategyCategory.BREAKOUT),
    "range_breakout":           (StrategyLayer.TRIGGER, StrategyCategory.BREAKOUT),
    "volatility_squeeze":       (StrategyLayer.TRIGGER, StrategyCategory.VOLATILITY),
    "opening_range_breakout":   (StrategyLayer.TRIGGER, StrategyCategory.BREAKOUT),
    "break_and_retest":         (StrategyLayer.TRIGGER, StrategyCategory.PULLBACK),

    # Breakdown triggers
    "support_breakdown":        (StrategyLayer.TRIGGER, StrategyCategory.BREAKDOWN),

    # Reversal triggers
    "hammer_reversal":          (StrategyLayer.TRIGGER, StrategyCategory.REVERSAL),
    "shooting_star_reversal":   (StrategyLayer.TRIGGER, StrategyCategory.REVERSAL),
    "candlestick_reversal":     (StrategyLayer.TRIGGER, StrategyCategory.REVERSAL),
    "failed_breakout_reversal": (StrategyLayer.TRIGGER, StrategyCategory.REVERSAL),
    "liquidity_sweep":          (StrategyLayer.TRIGGER, StrategyCategory.REVERSAL),

    # Trend triggers
    "ema_crossover":            (StrategyLayer.TRIGGER, StrategyCategory.TREND),
    "sma_crossover":            (StrategyLayer.TRIGGER, StrategyCategory.TREND),

    # Level triggers
    "fibonacci_bounce":         (StrategyLayer.TRIGGER, StrategyCategory.LEVEL),
    "vwap_reclaim":             (StrategyLayer.TRIGGER, StrategyCategory.LEVEL),

    # Mean reversion triggers
    "bollinger_mean_reversion": (StrategyLayer.TRIGGER, StrategyCategory.MEAN_REVERSION),
    "gap_fill":                 (StrategyLayer.TRIGGER, StrategyCategory.MEAN_REVERSION),
    "range_fade":               (StrategyLayer.TRIGGER, StrategyCategory.MEAN_REVERSION),

    # Confirmation layer — these SUPPORT triggers, don't open trades alone
    "candle_momentum":          (StrategyLayer.CONFIRMATION, StrategyCategory.SCALP),
    "candle_direction_flip":    (StrategyLayer.CONFIRMATION, StrategyCategory.SCALP),
    "mtf_alignment":            (StrategyLayer.CONFIRMATION, StrategyCategory.CONFIRMATION),
    "momentum_continuation":    (StrategyLayer.CONFIRMATION, StrategyCategory.MOMENTUM),
    "pullback_continuation":    (StrategyLayer.CONFIRMATION, StrategyCategory.PULLBACK),
    "pullback_bear_continuation": (StrategyLayer.CONFIRMATION, StrategyCategory.PULLBACK),
    "trend_following":          (StrategyLayer.CONFIRMATION, StrategyCategory.TREND),
    "rsi_mean_reversion":       (StrategyLayer.CONFIRMATION, StrategyCategory.MOMENTUM),
    "macd_crossover":           (StrategyLayer.CONFIRMATION, StrategyCategory.MOMENTUM),
}


def get_layer(strategy_name: str) -> StrategyLayer:
    """Return the layer for a strategy. Unknown strategies default to TRIGGER."""
    entry = _REGISTRY.get(strategy_name)
    return entry[0] if entry else StrategyLayer.TRIGGER


def get_category(strategy_name: str) -> StrategyCategory:
    """Return the category for a strategy. Unknown strategies get BREAKOUT."""
    entry = _REGISTRY.get(strategy_name)
    return entry[1] if entry else StrategyCategory.BREAKOUT


def is_trigger(strategy_name: str) -> bool:
    return get_layer(strategy_name) == StrategyLayer.TRIGGER


def is_confirmation(strategy_name: str) -> bool:
    return get_layer(strategy_name) == StrategyLayer.CONFIRMATION


def get_all_triggers() -> list[str]:
    return [k for k, v in _REGISTRY.items() if v[0] == StrategyLayer.TRIGGER]


def get_all_confirmations() -> list[str]:
    return [k for k, v in _REGISTRY.items() if v[0] == StrategyLayer.CONFIRMATION]
