"""
Layer 4 — Confirmation Layer.

Stacks confirmation signals on top of the primary trigger.
Each confirmation adds 5-10 points. Max 1 per category (dedup correlated).
Cross-category diversity bonus: +5 per additional unique category.

Also checks indicator-based confirmations (volume spike, RSI alignment, etc.)
"""
from __future__ import annotations

from libs.confluence.strategy_registry import (
    StrategyCategory, get_category, is_confirmation,
)
from libs.core.models.domain import SignalCandidate


# Points per confirmation signal type
_CONFIRMATION_POINTS: dict[str, float] = {
    # Strategy-based confirmations
    "mtf_alignment":              10.0,
    "trend_following":             8.0,
    "momentum_continuation":       8.0,
    "candle_momentum":             7.0,
    "candle_direction_flip":       6.0,
    "pullback_continuation":       7.0,
    "pullback_bear_continuation":  7.0,
    "rsi_mean_reversion":          6.0,
    "macd_crossover":              6.0,
}

# Indicator-based confirmation points
_INDICATOR_CONFIRMATIONS: dict[str, float] = {
    "volume_spike":    7.0,   # relative volume > 1.5x
    "rsi_aligned":     5.0,   # RSI > 50 for BUY, < 50 for SELL
    "macd_aligned":    5.0,   # MACD histogram positive for BUY, negative for SELL
    "vwap_support":    6.0,   # price above VWAP for BUY, below for SELL
    "ema_stack":       8.0,   # EMA9 > EMA20 > EMA50 for BUY (reverse for SELL)
    "adx_strong":      5.0,   # ADX > 25
}


def score_confirmations(
    candidates: list[SignalCandidate],
    direction: str,
    indicators: dict | None = None,
) -> tuple[float, list[str], list[str], float]:
    """Score the confirmation layer.

    Args:
        candidates: All strategy candidates for this symbol.
        direction: "BUY" or "SELL".
        indicators: Dict of computed indicator values (rsi, macd_histogram, etc.)

    Returns:
        (score, supporting_strategies, confirmation_signals, diversity_bonus)
    """
    total_score = 0.0
    supporting: list[str] = []
    confirmations: list[str] = []
    categories_seen: set[StrategyCategory] = set()

    # ── Strategy-based confirmations ──
    # Pick best per category
    best_per_cat: dict[StrategyCategory, SignalCandidate] = {}
    for c in candidates:
        if not is_confirmation(c.strategy_name):
            continue
        if c.proposed_action.value != direction:
            continue
        cat = get_category(c.strategy_name)
        existing = best_per_cat.get(cat)
        if existing is None or c.confidence > existing.confidence:
            best_per_cat[cat] = c

    for cat, cand in best_per_cat.items():
        points = _CONFIRMATION_POINTS.get(cand.strategy_name, 5.0)
        total_score += points
        supporting.append(cand.strategy_name)
        categories_seen.add(cat)

    # ── Indicator-based confirmations ──
    if indicators:
        # Volume spike
        rel_vol = indicators.get("volume_relative", 1.0)
        if rel_vol and rel_vol > 1.5:
            total_score += _INDICATOR_CONFIRMATIONS["volume_spike"]
            confirmations.append("volume_spike")

        # RSI alignment
        rsi = indicators.get("rsi")
        if rsi is not None:
            if (direction == "BUY" and 40 < rsi < 70) or \
               (direction == "SELL" and 30 < rsi < 60):
                total_score += _INDICATOR_CONFIRMATIONS["rsi_aligned"]
                confirmations.append("rsi_aligned")

        # MACD alignment
        macd_hist = indicators.get("macd_histogram")
        if macd_hist is not None:
            if (direction == "BUY" and macd_hist > 0) or \
               (direction == "SELL" and macd_hist < 0):
                total_score += _INDICATOR_CONFIRMATIONS["macd_aligned"]
                confirmations.append("macd_aligned")

        # VWAP support
        vwap = indicators.get("vwap")
        close = indicators.get("close")
        if vwap and close:
            if (direction == "BUY" and close >= vwap) or \
               (direction == "SELL" and close <= vwap):
                total_score += _INDICATOR_CONFIRMATIONS["vwap_support"]
                confirmations.append("vwap_support")

        # EMA stack
        ema9 = indicators.get("ema_9")
        ema20 = indicators.get("ema_20")
        ema50 = indicators.get("ema_50")
        if ema9 and ema20 and ema50:
            if direction == "BUY" and ema9 > ema20 > ema50:
                total_score += _INDICATOR_CONFIRMATIONS["ema_stack"]
                confirmations.append("ema_stack")
            elif direction == "SELL" and ema9 < ema20 < ema50:
                total_score += _INDICATOR_CONFIRMATIONS["ema_stack"]
                confirmations.append("ema_stack")

        # ADX strong trend
        adx = indicators.get("adx")
        if adx and adx > 25:
            total_score += _INDICATOR_CONFIRMATIONS["adx_strong"]
            confirmations.append("adx_strong")

    # ── Diversity bonus: +5 per unique category beyond the first ──
    unique_categories = len(categories_seen)
    diversity_bonus = max(0, (unique_categories - 1)) * 5.0

    return total_score, supporting, confirmations, diversity_bonus
