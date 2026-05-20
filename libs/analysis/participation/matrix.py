"""
MarketParticipationMatrix — decides which trading modes and directions
are permitted based on market state classification.

Market states:
  STRONG_BULL, WEAK_BULL, NEUTRAL, WEAK_BEAR, STRONG_BEAR,
  HIGH_VOLATILITY_EVENT, NEWS_LOCKDOWN, TREND_TRANSITION

For each state, defines:
  - Whether SPOT is allowed
  - Whether FUTURES LONG is allowed
  - Whether FUTURES SHORT is allowed
  - Confidence multiplier
  - Size multiplier
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MarketState(str, Enum):
    STRONG_BULL = "STRONG_BULL"
    WEAK_BULL = "WEAK_BULL"
    NEUTRAL = "NEUTRAL"
    WEAK_BEAR = "WEAK_BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    HIGH_VOLATILITY_EVENT = "HIGH_VOLATILITY_EVENT"
    NEWS_LOCKDOWN = "NEWS_LOCKDOWN"
    TREND_TRANSITION = "TREND_TRANSITION"


@dataclass(frozen=True)
class ParticipationDecision:
    """What trading is permitted for a given symbol."""
    market_state: MarketState
    spot_allowed: bool
    futures_long_allowed: bool
    futures_short_allowed: bool
    confidence_multiplier: float    # 0.0 to 1.0 — scale down confidence
    size_multiplier: float          # 0.0 to 1.0 — scale down position size
    reason: str


# ── Classification thresholds ────────────────────────────────────────────────

ADX_STRONG_TREND = 25
ADX_WEAK_TREND = 15
CONFLICT_HIGH = 0.90
CONFLICT_MODERATE = 0.75
CONFLICT_LOW = 0.50
BULL_BEAR_MARGIN = 0.10   # bull - bear must exceed this for directional call


def classify_market_state(
    net_bias: str,
    bullish_score: float,
    bearish_score: float,
    conflict_score: float,
    adx: float,
    is_news_lockdown: bool = False,
    is_high_volatility: bool = False,
) -> MarketState:
    """
    Classify current market into one of 8 states.

    Uses bias scores, conflict, ADX, and event flags.
    """
    if is_news_lockdown:
        return MarketState.NEWS_LOCKDOWN

    if is_high_volatility:
        return MarketState.HIGH_VOLATILITY_EVENT

    # Very high conflict with no clear winner — blocked
    if conflict_score >= CONFLICT_HIGH and abs(bullish_score - bearish_score) < 0.02:
        return MarketState.NEUTRAL

    spread = bullish_score - bearish_score

    # Strong trends: clear direction + ADX confirms
    if net_bias == "bullish" and spread > BULL_BEAR_MARGIN and adx >= ADX_STRONG_TREND:
        return MarketState.STRONG_BULL

    if net_bias == "bearish" and (-spread) > BULL_BEAR_MARGIN and adx >= ADX_STRONG_TREND:
        return MarketState.STRONG_BEAR

    # Weak trends: direction exists but not strong
    # Also catch cases where bias engine said "neutral" but one side leads
    if (net_bias == "bullish" or spread > 0.03) and spread > 0.01:
        return MarketState.WEAK_BULL

    if (net_bias == "bearish" or (-spread) > 0.03) and (-spread) > 0.01:
        return MarketState.WEAK_BEAR

    # Transition: ADX rising but no clear direction yet
    if adx >= ADX_WEAK_TREND and conflict_score >= CONFLICT_MODERATE:
        return MarketState.TREND_TRANSITION

    return MarketState.NEUTRAL


# ── Participation rules ──────────────────────────────────────────────────────

_RULES: dict[MarketState, dict] = {
    MarketState.STRONG_BULL: {
        "spot": True, "futures_long": True, "futures_short": False,
        "conf_mult": 1.0, "size_mult": 1.0,
        "reason": "Strong bullish — full participation",
    },
    MarketState.WEAK_BULL: {
        "spot": True, "futures_long": True, "futures_short": False,
        "conf_mult": 0.85, "size_mult": 0.8,
        "reason": "Weak bullish — reduced size",
    },
    MarketState.NEUTRAL: {
        "spot": False, "futures_long": False, "futures_short": False,
        "conf_mult": 0.0, "size_mult": 0.0,
        "reason": "Neutral/high conflict — no new entries",
    },
    MarketState.WEAK_BEAR: {
        "spot": False, "futures_long": False, "futures_short": True,
        "conf_mult": 0.85, "size_mult": 0.7,
        "reason": "Weak bearish — futures short only, reduced size",
    },
    MarketState.STRONG_BEAR: {
        "spot": False, "futures_long": False, "futures_short": True,
        "conf_mult": 1.0, "size_mult": 0.8,
        "reason": "Strong bearish — futures short allowed",
    },
    MarketState.HIGH_VOLATILITY_EVENT: {
        "spot": False, "futures_long": False, "futures_short": False,
        "conf_mult": 0.0, "size_mult": 0.0,
        "reason": "High volatility — exit management only",
    },
    MarketState.NEWS_LOCKDOWN: {
        "spot": False, "futures_long": False, "futures_short": False,
        "conf_mult": 0.0, "size_mult": 0.0,
        "reason": "News lockdown — no new entries",
    },
    MarketState.TREND_TRANSITION: {
        "spot": False, "futures_long": False, "futures_short": False,
        "conf_mult": 0.5, "size_mult": 0.5,
        "reason": "Trend transition — wait for confirmation",
    },
}


def get_participation(
    net_bias: str,
    bullish_score: float,
    bearish_score: float,
    conflict_score: float,
    adx: float,
    is_news_lockdown: bool = False,
    is_high_volatility: bool = False,
) -> ParticipationDecision:
    """
    Get trading participation decision for current market state.

    Returns which modes/directions are allowed + scaling factors.
    """
    state = classify_market_state(
        net_bias=net_bias,
        bullish_score=bullish_score,
        bearish_score=bearish_score,
        conflict_score=conflict_score,
        adx=adx,
        is_news_lockdown=is_news_lockdown,
        is_high_volatility=is_high_volatility,
    )

    rule = _RULES[state]

    # Conflict score override: reduce participation even in allowed states
    conf_mult = rule["conf_mult"]
    size_mult = rule["size_mult"]
    if conflict_score >= CONFLICT_MODERATE and conf_mult > 0:
        conf_mult *= 0.7
        size_mult *= 0.7

    return ParticipationDecision(
        market_state=state,
        spot_allowed=rule["spot"],
        futures_long_allowed=rule["futures_long"],
        futures_short_allowed=rule["futures_short"],
        confidence_multiplier=round(conf_mult, 3),
        size_multiplier=round(size_mult, 3),
        reason=rule["reason"],
    )
