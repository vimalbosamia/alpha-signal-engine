"""
Market Sentiment Filter.

Fetches Fear & Greed Index (crypto) and assesses volatility-based conditions
to adjust trade confidence. Extreme sentiment or volatility reduces confidence
and may flag position-size reduction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_FNG_URL = "https://api.alternative.me/fng/?limit=1"

# ── Fear & Greed thresholds ────────────────────────────────────────────────────
_EXTREME_FEAR_MAX = 20
_FEAR_MAX = 35
_NEUTRAL_MAX = 65
_GREED_MAX = 80
# > 80 → extreme_greed

# ── Volatility thresholds (ATR as % of price) ─────────────────────────────────
_HIGH_VOL_MIN = 2.0
_EXTREME_VOL_MIN = 4.0

# ── Confidence adjustments ────────────────────────────────────────────────────
_ADJ_EXTREME_FEAR = -0.10
_ADJ_FEAR = -0.05
_ADJ_NEUTRAL = 0.0
_ADJ_GREED = -0.03
_ADJ_EXTREME_GREED = -0.08

_ADJ_HIGH_VOL = -0.03
_ADJ_EXTREME_VOL = -0.08


@dataclass(frozen=True)
class SentimentResult:
    """Immutable snapshot of assessed market sentiment."""

    fear_greed_value: int
    """Raw Fear & Greed index value, 0–100."""

    fear_greed_label: str
    """One of: extreme_fear, fear, neutral, greed, extreme_greed."""

    volatility_level: str
    """One of: low, normal, high, extreme."""

    confidence_adjustment: float
    """Signed delta applied to signal confidence. Range: -0.15 to +0.05."""

    should_reduce_size: bool
    """True when extreme conditions warrant reducing position size."""

    explanation: str
    """Human-readable summary of the assessment."""


def _classify_fear_greed(value: int) -> tuple[str, float, bool]:
    """Return (label, confidence_adjustment, should_reduce_size) for a F&G value."""
    if value <= _EXTREME_FEAR_MAX:
        return "extreme_fear", _ADJ_EXTREME_FEAR, True
    if value <= _FEAR_MAX:
        return "fear", _ADJ_FEAR, False
    if value <= _NEUTRAL_MAX:
        return "neutral", _ADJ_NEUTRAL, False
    if value <= _GREED_MAX:
        return "greed", _ADJ_GREED, False
    return "extreme_greed", _ADJ_EXTREME_GREED, True


def _classify_volatility(volatility_pct: float) -> tuple[str, float, bool]:
    """Return (level, confidence_adjustment, should_reduce_size) for a volatility %."""
    if volatility_pct >= _EXTREME_VOL_MIN:
        return "extreme", _ADJ_EXTREME_VOL, True
    if volatility_pct >= _HIGH_VOL_MIN:
        return "high", _ADJ_HIGH_VOL, False
    if volatility_pct > 0:
        return "normal", 0.0, False
    return "low", 0.0, False


def _build_explanation(
    fg_label: str,
    vol_level: str,
    total_adj: float,
    reduce_size: bool,
) -> str:
    """Compose a concise explanation string."""
    parts: list[str] = []

    fg_phrases = {
        "extreme_fear": "panic — reduce exposure",
        "fear": "cautious sentiment",
        "neutral": "neutral sentiment",
        "greed": "overheated sentiment",
        "extreme_greed": "euphoria — reversal risk",
    }
    parts.append(fg_phrases.get(fg_label, fg_label))

    if vol_level in {"high", "extreme"}:
        parts.append(f"{vol_level} volatility")

    adj_pct = round(total_adj * 100)
    if adj_pct < 0:
        parts.append(f"confidence {adj_pct}%")
    elif adj_pct == 0:
        parts.append("no confidence change")

    if reduce_size:
        parts.append("reduce size")

    return "; ".join(parts)


class SentimentFilter:
    """Assesses market sentiment from Fear & Greed index and price volatility."""

    def assess(self, fear_greed_value: int, volatility_pct: float) -> SentimentResult:
        """Assess market sentiment and return a confidence adjustment.

        Args:
            fear_greed_value: 0-100 from the Fear & Greed API (default 50 on failure).
            volatility_pct: recent ATR as a percentage of price (e.g. 1.5 = 1.5%).

        Returns:
            SentimentResult with signed confidence_adjustment clamped to [-0.15, 0.05].
        """
        fg_label, fg_adj, fg_reduce = _classify_fear_greed(fear_greed_value)
        vol_level, vol_adj, vol_reduce = _classify_volatility(volatility_pct)

        raw_adjustment = fg_adj + vol_adj
        # Clamp to documented range
        clamped_adjustment = max(-0.15, min(0.05, raw_adjustment))

        reduce_size = fg_reduce or vol_reduce

        explanation = _build_explanation(
            fg_label, vol_level, clamped_adjustment, reduce_size
        )

        return SentimentResult(
            fear_greed_value=fear_greed_value,
            fear_greed_label=fg_label,
            volatility_level=vol_level,
            confidence_adjustment=clamped_adjustment,
            should_reduce_size=reduce_size,
            explanation=explanation,
        )

    async def fetch_fear_greed(self) -> int:
        """Fetch the current Fear & Greed index from alternative.me.

        Returns:
            Integer 0-100. Returns 50 (neutral) on any failure.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(_FNG_URL)
                response.raise_for_status()
                payload = response.json()
                raw_value = payload["data"][0]["value"]
                value = int(raw_value)
                if not (0 <= value <= 100):
                    logger.warning(
                        "fear_greed_out_of_range",
                        extra={"value": value},
                    )
                    return 50
                return value
        except Exception:
            logger.warning("fetch_fear_greed_failed", exc_info=True)
            return 50
