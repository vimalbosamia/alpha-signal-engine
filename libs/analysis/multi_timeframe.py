"""
libs.analysis.multi_timeframe — Multi-Timeframe Alignment Scoring.

Analyzes trend direction across multiple timeframes and computes a
unified alignment score.  Higher timeframes carry stronger weight
(weekly > daily > 4h > 1h > 15m > 5m).

Design rules:
  - All results are immutable (frozen dataclass)
  - Higher timeframe dominance — weighted by TF rank
  - Alignment score: 1.0 = all TFs agree, 0.0 = full disagreement
  - No magic numbers — all thresholds are named constants
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from libs.core.logging.logger import get_logger

_log = get_logger(__name__)


# ── Timeframe weights (higher TF = more weight) ─────────────────────────────

TF_WEIGHTS: dict[str, float] = {
    "1m": 0.05,
    "3m": 0.08,
    "5m": 0.10,
    "15m": 0.15,
    "30m": 0.18,
    "1h": 0.25,
    "4h": 0.40,
    "1d": 0.60,
    "1w": 0.80,
}

# Trend encoding for scoring
TREND_VALUES: dict[str, float] = {
    "strong_up": 1.0,
    "up": 0.5,
    "uptrend": 0.5,
    "flat": 0.0,
    "ranging": 0.0,
    "unknown": 0.0,
    "down": -0.5,
    "downtrend": -0.5,
    "strong_down": -1.0,
}


@dataclass(frozen=True)
class TimeframeSignal:
    """Trend signal from a single timeframe."""
    timeframe: str
    trend: str              # up/down/flat/strong_up/strong_down
    regime: str = "unknown"
    rsi: float = 50.0
    ema_alignment: float = 1.0
    weight: float = 1.0


@dataclass(frozen=True)
class MTFAlignment:
    """Multi-timeframe alignment result."""
    alignment_score: float       # 0.0 (full divergence) to 1.0 (full agreement)
    direction_score: float       # -1.0 (bearish) to +1.0 (bullish)
    dominant_trend: str          # trend from highest weighted TF
    htf_agrees: bool             # do 4h+ timeframes agree?
    ltf_agrees: bool             # do 15m and below agree?
    stacked_bullish: bool        # all TFs bullish
    stacked_bearish: bool        # all TFs bearish
    divergence_detected: bool    # HTF vs LTF disagree
    signals: list[TimeframeSignal] = field(default_factory=list)
    notes: str = ""


def compute_alignment(tf_data: list[dict[str, Any]]) -> MTFAlignment:
    """Compute multi-timeframe alignment from a list of per-TF data.

    Parameters
    ----------
    tf_data: List of dicts, each with keys:
        timeframe, trend, regime (optional), rsi (optional), ema_alignment (optional)

    Returns
    -------
    MTFAlignment with scoring and divergence detection.
    """
    if not tf_data:
        return MTFAlignment(
            alignment_score=0.0,
            direction_score=0.0,
            dominant_trend="unknown",
            htf_agrees=False,
            ltf_agrees=False,
            stacked_bullish=False,
            stacked_bearish=False,
            divergence_detected=False,
            notes="No timeframe data provided",
        )

    signals: list[TimeframeSignal] = []
    for d in tf_data:
        tf = d.get("timeframe", "1h")
        signals.append(TimeframeSignal(
            timeframe=tf,
            trend=d.get("trend", "unknown"),
            regime=d.get("regime", "unknown"),
            rsi=d.get("rsi", 50.0),
            ema_alignment=d.get("ema_alignment", 1.0),
            weight=TF_WEIGHTS.get(tf, 0.15),
        ))

    # Weighted direction score
    total_weight = sum(s.weight for s in signals)
    if total_weight == 0:
        total_weight = 1.0

    weighted_direction = sum(
        TREND_VALUES.get(s.trend, 0.0) * s.weight
        for s in signals
    ) / total_weight

    direction_score = max(-1.0, min(1.0, weighted_direction))

    # Alignment score: how much do all TFs agree?
    trend_values = [TREND_VALUES.get(s.trend, 0.0) for s in signals]
    if len(trend_values) > 1:
        mean_trend = sum(trend_values) / len(trend_values)
        variance = sum((t - mean_trend) ** 2 for t in trend_values) / len(trend_values)
        # Max variance for [-1, 1] values is 1.0
        alignment_score = max(0.0, 1.0 - variance)
    else:
        alignment_score = 1.0

    # HTF analysis (4h+)
    htf_signals = [s for s in signals if TF_WEIGHTS.get(s.timeframe, 0) >= 0.40]
    htf_trends = [TREND_VALUES.get(s.trend, 0.0) for s in htf_signals]
    htf_agrees = len(htf_trends) > 0 and all(
        (t > 0) == (htf_trends[0] > 0) for t in htf_trends if t != 0
    )

    # LTF analysis (15m and below)
    ltf_signals = [s for s in signals if TF_WEIGHTS.get(s.timeframe, 0) <= 0.15]
    ltf_trends = [TREND_VALUES.get(s.trend, 0.0) for s in ltf_signals]
    ltf_agrees = len(ltf_trends) > 0 and all(
        (t > 0) == (ltf_trends[0] > 0) for t in ltf_trends if t != 0
    )

    # Stacked trends
    stacked_bullish = all(TREND_VALUES.get(s.trend, 0.0) > 0 for s in signals)
    stacked_bearish = all(TREND_VALUES.get(s.trend, 0.0) < 0 for s in signals)

    # Divergence: HTF and LTF disagree in direction
    htf_avg = sum(htf_trends) / len(htf_trends) if htf_trends else 0.0
    ltf_avg = sum(ltf_trends) / len(ltf_trends) if ltf_trends else 0.0
    divergence_detected = (htf_avg > 0.1 and ltf_avg < -0.1) or (htf_avg < -0.1 and ltf_avg > 0.1)

    # Dominant trend from highest weighted TF
    sorted_by_weight = sorted(signals, key=lambda s: s.weight, reverse=True)
    dominant_trend = sorted_by_weight[0].trend if sorted_by_weight else "unknown"

    # Notes
    notes_parts: list[str] = []
    if stacked_bullish:
        notes_parts.append("All TFs bullish — strong trend stacking")
    elif stacked_bearish:
        notes_parts.append("All TFs bearish — strong trend stacking")
    if divergence_detected:
        notes_parts.append("HTF/LTF divergence — caution")
    if not htf_agrees and htf_signals:
        notes_parts.append("HTF trend conflict")

    return MTFAlignment(
        alignment_score=round(alignment_score, 4),
        direction_score=round(direction_score, 4),
        dominant_trend=dominant_trend,
        htf_agrees=htf_agrees,
        ltf_agrees=ltf_agrees,
        stacked_bullish=stacked_bullish,
        stacked_bearish=stacked_bearish,
        divergence_detected=divergence_detected,
        signals=signals,
        notes="; ".join(notes_parts) or "Mixed timeframe signals",
    )
