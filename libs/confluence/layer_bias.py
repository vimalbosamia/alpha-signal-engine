"""
Layer 2 — Directional Bias Layer.

Computes probability_long and probability_short from:
  - Bull/bear bias engine output
  - MTF alignment (if available among candidates)
  - Trend direction

Does NOT trigger trades. Only adjusts direction confidence.
Score: 0-20 points.
"""
from __future__ import annotations

from libs.analysis.regime.engine import RegimeAnalysis
from libs.analysis.structure.engine import MarketStructure
from libs.core.models.domain import TrendDirection


def score_bias(
    bias_data: dict | None,
    structure: MarketStructure | None,
    regime: RegimeAnalysis | None,
    direction: str,
) -> tuple[float, float, float, str]:
    """Score the directional bias layer.

    Args:
        bias_data: Output from BullBearBiasEngine (dict with bullish/bearish/net keys).
        structure: Market structure analysis.
        regime: Regime classification.
        direction: "BUY" or "SELL".

    Returns:
        (bias_score, probability_long, probability_short, note)
    """
    prob_long = 0.5
    prob_short = 0.5
    score = 10.0  # neutral default
    notes: list[str] = []

    # ── Bias engine contribution (strongest signal) ──
    if bias_data:
        bullish = bias_data.get("bullish", 0.0)
        bearish = bias_data.get("bearish", 0.0)
        net = bias_data.get("net", "neutral")

        prob_long = max(0.0, min(1.0, 0.5 + (bullish - bearish)))
        prob_short = 1.0 - prob_long

        # Score based on alignment with trade direction
        if direction == "BUY" and net == "bullish":
            score += bullish * 8.0  # up to +8
            notes.append(f"bias bullish ({bullish:.0%}) aligned with BUY")
        elif direction == "SELL" and net == "bearish":
            score += bearish * 8.0
            notes.append(f"bias bearish ({bearish:.0%}) aligned with SELL")
        elif net == "neutral":
            notes.append("bias neutral — no adjustment")
        else:
            # Counter-bias: penalize
            penalty = max(bullish, bearish) * 6.0
            score -= penalty
            notes.append(f"bias {net} AGAINST {direction} — penalty {penalty:.1f}")

    # ── Structure trend alignment (secondary) ──
    if structure and structure.trend:
        if direction == "BUY" and structure.trend == TrendDirection.UPTREND:
            score += 2.0
            notes.append("structure uptrend confirms BUY")
        elif direction == "SELL" and structure.trend == TrendDirection.DOWNTREND:
            score += 2.0
            notes.append("structure downtrend confirms SELL")
        elif structure.trend == TrendDirection.RANGING:
            notes.append("ranging — no trend bias")
        else:
            score -= 1.5
            notes.append("counter-trend structure")

    # ── Regime alignment (tertiary) ──
    if regime:
        regime_name = regime.name.lower() if hasattr(regime, "name") else "unknown"
        if direction == "BUY" and "up" in regime_name:
            score += 1.0
        elif direction == "SELL" and "down" in regime_name:
            score += 1.0

    score = max(0.0, min(20.0, score))
    return score, prob_long, prob_short, "; ".join(notes) if notes else "no bias data"
