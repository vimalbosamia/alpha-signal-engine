"""
Explainability Engine — Structured Decision Trace.

Converts the raw output of each pipeline stage (bias, indicators, structure,
grading) into a human-readable ExplanationTree that captures why a trade
signal was generated, what evidence supported it, and what evidence opposed it.

Design rules:
  - All result objects are immutable (frozen dataclass)
  - No magic numbers — all limits are named constants
  - Never raises — returns a safe ExplanationTree on any valid inputs
  - Accepts None for optional pipeline components (indicator_bias, structure,
    grading_result) so callers do not need to supply every component
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── Limits ────────────────────────────────────────────────────────────────────

_MAX_DOMINANT_FACTORS: int = 3
_MAX_SUPPORTING_FACTORS: int = 5
_MAX_OPPOSING_FACTORS: int = 3

_STRONG_INDICATOR_THRESHOLD: float = 0.5   # strength > this → dominant


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExplanationTree:
    """Structured decision trace for a single trading signal."""

    signal_id: str
    symbol: str
    action: str
    dominant_factors: list[str]      # top 3 reasons FOR the trade
    supporting_factors: list[str]    # weaker supporting evidence
    opposing_factors: list[str]      # reasons AGAINST (but overridden)
    blocked_factors: list[str]       # hard blocks that were triggered
    confidence_breakdown: dict[str, float]   # factor → contribution
    decision_trace: list[str]        # step-by-step pipeline decisions


# ── Engine ────────────────────────────────────────────────────────────────────

class ExplainabilityEngine:
    """
    Builds a structured ExplanationTree from all pipeline components.

    All pipeline arguments except signal_id, symbol, strategy_name, and action
    accept None so callers can omit components that were not run.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        *,
        bias_result,
        indicator_bias,
        structure,
        regime,
        grading_result,
        strategy_name: str,
        action: str,
        signal_id: str,
        symbol: str,
    ) -> ExplanationTree:
        """Build a structured ExplanationTree from all pipeline components."""
        dominant: list[str] = []
        supporting: list[str] = []
        opposing: list[str] = []
        breakdown: dict[str, float] = {}
        trace: list[str] = []

        # ── Bias ──────────────────────────────────────────────────────────────
        self._process_bias(
            bias_result=bias_result,
            action=action,
            dominant=dominant,
            trace=trace,
        )

        # ── Indicator bias ────────────────────────────────────────────────────
        if indicator_bias is not None:
            self._process_indicators(
                indicator_bias=indicator_bias,
                action=action,
                dominant=dominant,
                supporting=supporting,
                opposing=opposing,
                breakdown=breakdown,
                trace=trace,
            )

        # ── Market structure ──────────────────────────────────────────────────
        if structure is not None:
            self._process_structure(
                structure=structure,
                supporting=supporting,
                trace=trace,
            )

        # ── Grading ───────────────────────────────────────────────────────────
        if grading_result is not None:
            trace.append(
                f"Grade: {grading_result.setup_grade} "
                f"({grading_result.quality_score}/100)"
            )

        trace.append(f"Strategy: {strategy_name}")
        trace.append(f"Decision: {action}")

        return ExplanationTree(
            signal_id=signal_id,
            symbol=symbol,
            action=action,
            dominant_factors=dominant[:_MAX_DOMINANT_FACTORS],
            supporting_factors=supporting[:_MAX_SUPPORTING_FACTORS],
            opposing_factors=opposing[:_MAX_OPPOSING_FACTORS],
            blocked_factors=[],
            confidence_breakdown=breakdown,
            decision_trace=trace,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _process_bias(
        self,
        *,
        bias_result,
        action: str,
        dominant: list[str],
        trace: list[str],
    ) -> None:
        """Extract dominant bias factors and add a trace entry."""
        trace.append(
            f"Bias: {bias_result.net_bias} "
            f"(bull={bias_result.bullish_score:.0%}, "
            f"bear={bias_result.bearish_score:.0%})"
        )
        if bias_result.net_bias == "bullish" and action == "BUY":
            dominant.append("Directional bias is bullish")
        elif bias_result.net_bias == "bearish" and action == "SELL":
            dominant.append("Directional bias is bearish")

    def _process_indicators(
        self,
        *,
        indicator_bias,
        action: str,
        dominant: list[str],
        supporting: list[str],
        opposing: list[str],
        breakdown: dict[str, float],
        trace: list[str],
    ) -> None:
        """Classify each indicator as dominant, supporting, or opposing."""
        action_lower = action.lower()

        for ind in indicator_bias.indicators:
            ind_is_aligned = (
                ind.direction == action_lower
                or (ind.direction == "bullish" and action == "BUY")
                or (ind.direction == "bearish" and action == "SELL")
            )

            if ind_is_aligned:
                if ind.strength > _STRONG_INDICATOR_THRESHOLD:
                    dominant.append(f"{ind.name}: {ind.explanation}")
                else:
                    supporting.append(f"{ind.name}: {ind.explanation}")
                breakdown[ind.name] = ind.strength
            elif ind.direction != "neutral":
                opposing.append(f"{ind.name}: {ind.explanation}")

        trace.append(f"Indicators: {indicator_bias.net_bias}")

    def _process_structure(
        self,
        *,
        structure,
        supporting: list[str],
        trace: list[str],
    ) -> None:
        """Add structure trace entry and latest event as supporting evidence."""
        trace.append(
            f"Structure: {structure.trend_bias} "
            f"(strength={structure.strength:.0%})"
        )
        if structure.events:
            latest = structure.events[-1]
            supporting.append(
                f"{latest.kind} {latest.direction} at {latest.price:.4f}"
            )
