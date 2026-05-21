"""
Layer 3 — Entry Trigger Layer.

Selects the best trigger strategy per category from all candidates.
Only trigger-layer strategies can open trades.

Score: 15-30 points based on trigger confidence and quality.
"""
from __future__ import annotations

from libs.confluence.strategy_registry import (
    StrategyCategory, get_category, is_trigger,
)
from libs.core.models.domain import SignalCandidate


def score_triggers(
    candidates: list[SignalCandidate],
    direction: str,
) -> tuple[float, str, SignalCandidate | None, list[str]]:
    """Score the trigger layer.

    Args:
        candidates: All strategy candidates for this symbol+direction.
        direction: "BUY" or "SELL".

    Returns:
        (score, note, best_trigger_candidate, categories_used)
    """
    # Filter to triggers matching direction
    triggers = [
        c for c in candidates
        if is_trigger(c.strategy_name)
        and c.proposed_action.value == direction
    ]

    if not triggers:
        return 0.0, "no trigger strategies fired", None, []

    def _candidate_confidence(c: SignalCandidate) -> float:
        """Extract confidence from pattern results or default to 0.5."""
        detected = [p for p in c.pattern_results if p.detected]
        if detected:
            return sum(p.confidence for p in detected) / len(detected)
        return 0.5

    # Pick best trigger per category (highest confidence)
    best_per_category: dict[StrategyCategory, SignalCandidate] = {}
    for t in triggers:
        cat = get_category(t.strategy_name)
        existing = best_per_category.get(cat)
        if existing is None or _candidate_confidence(t) > _candidate_confidence(existing):
            best_per_category[cat] = t

    # Primary trigger = highest confidence across all categories
    primary = max(best_per_category.values(), key=_candidate_confidence)
    categories_used = [cat.value for cat in best_per_category]

    # Score: base 15 + confidence-scaled bonus up to 15
    confidence = _candidate_confidence(primary)
    score = 15.0 + (confidence * 15.0)  # 15-30 range
    score = max(15.0, min(30.0, score))

    names = [c.strategy_name for c in best_per_category.values()]
    note = f"triggers: {', '.join(names)} (primary={primary.strategy_name} conf={confidence:.2f})"

    return score, note, primary, categories_used
