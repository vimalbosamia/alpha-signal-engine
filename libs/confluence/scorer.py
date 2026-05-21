"""
Confluence Scorer — combines all 5 layer scores into final confluence score.

Scoring formula:
  base_score = regime + bias + trigger + confirmation + suppression
  diversity_bonus = 5 × (unique_categories - 1)
  final = clamp(base + diversity, 0, 100)

Graduated thresholds based on bot trade count:
  Phase 1 (< 50 trades):  score >= 35, trigger + 1 layer
  Phase 2 (50-200):       score >= 50, trigger + 2 layers
  Phase 3 (200+):         score >= 65, trigger + 3 layers
"""
from __future__ import annotations


def compute_final_score(
    regime_score: float,
    bias_score: float,
    trigger_score: float,
    confirmation_score: float,
    suppression_penalty: float,
    diversity_bonus: float,
) -> float:
    """Compute final confluence score (0-100)."""
    raw = regime_score + bias_score + trigger_score + confirmation_score + suppression_penalty + diversity_bonus
    return max(0.0, min(100.0, raw))


def quality_grade(score: float) -> str:
    """Map score to letter grade."""
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"


def meets_threshold(
    score: float,
    layer_count: int,
    has_trigger: bool,
    trade_count: int = 0,
) -> bool:
    """Check if trade meets graduated threshold requirements.

    Args:
        score: Final confluence score (0-100).
        layer_count: Number of layers that contributed.
        has_trigger: Whether a trigger strategy fired.
        trade_count: Bot's total trade count (determines phase).
    """
    if not has_trigger:
        return False

    if trade_count < 50:
        # Phase 1 — cold start: lenient
        return score >= 35 and layer_count >= 2
    elif trade_count < 200:
        # Phase 2 — learning
        return score >= 50 and layer_count >= 3
    else:
        # Phase 3 — mature
        return score >= 65 and layer_count >= 4
