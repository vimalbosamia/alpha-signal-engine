"""
Layer 1 — Market Regime Layer.

Classifies the current regime and assigns:
  - regime score (0-15 points)
  - strategy permissions (which strategies allowed)
  - hard blocks for dangerous regimes
"""
from __future__ import annotations

from libs.analysis.regime.engine import RegimeAnalysis

# ── Regime → score mapping (0-15) ────────────────────────────────────────────
_REGIME_SCORES: dict[str, float] = {
    # Strong trending = high score
    "trending_up":       13.0,
    "trending_down":     13.0,
    "breakout":          14.0,
    "expansion":         12.0,
    "momentum_ignition": 11.0,

    # Ranging = moderate
    "ranging_low_vol":   10.0,
    "ranging_high_vol":   8.0,
    "mean_reversion":    10.0,
    "compression":       11.0,

    # Accumulation/distribution = moderate
    "accumulation":       9.0,
    "distribution":       9.0,

    # Choppy/low liquidity = low
    "low_liquidity":      5.0,
    "choppy":             4.0,

    # Dangerous = very low or blocked
    "reversal":           7.0,
    "news_driven":        3.0,
    "climactic":          0.0,    # hard block
    "panic_selloff":      0.0,    # hard block
    "liquidation_event":  0.0,    # hard block
    "exhaustion":         5.0,

    "unknown":            6.0,
}

# Regimes that hard-block all trading
_BLOCKED_REGIMES: frozenset[str] = frozenset({
    "climactic", "panic_selloff", "liquidation_event",
})


def score_regime(regime: RegimeAnalysis | None) -> tuple[float, str, bool]:
    """Score the regime layer.

    Returns:
        (score, note, is_blocked)
    """
    if regime is None:
        return 6.0, "no regime data — default score", False

    name = regime.name.lower() if hasattr(regime, "name") else "unknown"
    is_blocked = name in _BLOCKED_REGIMES
    score = _REGIME_SCORES.get(name, 6.0)

    if is_blocked:
        return 0.0, f"regime {name} — BLOCKED", True

    return score, f"regime={name} score={score}", False


def get_regime_name(regime: RegimeAnalysis | None) -> str:
    if regime is None:
        return "unknown"
    return regime.name.lower() if hasattr(regime, "name") else "unknown"
