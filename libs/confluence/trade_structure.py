"""
ConfluenceTradeResult — the enriched multi-layer trade structure.

Every trade produced by the confluence engine carries full provenance:
which layers contributed, scores, confirmations, suppressions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConfluenceTradeResult:
    """Institutional-grade multi-strategy trade result."""

    # Core
    symbol: str
    direction: str                          # "BUY" or "SELL"

    # Layer 1 — Regime
    market_regime: str = "unknown"
    regime_score: float = 0.0               # 0-15

    # Layer 2 — Bias
    bias_direction: str = "neutral"         # "bullish", "bearish", "neutral"
    probability_long: float = 0.5
    probability_short: float = 0.5
    bias_score: float = 0.0                 # 0-20

    # Layer 3 — Entry Trigger
    primary_strategy: str = ""
    trigger_confidence: float = 0.0
    trigger_score: float = 0.0              # 15-30

    # Layer 4 — Confirmations
    supporting_strategies: tuple[str, ...] = ()
    confirmation_signals: tuple[str, ...] = ()
    confirmation_score: float = 0.0         # 0-10 each, stacked

    # Layer 5 — Suppressions
    suppression_signals: tuple[str, ...] = ()
    suppression_penalty: float = 0.0        # negative

    # Scoring
    diversity_bonus: float = 0.0
    confluence_score: float = 0.0           # 0-100 final
    layer_count: int = 0
    categories_represented: tuple[str, ...] = ()

    # Quality classification
    trade_quality: str = "D"                # A/B/C/D

    # Entry/exit from primary trigger
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float | None = None
    risk_reward: float = 0.0

    # Metadata
    factor_notes: dict[str, str] = field(default_factory=dict)

    @property
    def passed_threshold(self) -> bool:
        """Check against minimum score. Called by engine with trade_count context."""
        return self.confluence_score >= 35 and self.layer_count >= 2

    def quality_grade(self, score: float) -> str:
        if score >= 80:
            return "A"
        if score >= 60:
            return "B"
        if score >= 40:
            return "C"
        return "D"
