"""
FeatureCorrelationReducer — identifies redundant technical indicators.

Uses domain knowledge (not computed correlation matrices) to classify
indicators into groups and flag those that would inflate signal confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorrelationGroup:
    name: str                  # "momentum", "trend", "volatility", "volume"
    indicators: list[str]      # all members of the group
    representative: str        # the one canonical indicator to keep
    redundant: list[str]       # indicators to discount (not zero — 30% weight)


@dataclass(frozen=True)
class CorrelationAudit:
    groups: list[CorrelationGroup]
    total_indicators: int
    kept_indicators: int
    removed_indicators: int
    inflation_risk: str        # "low" | "moderate" | "high"
    explanation: str


class FeatureCorrelationReducer:
    """Identifies redundant indicators that inflate confidence scores."""

    GROUPS: list[CorrelationGroup] = [
        CorrelationGroup(
            name="momentum",
            indicators=["rsi", "stoch_rsi_k", "stoch_rsi_d", "macd_histogram", "roc"],
            representative="rsi",
            redundant=["stoch_rsi_k", "stoch_rsi_d", "roc"],
        ),
        CorrelationGroup(
            name="trend",
            indicators=["ema_9", "ema_20", "ema_50", "sma_20", "sma_50", "sma_200"],
            representative="ema_20",
            redundant=["sma_20", "sma_50"],
        ),
        CorrelationGroup(
            name="volatility",
            indicators=["atr", "bb_width", "bb_pct_b"],
            representative="atr",
            redundant=[],
        ),
        CorrelationGroup(
            name="volume",
            indicators=["relative_volume", "vwap"],
            representative="relative_volume",
            redundant=[],
        ),
    ]

    def audit(self) -> CorrelationAudit:
        """Return analysis of indicator redundancy across all known groups."""
        total = sum(len(g.indicators) for g in self.GROUPS)
        removed = sum(len(g.redundant) for g in self.GROUPS)
        kept = total - removed

        if removed > 3:
            risk = "high"
        elif removed > 1:
            risk = "moderate"
        else:
            risk = "low"

        return CorrelationAudit(
            groups=list(self.GROUPS),
            total_indicators=total,
            kept_indicators=kept,
            removed_indicators=removed,
            inflation_risk=risk,
            explanation=(
                f"{removed} redundant indicators can inflate confidence by ~{removed * 5}%"
            ),
        )

    def should_discount(self, indicator_name: str) -> float:
        """Return weight factor for an indicator.

        Returns:
            1.0  — representative or unknown indicator (keep full weight)
            0.3  — redundant indicator (still has some unique info, not zero)
        """
        for group in self.GROUPS:
            if indicator_name in group.redundant:
                return 0.3  # Count at 30% — still has marginal unique info
            if indicator_name == group.representative:
                return 1.0
        return 1.0  # Unknown indicator — keep full weight
