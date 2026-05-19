"""ConfidenceCalibrator — adjusts raw confluence confidence using historical stats."""
from __future__ import annotations

from dataclasses import dataclass, field

MIN_TRADES_FOR_VALIDATION = 10
UNVALIDATED_CAP = 0.60
MIN_TRADES_FOR_BOOST = 30


@dataclass(frozen=True)
class CalibrationResult:
    raw_confidence: float
    calibrated_confidence: float
    validation_status: str  # "unvalidated", "learning", "validated"
    adjustments: list[str] = field(default_factory=list)
    explanation: str = ""


class ConfidenceCalibrator:
    """Adjusts confidence based on historical strategy/symbol/regime performance."""

    def calibrate(
        self,
        raw_confidence: float,
        strategy_name: str,
        strategy_win_rate: float | None,
        strategy_trade_count: int,
        symbol_win_rate: float | None,
        regime_win_rate: float | None,
        conflict_score: float = 0.0,
    ) -> CalibrationResult:
        confidence = raw_confidence
        adjustments: list[str] = []

        # Validation status
        if strategy_trade_count < MIN_TRADES_FOR_VALIDATION:
            status = "unvalidated"
            confidence = min(confidence, UNVALIDATED_CAP)
            adjustments.append(f"Unvalidated ({strategy_trade_count} trades) — capped at {UNVALIDATED_CAP:.0%}")
        elif strategy_trade_count < MIN_TRADES_FOR_BOOST:
            status = "learning"
        else:
            status = "validated"

        # Strategy win rate adjustment
        if strategy_win_rate is not None and strategy_trade_count >= MIN_TRADES_FOR_VALIDATION:
            if strategy_win_rate >= 0.60:
                boost = (strategy_win_rate - 0.50) * 0.2
                confidence = min(1.0, confidence + boost)
                adjustments.append(f"Strategy WR {strategy_win_rate:.0%} — +{boost:.0%}")
            elif strategy_win_rate < 0.40:
                penalty = (0.50 - strategy_win_rate) * 0.3
                confidence = max(0.1, confidence - penalty)
                adjustments.append(f"Poor strategy WR {strategy_win_rate:.0%} — reduced by {penalty:.0%}")

        # Symbol-specific
        if symbol_win_rate is not None:
            if symbol_win_rate < 0.35:
                confidence = max(0.1, confidence - 0.05)
                adjustments.append(f"Poor symbol WR {symbol_win_rate:.0%} — -5%")
            elif symbol_win_rate >= 0.65:
                confidence = min(1.0, confidence + 0.03)
                adjustments.append(f"Strong symbol WR {symbol_win_rate:.0%} — +3%")

        # Regime-specific
        if regime_win_rate is not None:
            if regime_win_rate < 0.35:
                confidence = max(0.1, confidence - 0.05)
                adjustments.append(f"Poor regime WR {regime_win_rate:.0%} — -5%")

        # Conflict penalty
        if conflict_score > 0.6:
            penalty = (conflict_score - 0.5) * 0.15
            confidence = max(0.1, confidence - penalty)
            adjustments.append(f"High conflict ({conflict_score:.0%}) — -{penalty:.0%}")

        confidence = round(max(0.0, min(1.0, confidence)), 4)

        explanation = f"Raw: {raw_confidence:.0%} → Calibrated: {confidence:.0%}"
        if adjustments:
            explanation += ". " + ". ".join(adjustments)

        return CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=confidence,
            validation_status=status,
            adjustments=adjustments,
            explanation=explanation,
        )
