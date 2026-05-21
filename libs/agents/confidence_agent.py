"""Agent 13: Confidence Calibration Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class ConfidenceCalibrationAgent(TradingAgent):
    """Calibrates signal confidence using historical accuracy and multi-source agreement."""

    @property
    def name(self) -> str:
        return "confidence_calibration"

    @property
    def weight(self) -> float:
        return 1.0

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        raw_confidence = float(self._safe_get(context, "confidence", 0.5))
        regime = str(self._safe_get(context, "regime", "unknown"))
        htf_alignment = bool(self._safe_get(context, "htf_alignment", False))
        pattern_count = int(self._safe_get(context, "pattern_count", 0))
        strategy = str(self._safe_get(context, "strategy", ""))

        # Historical accuracy lookup (best-effort)
        historical_accuracy = 0.5
        try:
            from libs.learning.pattern_scorer import get_pattern_store
            store = get_pattern_store()
            all_stats = store.get_all_stats()
            if all_stats:
                total_wins = sum(s.get("wins", 0) for s in all_stats.values())
                total_losses = sum(s.get("losses", 0) for s in all_stats.values())
                total = total_wins + total_losses
                if total > 10:
                    historical_accuracy = total_wins / total
        except Exception:
            pass

        # Calibration: adjust raw confidence based on historical accuracy
        calibrated = raw_confidence

        # Pull toward historical accuracy
        if historical_accuracy != 0.5:
            calibrated = raw_confidence * 0.6 + historical_accuracy * 0.4

        # Boost for confirming factors
        if htf_alignment:
            calibrated = min(1.0, calibrated + 0.1)
        if pattern_count >= 3:
            calibrated = min(1.0, calibrated + 0.05)

        # Penalize overconfidence in uncertain regimes
        if regime in ("unknown", "news_driven", "climactic"):
            calibrated *= 0.7

        calibrated = max(0.0, min(1.0, calibrated))
        score = (calibrated - 0.5) * 2.0  # map to [-1, 1]
        score = self._clamp(score)

        overconfidence_gap = raw_confidence - calibrated
        reasons: list[str] = []
        if overconfidence_gap > 0.1:
            reasons.append(f"Overconfident by {overconfidence_gap:.2f}")
        elif overconfidence_gap < -0.1:
            reasons.append(f"Underconfident by {abs(overconfidence_gap):.2f}")

        reasons.append(f"Raw={raw_confidence:.2f} → Calibrated={calibrated:.2f}")

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=calibrated,
            reasoning="; ".join(reasons),
            signals={
                "raw_confidence": raw_confidence,
                "calibrated_confidence": calibrated,
                "historical_accuracy": historical_accuracy,
                "adjustment": round(overconfidence_gap, 4),
            },
            should_trade=calibrated > 0.35,
            risk_level=0.3 if calibrated > 0.6 else 0.6,
        )
