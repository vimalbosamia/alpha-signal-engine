"""Agent 15: Self-Training Coordinator Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class SelfTrainingCoordinatorAgent(TradingAgent):
    """Reports training phase status and recommends action based on learning maturity."""

    @property
    def name(self) -> str:
        return "self_training_coordinator"

    @property
    def weight(self) -> float:
        return 0.6

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        try:
            from libs.learning.coordinator import get_coordinator
            coord = get_coordinator()
            progress = coord.get_training_progress()

            phase = progress["phase"]
            phase_number = progress["phase_number"]
            total_trades = progress["total_trades"]
            win_rate = progress["win_rate"]
            phase_progress_pct = progress["phase_progress_pct"]

            # Score reflects training maturity
            maturity = min(1.0, phase_number / 8.0)
            score = (maturity - 0.5) * 0.6  # slight bullish bias as maturity grows

            # In early phases, be more cautious
            if phase_number <= 2:
                should_trade = True  # still learning, allow trades for data
                risk_level = 0.6
                reasons = (
                    f"Phase {phase_number}/8: {phase} ({phase_progress_pct:.0f}% complete). "
                    f"Learning mode — {total_trades} trades, {win_rate:.0%} win rate"
                )
            elif phase_number <= 4:
                should_trade = True
                risk_level = 0.4
                reasons = (
                    f"Phase {phase_number}/8: {phase}. "
                    f"Pattern intelligence active — {total_trades} trades"
                )
            else:
                should_trade = True
                risk_level = 0.3
                reasons = (
                    f"Phase {phase_number}/8: {phase}. "
                    f"Advanced learning — {total_trades} trades, {win_rate:.0%} win rate"
                )

            confidence = min(0.7, 0.2 + maturity * 0.5)

            return AgentOutput(
                agent_name=self.name,
                score=self._clamp(score),
                confidence=round(confidence, 3),
                reasoning=reasons,
                signals=progress,
                should_trade=should_trade,
                risk_level=risk_level,
            )

        except Exception as exc:
            return AgentOutput(
                agent_name=self.name,
                score=0.0,
                confidence=0.1,
                reasoning=f"Training coordinator unavailable: {exc}",
                should_trade=True,
                risk_level=0.5,
            )
