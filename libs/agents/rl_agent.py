"""Agent 8: Reinforcement Learning Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class ReinforcementLearningAgent(TradingAgent):
    """Uses Q-learning engine recommendations for trade decisions."""

    @property
    def name(self) -> str:
        return "reinforcement_learning"

    @property
    def weight(self) -> float:
        return 1.0

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        regime = str(self._safe_get(context, "regime", "unknown"))
        atr_pct = float(self._safe_get(context, "atr_pct", 0.0))
        ema_alignment = float(self._safe_get(context, "ema_alignment", 1.0))

        try:
            from libs.learning.rl_engine import get_rl_engine
            engine = get_rl_engine()
            rec = engine.get_policy_recommendation(regime, atr_pct, ema_alignment)

            q_value = rec["q_value"]
            confidence = rec["confidence"]
            strategy = rec["strategy"]
            size = rec["size_bucket"]
            visits = rec["state_visits"]

            # Score based on Q-value sign and magnitude
            score = self._clamp(q_value / 3.0)  # normalize Q to [-1, 1] range

            should_trade = size != "skip" and confidence > 0.1
            reasoning = (
                f"Q-rec: {strategy}/{size} (Q={q_value:.3f}, "
                f"visits={visits}, ε={rec['exploration_rate']:.3f})"
            )

            return AgentOutput(
                agent_name=self.name,
                score=score,
                confidence=min(0.9, confidence),
                reasoning=reasoning,
                signals=rec,
                should_trade=should_trade,
                risk_level=0.5 if confidence < 0.3 else 0.3,
            )

        except Exception as exc:
            return AgentOutput(
                agent_name=self.name,
                score=0.0,
                confidence=0.1,
                reasoning=f"RL engine unavailable: {exc}",
                should_trade=True,
                risk_level=0.5,
            )
