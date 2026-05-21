"""Agent 12: Memory & Similarity Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class MemorySimilarityAgent(TradingAgent):
    """Uses vector memory to compare current conditions against historical trades."""

    @property
    def name(self) -> str:
        return "memory_similarity"

    @property
    def weight(self) -> float:
        return 1.1

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        try:
            from libs.learning.vector_memory import get_vector_memory
            memory = get_vector_memory()

            if memory.size == 0:
                return AgentOutput(
                    agent_name=self.name,
                    score=0.0,
                    confidence=0.1,
                    reasoning="No historical memory — insufficient data",
                    should_trade=True,
                    risk_level=0.5,
                )

            stats = memory.get_win_rate_for_similar(
                context=context,
                k=10,
                min_similarity=0.5,
            )

            win_rate = stats["win_rate"]
            avg_pnl = stats["avg_pnl"]
            sample_size = stats["sample_size"]
            avg_similarity = stats["avg_similarity"]
            historical_edge = stats["historical_edge"]

            if sample_size == 0:
                return AgentOutput(
                    agent_name=self.name,
                    score=0.0,
                    confidence=0.15,
                    reasoning="No similar historical conditions found",
                    should_trade=True,
                    risk_level=0.5,
                )

            # Score based on historical edge
            score = (historical_edge - 0.5) * 2.0  # map 0-1 to -1 to +1
            score = self._clamp(score)

            # Confidence scales with sample size and similarity quality
            confidence = min(0.85, 0.2 + sample_size * 0.05 + avg_similarity * 0.2)

            reasons = (
                f"Historical: {win_rate:.0%} win rate over {sample_size} "
                f"similar trades (avg sim={avg_similarity:.2f}, edge={historical_edge:.2f})"
            )

            should_trade = win_rate > 0.4 or sample_size < 3  # don't block on thin data

            return AgentOutput(
                agent_name=self.name,
                score=score,
                confidence=round(confidence, 3),
                reasoning=reasons,
                signals=stats,
                should_trade=should_trade,
                risk_level=0.3 if win_rate > 0.55 else 0.6,
            )

        except Exception as exc:
            return AgentOutput(
                agent_name=self.name,
                score=0.0,
                confidence=0.1,
                reasoning=f"Memory engine unavailable: {exc}",
                should_trade=True,
                risk_level=0.5,
            )
