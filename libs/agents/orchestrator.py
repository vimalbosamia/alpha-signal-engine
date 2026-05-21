"""
libs.agents.orchestrator — Multi-agent consensus engine.

Collects outputs from all 15 trading agents, computes weighted consensus,
and produces a final recommendation with full transparency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from libs.agents.base import AgentOutput, TradingAgent
from libs.core.logging.logger import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class ConsensusResult:
    """Immutable result of multi-agent consensus."""

    consensus_score: float         # weighted score [-1, 1]
    consensus_confidence: float    # weighted confidence [0, 1]
    should_trade: bool             # majority vote on trading
    risk_level: float              # weighted average risk [0, 1]
    bullish_count: int
    bearish_count: int
    neutral_count: int
    blocker_count: int             # agents that said should_trade=False
    agent_outputs: list[AgentOutput] = field(default_factory=list)
    agent_agreement: float = 0.0   # 0-1 how much agents agree
    dominant_signal: str = "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "consensus_score": round(self.consensus_score, 4),
            "consensus_confidence": round(self.consensus_confidence, 4),
            "should_trade": self.should_trade,
            "risk_level": round(self.risk_level, 4),
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "blocker_count": self.blocker_count,
            "agent_agreement": round(self.agent_agreement, 4),
            "dominant_signal": self.dominant_signal,
            "agents": [
                {
                    "name": a.agent_name,
                    "score": round(a.score, 3),
                    "confidence": round(a.confidence, 3),
                    "should_trade": a.should_trade,
                    "reasoning": a.reasoning,
                }
                for a in self.agent_outputs
            ],
        }


class AgentOrchestrator:
    """Orchestrates all 15 trading agents and computes weighted consensus.

    Parameters
    ----------
    agents: List of TradingAgent instances. If None, initializes all 15 agents.
    blocker_threshold: Minimum fraction of agents that must agree to trade (default 0.5).
    """

    BLOCKER_THRESHOLD: float = 0.5

    def __init__(
        self,
        agents: list[TradingAgent] | None = None,
        blocker_threshold: float = BLOCKER_THRESHOLD,
    ) -> None:
        self._agents = agents or self._default_agents()
        self._blocker_threshold = blocker_threshold

    @staticmethod
    def _default_agents() -> list[TradingAgent]:
        """Initialize all 15 default trading agents."""
        from libs.agents.regime_agent import MarketRegimeAgent
        from libs.agents.technical_agent import TechnicalAnalysisAgent
        from libs.agents.pattern_agent import PatternRecognitionAgent
        from libs.agents.volume_agent import VolumeLiquidityAgent
        from libs.agents.macro_agent import MacroNewsAgent
        from libs.agents.sentiment_agent import SentimentAgent
        from libs.agents.strategy_agent import StrategySelectionAgent
        from libs.agents.rl_agent import ReinforcementLearningAgent
        from libs.agents.risk_agent import RiskManagementAgent
        from libs.agents.portfolio_agent import PortfolioIntelligenceAgent
        from libs.agents.execution_agent import ExecutionOptimizationAgent
        from libs.agents.memory_agent import MemorySimilarityAgent
        from libs.agents.confidence_agent import ConfidenceCalibrationAgent
        from libs.agents.performance_agent import PerformanceAnalyticsAgent
        from libs.agents.training_agent import SelfTrainingCoordinatorAgent

        return [
            MarketRegimeAgent(),
            TechnicalAnalysisAgent(),
            PatternRecognitionAgent(),
            VolumeLiquidityAgent(),
            MacroNewsAgent(),
            SentimentAgent(),
            StrategySelectionAgent(),
            ReinforcementLearningAgent(),
            RiskManagementAgent(),
            PortfolioIntelligenceAgent(),
            ExecutionOptimizationAgent(),
            MemorySimilarityAgent(),
            ConfidenceCalibrationAgent(),
            PerformanceAnalyticsAgent(),
            SelfTrainingCoordinatorAgent(),
        ]

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    def analyze(self, context: dict[str, Any]) -> ConsensusResult:
        """Run all agents and compute weighted consensus.

        Parameters
        ----------
        context: Shared market context dict. Each agent extracts what it needs.

        Returns
        -------
        ConsensusResult with weighted scores and full agent breakdown.
        """
        outputs: list[AgentOutput] = []

        for agent in self._agents:
            try:
                output = agent.analyze(context)
                outputs.append(output)
            except Exception as exc:
                _log.warning(
                    "orchestrator.agent_error",
                    agent=agent.name,
                    error=str(exc),
                )
                outputs.append(AgentOutput(
                    agent_name=agent.name,
                    score=0.0,
                    confidence=0.0,
                    reasoning=f"Error: {exc}",
                    should_trade=True,
                    risk_level=0.5,
                ))

        return self._compute_consensus(outputs)

    def _compute_consensus(self, outputs: list[AgentOutput]) -> ConsensusResult:
        """Compute weighted consensus from all agent outputs."""
        if not outputs:
            return ConsensusResult(
                consensus_score=0.0,
                consensus_confidence=0.0,
                should_trade=False,
                risk_level=1.0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                blocker_count=0,
                agent_outputs=[],
            )

        # Get weights: learned weights override static defaults
        weight_map: dict[str, float] = {
            agent.name: agent.weight for agent in self._agents
        }
        try:
            from libs.learning.agent_weight_tuner import get_weight_tuner
            tuner = get_weight_tuner()
            learned = tuner.get_all_weights()
            for name, w in learned.items():
                if name in weight_map:
                    weight_map[name] = w
        except Exception:
            pass

        # Weighted score and confidence
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        weighted_risk = 0.0

        bullish = 0
        bearish = 0
        neutral = 0
        blockers = 0

        for output in outputs:
            w = weight_map.get(output.agent_name, 1.0) * output.confidence
            total_weight += w
            weighted_score += output.score * w
            weighted_confidence += output.confidence * w
            weighted_risk += output.risk_level * w

            if output.score > 0.1:
                bullish += 1
            elif output.score < -0.1:
                bearish += 1
            else:
                neutral += 1

            if not output.should_trade:
                blockers += 1

        if total_weight > 0:
            consensus_score = weighted_score / total_weight
            consensus_confidence = weighted_confidence / total_weight
            risk_level = weighted_risk / total_weight
        else:
            consensus_score = 0.0
            consensus_confidence = 0.0
            risk_level = 0.5

        # Clamp
        consensus_score = max(-1.0, min(1.0, consensus_score))
        consensus_confidence = max(0.0, min(1.0, consensus_confidence))
        risk_level = max(0.0, min(1.0, risk_level))

        # Should trade: majority vote (accounting for blockers)
        trade_votes = sum(1 for o in outputs if o.should_trade)
        should_trade = (trade_votes / len(outputs)) >= self._blocker_threshold

        # Agreement metric
        scores = [o.score for o in outputs]
        if len(scores) > 1:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            agreement = max(0.0, 1.0 - variance)  # low variance = high agreement
        else:
            agreement = 1.0

        # Dominant signal
        if bullish > bearish and bullish > neutral:
            dominant = "bullish"
        elif bearish > bullish and bearish > neutral:
            dominant = "bearish"
        else:
            dominant = "neutral"

        result = ConsensusResult(
            consensus_score=consensus_score,
            consensus_confidence=consensus_confidence,
            should_trade=should_trade,
            risk_level=risk_level,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            blocker_count=blockers,
            agent_outputs=outputs,
            agent_agreement=agreement,
            dominant_signal=dominant,
        )

        _log.info(
            "orchestrator.consensus",
            score=round(consensus_score, 3),
            confidence=round(consensus_confidence, 3),
            should_trade=should_trade,
            bullish=bullish,
            bearish=bearish,
            blockers=blockers,
            agreement=round(agreement, 3),
        )

        return result


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: AgentOrchestrator | None = None


def get_orchestrator() -> AgentOrchestrator:
    """Return the module-level singleton AgentOrchestrator."""
    global _singleton
    if _singleton is None:
        _singleton = AgentOrchestrator()
        _log.info("orchestrator.singleton_initialised", agents=_singleton.agent_count)
    return _singleton
