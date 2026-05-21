"""
libs.agents.base — Base protocol and data structures for trading agents.

Every agent follows the same interface: observe market data, produce a
score with confidence, and provide reasoning.  All outputs are immutable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentOutput:
    """Immutable output from a single trading agent.

    Attributes
    ----------
    agent_name:  Unique identifier for the agent.
    score:       Directional score from -1.0 (strong bearish) to +1.0 (strong bullish).
                 0.0 = neutral / no opinion.
    confidence:  How confident the agent is in its score (0.0–1.0).
    reasoning:   Human-readable explanation of the score.
    signals:     Additional structured data specific to the agent.
    should_trade: Whether this agent recommends entering a trade.
    risk_level:  Agent's assessed risk level (0.0 = safe, 1.0 = extreme).
    """
    agent_name: str
    score: float            # -1.0 to +1.0
    confidence: float       # 0.0 to 1.0
    reasoning: str = ""
    signals: dict[str, Any] = field(default_factory=dict)
    should_trade: bool = True
    risk_level: float = 0.5  # 0.0 to 1.0


class TradingAgent(ABC):
    """Abstract base class for all trading agents.

    Each agent independently analyzes market conditions from its own
    perspective and produces an AgentOutput.  Agents must be stateless
    per-call (state lives in the learning subsystems, not in agents).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier."""

    @property
    def weight(self) -> float:
        """Default weight in orchestrator consensus (override to customize)."""
        return 1.0

    @abstractmethod
    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        """Analyze market conditions and produce a recommendation.

        Parameters
        ----------
        context: Dict containing market data, indicators, regime info, etc.
                 Each agent extracts what it needs from this shared context.

        Returns
        -------
        AgentOutput with the agent's score, confidence, and reasoning.
        """

    def _safe_get(self, context: dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely extract a value from context with default."""
        return context.get(key, default)

    def _clamp(self, value: float, low: float = -1.0, high: float = 1.0) -> float:
        """Clamp a value to [low, high] range."""
        return max(low, min(high, value))
