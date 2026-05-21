"""
libs.agents — Multi-agent trading intelligence system.

15 specialized autonomous agents that each independently analyze market
conditions and produce scored recommendations.  The orchestrator
combines all agent outputs into a weighted consensus decision.
"""
from libs.agents.base import TradingAgent, AgentOutput
from libs.agents.orchestrator import AgentOrchestrator

__all__ = ["TradingAgent", "AgentOutput", "AgentOrchestrator"]
