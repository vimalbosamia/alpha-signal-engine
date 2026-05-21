"""
libs.learning.agent_weight_tuner — Auto-tune agent weights from trade outcomes.

Tracks each agent's predictive accuracy: did the agent's score direction
match the actual trade outcome?  Agents that consistently predict correctly
get higher weights; agents that are wrong get lower weights.

Design rules:
  - No magic numbers — all thresholds are named constants
  - JSON persistence for portability
  - Module-level singleton via get_weight_tuner()
  - Weights are bounded [MIN_WEIGHT, MAX_WEIGHT] to prevent extinction
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from libs.core.logging.logger import get_logger

_log = get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

MIN_WEIGHT: float = 0.3
MAX_WEIGHT: float = 3.0
DEFAULT_WEIGHT: float = 1.0
LEARNING_RATE: float = 0.05       # how fast weights adapt
DECAY_RATE: float = 0.995         # slow decay toward default (prevents overfit)
MIN_SAMPLES: int = 10             # minimum trades before adjusting weights


class AgentWeightTuner:
    """Learns optimal agent weights from trade outcome feedback.

    For each trade, records which agents were correct (score direction
    matched outcome) and adjusts weights proportionally.

    Parameters
    ----------
    default_weights: Optional dict of agent_name → initial weight.
    """

    def __init__(self, default_weights: dict[str, float] | None = None) -> None:
        self._weights: dict[str, float] = dict(default_weights) if default_weights else {}
        self._correct_counts: dict[str, int] = defaultdict(int)
        self._total_counts: dict[str, int] = defaultdict(int)
        self._total_updates: int = 0

    @property
    def total_updates(self) -> int:
        return self._total_updates

    # ── Weight access ────────────────────────────────────────────────────────

    def get_weight(self, agent_name: str) -> float:
        """Get current weight for an agent."""
        return self._weights.get(agent_name, DEFAULT_WEIGHT)

    def get_all_weights(self) -> dict[str, float]:
        """Get all current agent weights."""
        return dict(self._weights)

    # ── Learning ─────────────────────────────────────────────────────────────

    def record_outcome(
        self,
        agent_outputs: list[dict[str, Any]],
        won: bool,
        pnl: float = 0.0,
    ) -> None:
        """Record a trade outcome and update agent weights.

        Parameters
        ----------
        agent_outputs: List of dicts with keys: agent_name, score, confidence.
        won: Whether the trade was profitable.
        pnl: Realized PnL (for magnitude weighting).
        """
        for output in agent_outputs:
            name = output.get("agent_name", "")
            score = output.get("score", 0.0)
            confidence = output.get("confidence", 0.5)

            if not name:
                continue

            # Initialize weight if not seen before
            if name not in self._weights:
                self._weights[name] = DEFAULT_WEIGHT

            self._total_counts[name] += 1

            # Agent was "correct" if score direction matched outcome
            agent_bullish = score > 0.05
            agent_bearish = score < -0.05
            agent_neutral = not agent_bullish and not agent_bearish

            correct = False
            if won and agent_bullish:
                correct = True
            elif not won and agent_bearish:
                correct = True
            elif agent_neutral:
                # Neutral agents don't get credit or penalty
                continue

            if correct:
                self._correct_counts[name] += 1

            # Only adjust weights after minimum samples
            if self._total_counts[name] < MIN_SAMPLES:
                continue

            # Compute accuracy
            accuracy = self._correct_counts[name] / self._total_counts[name]

            # Weight adjustment: above 0.5 accuracy → increase, below → decrease
            adjustment = (accuracy - 0.5) * LEARNING_RATE * confidence
            new_weight = self._weights[name] + adjustment

            # Decay toward default to prevent runaway weights
            new_weight = new_weight * DECAY_RATE + DEFAULT_WEIGHT * (1 - DECAY_RATE)

            # Clamp
            new_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))
            self._weights[name] = new_weight

        self._total_updates += 1

        if self._total_updates % 50 == 0:
            _log.info(
                "agent_weight_tuner.update",
                total_updates=self._total_updates,
                weights={k: round(v, 3) for k, v in self._weights.items()},
            )

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_accuracy(self, agent_name: str) -> float:
        """Get accuracy for a specific agent (0.0–1.0)."""
        total = self._total_counts.get(agent_name, 0)
        if total == 0:
            return 0.5
        return self._correct_counts.get(agent_name, 0) / total

    def get_stats(self) -> dict[str, Any]:
        """Return summary of all agent weights and accuracies."""
        agents: list[dict[str, Any]] = []
        for name in sorted(self._weights.keys()):
            total = self._total_counts.get(name, 0)
            correct = self._correct_counts.get(name, 0)
            accuracy = correct / total if total > 0 else 0.5
            agents.append({
                "agent": name,
                "weight": round(self._weights[name], 4),
                "accuracy": round(accuracy, 4),
                "total_predictions": total,
                "correct_predictions": correct,
            })

        return {
            "total_updates": self._total_updates,
            "agents": agents,
            "best_agent": max(agents, key=lambda a: a["accuracy"])["agent"] if agents else "",
            "worst_agent": min(agents, key=lambda a: a["accuracy"])["agent"] if agents else "",
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist weights and counts to JSON."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "weights": self._weights,
            "correct_counts": dict(self._correct_counts),
            "total_counts": dict(self._total_counts),
            "total_updates": self._total_updates,
        }
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _log.info("agent_weight_tuner.saved", path=str(dest))

    def load(self, path: str | Path) -> None:
        """Restore weights from a previously saved JSON file."""
        src = Path(path)
        if not src.exists():
            return

        payload: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
        self._weights = payload.get("weights", {})
        self._correct_counts = defaultdict(int, payload.get("correct_counts", {}))
        self._total_counts = defaultdict(int, payload.get("total_counts", {}))
        self._total_updates = payload.get("total_updates", 0)
        _log.info("agent_weight_tuner.loaded", path=str(src), agents=len(self._weights))


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: AgentWeightTuner | None = None


def get_weight_tuner() -> AgentWeightTuner:
    """Return the module-level singleton AgentWeightTuner."""
    global _singleton
    if _singleton is None:
        _singleton = AgentWeightTuner()
        _log.info("agent_weight_tuner.singleton_initialised")
    return _singleton
