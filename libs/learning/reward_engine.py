"""
libs.learning.reward_engine — RL-lite bandit strategy selector.

Each strategy maintains a score influenced by trade outcomes.  Selection
probability is computed via softmax over scores with a minimum-probability
floor so no strategy is ever fully excluded.

Design rules:
  - All public methods are pure / side-effect-free except score mutation
  - No magic numbers — all thresholds are named class constants
  - JSON persistence for score portability across sessions
  - Module-level singleton via get_reward_engine()
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from libs.core.logging.logger import get_logger

_log = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

REWARD_SCALE: float = 0.1

# compute_reward component weights
_PNL_REWARD: float = 1.0
_PNL_PENALTY: float = -1.0
_RR_HIGH_REWARD: float = 0.5        # rr > 2.0
_RR_LOW_PENALTY: float = -0.5       # rr < 1.0 and loss
_DISCIPLINED_EXIT_REWARD: float = 0.3
_DISCIPLINED_EXIT_PENALTY: float = -0.3
_REGIME_ALIGNED_REWARD: float = 0.2
_REGIME_ALIGNED_PENALTY: float = -0.2
_LOW_DRAWDOWN_REWARD: float = 0.1
_LOW_DRAWDOWN_PENALTY: float = -0.2

# penalize component weights
_PENALIZE_BASE: float = -1.0
_PENALIZE_HOLDING_LOSERS: float = -0.5
_PENALIZE_OVERTRADING: float = -0.3
_PENALIZE_LARGE_LOSS: float = -0.5   # pnl < -100

# RR thresholds
_RR_HIGH_THRESHOLD: float = 2.0
_RR_LOW_THRESHOLD: float = 1.0

# Large-loss threshold
_LARGE_LOSS_THRESHOLD: float = -100.0


# ── RewardEngine ──────────────────────────────────────────────────────────────

class RewardEngine:
    """Bandit-style strategy selector using score-based softmax probabilities.

    Parameters
    ----------
    strategies:       List of strategy names to track.
    min_probability:  Floor applied to every strategy after softmax (default 0.05).
    temperature:      Softmax temperature; higher = more uniform (default 1.0).
    """

    def __init__(
        self,
        strategies: list[str],
        min_probability: float = 0.05,
        temperature: float = 1.0,
    ) -> None:
        if not strategies:
            raise ValueError("strategies must be a non-empty list")
        if not (0.0 <= min_probability < 1.0):
            raise ValueError("min_probability must be in [0, 1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        self._strategies: list[str] = list(strategies)
        self._min_probability = min_probability
        self._temperature = temperature
        self._scores: dict[str, float] = {s: 0.0 for s in strategies}
        self._reward_counts: dict[str, int] = {s: 0 for s in strategies}
        self._penalty_counts: dict[str, int] = {s: 0 for s in strategies}

    # ── Reward signal ─────────────────────────────────────────────────────────

    def compute_reward(
        self,
        pnl: float,
        rr: float = 1.0,
        disciplined_exit: bool = True,
        regime_aligned: bool = True,
        low_drawdown: bool = True,
    ) -> float:
        """Compute a scalar reward from trade quality attributes.

        Parameters
        ----------
        pnl:             Realised profit/loss.
        rr:              Realised risk/reward ratio.
        disciplined_exit: Trade was closed at planned exit.
        regime_aligned:  Trade direction matched the current market regime.
        low_drawdown:    Drawdown during the trade was within acceptable limits.

        Returns
        -------
        Scalar reward (positive = good, negative = bad).
        """
        reward = _PNL_REWARD if pnl > 0 else _PNL_PENALTY

        if rr > _RR_HIGH_THRESHOLD:
            reward += _RR_HIGH_REWARD
        elif rr < _RR_LOW_THRESHOLD and pnl <= 0:
            reward += _RR_LOW_PENALTY

        reward += _DISCIPLINED_EXIT_REWARD if disciplined_exit else _DISCIPLINED_EXIT_PENALTY
        reward += _REGIME_ALIGNED_REWARD if regime_aligned else _REGIME_ALIGNED_PENALTY
        reward += _LOW_DRAWDOWN_REWARD if low_drawdown else _LOW_DRAWDOWN_PENALTY

        return reward

    def reward(
        self,
        strategy: str,
        pnl: float,
        rr: float = 1.0,
        disciplined_exit: bool = True,
        regime_aligned: bool = True,
    ) -> None:
        """Apply a positive reward to a strategy's score.

        Uses compute_reward internally and scales the result by REWARD_SCALE.
        Only the positive portion of compute_reward is applied here; use
        penalize() for explicit loss feedback.

        Parameters
        ----------
        strategy: Name of the strategy to reward.
        pnl:      Realised profit/loss (expected > 0 for reward calls).
        rr:       Realised risk/reward ratio.
        disciplined_exit: Trade was closed at planned exit.
        regime_aligned:   Trade matched the market regime.
        """
        self._assert_known(strategy)
        raw = self.compute_reward(
            pnl=pnl,
            rr=rr,
            disciplined_exit=disciplined_exit,
            regime_aligned=regime_aligned,
        )
        delta = raw * REWARD_SCALE
        self._scores[strategy] += delta
        self._reward_counts[strategy] += 1
        _log.debug(
            "reward applied",
            strategy=strategy,
            raw_reward=raw,
            delta=delta,
            new_score=self._scores[strategy],
        )

    def penalize(
        self,
        strategy: str,
        pnl: float,
        rr: float = 0.5,
        holding_losers: bool = False,
        overtrading: bool = False,
    ) -> None:
        """Apply a penalty to a strategy's score.

        Parameters
        ----------
        strategy:       Name of the strategy to penalise.
        pnl:            Realised profit/loss (expected <= 0 for penalise calls).
        rr:             Realised risk/reward ratio (unused here but kept for API
                        symmetry with reward()).
        holding_losers: Position was held too long against the trade.
        overtrading:    Penalty for excessive trading frequency.
        """
        self._assert_known(strategy)
        penalty = _PENALIZE_BASE
        if holding_losers:
            penalty += _PENALIZE_HOLDING_LOSERS
        if overtrading:
            penalty += _PENALIZE_OVERTRADING
        if pnl < _LARGE_LOSS_THRESHOLD:
            penalty += _PENALIZE_LARGE_LOSS

        delta = penalty * REWARD_SCALE
        self._scores[strategy] += delta
        self._penalty_counts[strategy] += 1
        _log.debug(
            "penalty applied",
            strategy=strategy,
            raw_penalty=penalty,
            delta=delta,
            new_score=self._scores[strategy],
        )

    # ── Probability / selection ───────────────────────────────────────────────

    def get_probabilities(self) -> dict[str, float]:
        """Return softmax probabilities with a minimum-probability floor.

        The floor ensures every strategy retains at least ``min_probability``
        selection probability regardless of historical performance.  After
        flooring, probabilities are renormalised to sum exactly to 1.0.

        Returns
        -------
        Dict mapping strategy name → probability (all values sum to 1.0).
        """
        scores = [self._scores[s] / self._temperature for s in self._strategies]

        # Numerically stable softmax: subtract max before exp.
        max_score = max(scores)
        exps = [math.exp(s - max_score) for s in scores]
        total_exp = sum(exps)
        raw_probs = [e / total_exp for e in exps]

        # Apply minimum-probability floor and redistribute.
        n = len(self._strategies)

        # Edge case: if min_probability * n >= 1.0 force uniform.
        if self._min_probability * n >= 1.0:
            return dict(zip(self._strategies, [1.0 / n] * n))

        # Iteratively enforce floor: clamp low probs, redistribute excess.
        probs = list(raw_probs)
        for _ in range(n):  # converges in ≤ n iterations
            floor_deficit = 0.0
            above_floor_total = 0.0
            for i, p in enumerate(probs):
                if p < self._min_probability:
                    floor_deficit += self._min_probability - p
                    probs[i] = self._min_probability
                else:
                    above_floor_total += p
            if floor_deficit == 0.0:
                break
            # Shrink above-floor probs proportionally to fund the deficit
            if above_floor_total > 0:
                scale = (above_floor_total - floor_deficit) / above_floor_total
                for i, p in enumerate(probs):
                    if p > self._min_probability:
                        probs[i] = max(self._min_probability, p * scale)

        return dict(zip(self._strategies, probs))

    def should_take(self, strategy: str) -> bool:
        """Return True if this strategy should be executed now.

        Uses the strategy's current probability as the acceptance threshold.
        A uniform random draw determines acceptance.

        Parameters
        ----------
        strategy: Name of the strategy to evaluate.
        """
        self._assert_known(strategy)
        probs = self.get_probabilities()
        return random.random() < probs[strategy]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist scores and metadata to a JSON file.

        Parameters
        ----------
        path: Destination file path (parent directories are created).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "strategies": self._strategies,
            "min_probability": self._min_probability,
            "temperature": self._temperature,
            "scores": self._scores,
            "reward_counts": self._reward_counts,
            "penalty_counts": self._penalty_counts,
        }
        dest.write_text(json.dumps(payload, indent=2))
        _log.info("reward engine saved", path=str(dest))

    def load(self, path: str | Path) -> None:
        """Restore scores from a previously saved JSON file.

        Only the scores and counts are restored; constructor arguments
        (strategies, min_probability, temperature) remain unchanged.

        Parameters
        ----------
        path: Source file path.
        """
        src = Path(path)
        payload: dict[str, Any] = json.loads(src.read_text())
        for strategy in self._strategies:
            if strategy in payload.get("scores", {}):
                self._scores[strategy] = payload["scores"][strategy]
            if strategy in payload.get("reward_counts", {}):
                self._reward_counts[strategy] = payload["reward_counts"][strategy]
            if strategy in payload.get("penalty_counts", {}):
                self._penalty_counts[strategy] = payload["penalty_counts"][strategy]
        _log.info("reward engine loaded", path=str(src))

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return a dashboard summary of current engine state.

        Returns
        -------
        Dict with keys:
          - ``strategies``:     list of strategy names
          - ``scores``:         dict of raw scores
          - ``probabilities``:  dict of selection probabilities
          - ``reward_counts``:  dict of reward application counts
          - ``penalty_counts``: dict of penalty application counts
          - ``min_probability``: configured floor
          - ``temperature``:    configured temperature
        """
        return {
            "strategies": list(self._strategies),
            "scores": dict(self._scores),
            "probabilities": self.get_probabilities(),
            "reward_counts": dict(self._reward_counts),
            "penalty_counts": dict(self._penalty_counts),
            "min_probability": self._min_probability,
            "temperature": self._temperature,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _assert_known(self, strategy: str) -> None:
        """Auto-register unknown strategies so new strategies can be learned."""
        if strategy not in self._scores:
            self._strategies.append(strategy)
            self._scores[strategy] = 0.0
            self._reward_counts[strategy] = 0
            self._penalty_counts[strategy] = 0
            _log.debug("strategy_auto_registered", strategy=strategy)


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: RewardEngine | None = None
_DEFAULT_STRATEGIES = [
    "momentum",
    "reversal",
    "breakout",
    "continuation",
    "trend",
]


def get_reward_engine(strategies: list[str] | None = None) -> RewardEngine:
    """Return the module-level singleton RewardEngine.

    On first call the engine is initialised with *strategies* (or the default
    strategy list when omitted).  Subsequent calls return the same instance
    regardless of the *strategies* argument.

    Parameters
    ----------
    strategies: Strategy names for the engine.  Only used on first call.
    """
    global _singleton
    if _singleton is None:
        names = strategies if strategies is not None else _DEFAULT_STRATEGIES
        _singleton = RewardEngine(strategies=names)
        _log.info("reward engine singleton initialised", strategies=names)
    return _singleton
