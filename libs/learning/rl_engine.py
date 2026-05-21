"""
libs.learning.rl_engine — Tabular Q-Learning Reinforcement Learning Engine.

Implements a state-action-value table (Q-table) with epsilon-greedy
exploration for strategy selection and position sizing decisions.

State space: regime × volatility_bucket × trend_direction
Action space: strategy_id × size_bucket × trade_or_skip

Design rules:
  - Tabular Q-learning (interpretable, works with small data)
  - Epsilon-greedy with decay for exploration/exploitation balance
  - JSON persistence for Q-table portability
  - Module-level singleton via get_rl_engine()
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from libs.core.logging.logger import get_logger

_log = get_logger(__name__)


# ── State/Action encoding ────────────────────────────────────────────────────

REGIME_STATES: list[str] = [
    "trending_up", "trending_down", "ranging_low_vol", "ranging_high_vol",
    "breakout", "climactic", "accumulation", "distribution",
    "panic_selloff", "liquidation_event", "reversal", "low_liquidity",
    "compression", "expansion", "news_driven", "mean_reversion", "unknown",
]

VOLATILITY_BUCKETS: list[str] = ["very_low", "low", "medium", "high", "extreme"]

TREND_DIRECTIONS: list[str] = ["strong_up", "up", "flat", "down", "strong_down"]

SIZE_BUCKETS: list[str] = ["skip", "quarter", "half", "normal", "aggressive"]

DEFAULT_STRATEGIES: list[str] = [
    "momentum", "reversal", "breakout", "continuation", "trend",
    "mean_reversion",
]


def encode_state(
    regime: str = "unknown",
    volatility_bucket: str = "medium",
    trend_direction: str = "flat",
) -> str:
    """Encode a state tuple into a string key for the Q-table."""
    return f"{regime}|{volatility_bucket}|{trend_direction}"


def decode_state(state_key: str) -> dict[str, str]:
    """Decode a state string key back into components."""
    parts = state_key.split("|")
    return {
        "regime": parts[0] if len(parts) > 0 else "unknown",
        "volatility_bucket": parts[1] if len(parts) > 1 else "medium",
        "trend_direction": parts[2] if len(parts) > 2 else "flat",
    }


def encode_action(strategy: str, size_bucket: str = "normal") -> str:
    """Encode an action tuple into a string key."""
    return f"{strategy}|{size_bucket}"


def decode_action(action_key: str) -> dict[str, str]:
    """Decode an action string key back into components."""
    parts = action_key.split("|")
    return {
        "strategy": parts[0] if len(parts) > 0 else "",
        "size_bucket": parts[1] if len(parts) > 1 else "normal",
    }


def classify_volatility(atr_pct: float) -> str:
    """Classify ATR% into a volatility bucket."""
    if atr_pct < 0.5:
        return "very_low"
    if atr_pct < 1.5:
        return "low"
    if atr_pct < 3.0:
        return "medium"
    if atr_pct < 6.0:
        return "high"
    return "extreme"


def classify_trend(ema_ratio: float) -> str:
    """Classify EMA fast/slow ratio into trend direction."""
    if ema_ratio > 1.02:
        return "strong_up"
    if ema_ratio > 1.005:
        return "up"
    if ema_ratio < 0.98:
        return "strong_down"
    if ema_ratio < 0.995:
        return "down"
    return "flat"


# ── Q-Learning Engine ────────────────────────────────────────────────────────

class RLEngine:
    """Tabular Q-Learning engine for trading strategy and sizing decisions.

    Parameters
    ----------
    learning_rate: Alpha — step size for Q-value updates (default 0.1).
    discount_factor: Gamma — weight of future rewards (default 0.95).
    epsilon: Initial exploration rate (default 0.3).
    epsilon_min: Minimum exploration rate after decay (default 0.05).
    epsilon_decay: Multiplicative decay per update (default 0.999).
    strategies: List of strategy names for action space.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.3,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        strategies: list[str] | None = None,
    ) -> None:
        self._lr = learning_rate
        self._gamma = discount_factor
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._strategies = strategies or list(DEFAULT_STRATEGIES)

        # Q-table: state_key → action_key → Q-value
        self._q_table: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Visit counts for diagnostics
        self._visit_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        self._total_updates: int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def total_updates(self) -> int:
        return self._total_updates

    # ── Action generation ─────────────────────────────────────────────────────

    def _all_actions(self) -> list[str]:
        """Generate all valid action keys."""
        actions: list[str] = []
        for strategy in self._strategies:
            for size in SIZE_BUCKETS:
                actions.append(encode_action(strategy, size))
        return actions

    def select_action(
        self,
        regime: str = "unknown",
        volatility_bucket: str = "medium",
        trend_direction: str = "flat",
    ) -> dict[str, str]:
        """Select an action using epsilon-greedy policy.

        Parameters
        ----------
        regime: Current market regime.
        volatility_bucket: Current volatility classification.
        trend_direction: Current trend classification.

        Returns
        -------
        Dict with keys: strategy, size_bucket.
        """
        state = encode_state(regime, volatility_bucket, trend_direction)
        all_actions = self._all_actions()

        # Epsilon-greedy: explore with probability epsilon
        if random.random() < self._epsilon:
            action_key = random.choice(all_actions)
        else:
            # Exploit: pick action with highest Q-value
            q_values = self._q_table[state]
            if not q_values:
                action_key = random.choice(all_actions)
            else:
                action_key = max(q_values, key=lambda a: q_values[a])

        return decode_action(action_key)

    def get_q_value(self, state_key: str, action_key: str) -> float:
        """Get Q-value for a state-action pair."""
        return self._q_table[state_key][action_key]

    def get_best_action(self, state_key: str) -> tuple[str, float]:
        """Get best action and its Q-value for a state.

        Returns ("", 0.0) if no actions recorded for this state.
        """
        q_values = self._q_table[state_key]
        if not q_values:
            return "", 0.0
        best_action = max(q_values, key=lambda a: q_values[a])
        return best_action, q_values[best_action]

    # ── Learning ──────────────────────────────────────────────────────────────

    def compute_reward(
        self,
        pnl: float,
        rr: float = 1.0,
        won: bool = False,
        disciplined_exit: bool = True,
        regime_aligned: bool = True,
        drawdown_pct: float = 0.0,
    ) -> float:
        """Compute scalar reward from trade outcome.

        Reward components:
          - PnL-based: +1.0 win, -1.0 loss, scaled by magnitude
          - RR bonus: +0.5 if rr > 2.0
          - Discipline: +0.3 / -0.3
          - Regime alignment: +0.2 / -0.2
          - Drawdown penalty: -0.5 if > 5%
        """
        reward = 1.0 if won else -1.0

        # Scale by PnL magnitude (clamped)
        pnl_scale = min(abs(pnl) / 100.0, 2.0)
        reward *= (1.0 + pnl_scale * 0.5)

        if rr > 2.0:
            reward += 0.5
        elif rr < 0.5 and not won:
            reward -= 0.5

        reward += 0.3 if disciplined_exit else -0.3
        reward += 0.2 if regime_aligned else -0.2

        if drawdown_pct > 5.0:
            reward -= 0.5

        return reward

    def update(
        self,
        state_key: str,
        action_key: str,
        reward: float,
        next_state_key: str | None = None,
    ) -> float:
        """Update Q-value for a state-action pair using Q-learning rule.

        Q(s,a) ← Q(s,a) + α[r + γ·max_a'(Q(s',a')) - Q(s,a)]

        Parameters
        ----------
        state_key: Current state string.
        action_key: Action taken string.
        reward: Observed reward.
        next_state_key: Next state (None = terminal).

        Returns
        -------
        Updated Q-value.
        """
        current_q = self._q_table[state_key][action_key]

        # Max Q-value for next state (0 if terminal)
        max_next_q = 0.0
        if next_state_key is not None:
            next_q_values = self._q_table[next_state_key]
            if next_q_values:
                max_next_q = max(next_q_values.values())

        # Q-learning update
        td_target = reward + self._gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self._lr * td_error

        self._q_table[state_key][action_key] = new_q
        self._visit_counts[state_key][action_key] += 1
        self._total_updates += 1

        # Decay epsilon
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

        _log.debug(
            "rl_engine.update",
            state=state_key,
            action=action_key,
            reward=round(reward, 3),
            old_q=round(current_q, 4),
            new_q=round(new_q, 4),
            epsilon=round(self._epsilon, 4),
        )

        return new_q

    def learn_from_trade(
        self,
        regime: str,
        atr_pct: float,
        ema_ratio: float,
        strategy: str,
        size_bucket: str,
        pnl: float,
        rr: float = 1.0,
        won: bool = False,
        disciplined_exit: bool = True,
        regime_aligned: bool = True,
        drawdown_pct: float = 0.0,
        next_regime: str | None = None,
        next_atr_pct: float | None = None,
        next_ema_ratio: float | None = None,
    ) -> float:
        """Convenience method: encode state/action, compute reward, update Q-table.

        Returns the computed reward.
        """
        vol_bucket = classify_volatility(atr_pct)
        trend_dir = classify_trend(ema_ratio)

        state_key = encode_state(regime, vol_bucket, trend_dir)
        action_key = encode_action(strategy, size_bucket)

        reward = self.compute_reward(
            pnl=pnl,
            rr=rr,
            won=won,
            disciplined_exit=disciplined_exit,
            regime_aligned=regime_aligned,
            drawdown_pct=drawdown_pct,
        )

        next_state_key = None
        if next_regime is not None and next_atr_pct is not None and next_ema_ratio is not None:
            next_vol = classify_volatility(next_atr_pct)
            next_trend = classify_trend(next_ema_ratio)
            next_state_key = encode_state(next_regime, next_vol, next_trend)

        self.update(state_key, action_key, reward, next_state_key)
        return reward

    # ── Policy query ──────────────────────────────────────────────────────────

    def get_policy_recommendation(
        self,
        regime: str,
        atr_pct: float,
        ema_ratio: float,
    ) -> dict[str, Any]:
        """Get the current policy recommendation for given market conditions.

        Returns
        -------
        Dict with: strategy, size_bucket, q_value, confidence,
        state_visits, exploration_rate.
        """
        vol_bucket = classify_volatility(atr_pct)
        trend_dir = classify_trend(ema_ratio)
        state_key = encode_state(regime, vol_bucket, trend_dir)

        best_action, best_q = self.get_best_action(state_key)
        state_visits = sum(self._visit_counts[state_key].values())

        if best_action:
            action_parts = decode_action(best_action)
            # Confidence scales with visits and Q-value magnitude
            confidence = min(1.0, state_visits / 50.0) * min(1.0, abs(best_q) / 2.0)
        else:
            action_parts = {"strategy": "", "size_bucket": "normal"}
            confidence = 0.0

        return {
            "strategy": action_parts["strategy"],
            "size_bucket": action_parts["size_bucket"],
            "q_value": round(best_q, 4),
            "confidence": round(confidence, 4),
            "state_visits": state_visits,
            "exploration_rate": round(self._epsilon, 4),
            "state": state_key,
        }

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics of the RL engine state."""
        total_states = len(self._q_table)
        total_actions = sum(len(v) for v in self._q_table.values())
        total_visits = sum(
            sum(v.values()) for v in self._visit_counts.values()
        )

        # Best and worst Q-values
        all_q_values: list[float] = []
        for actions in self._q_table.values():
            all_q_values.extend(actions.values())

        return {
            "total_states": total_states,
            "total_state_action_pairs": total_actions,
            "total_updates": self._total_updates,
            "total_visits": total_visits,
            "epsilon": round(self._epsilon, 4),
            "learning_rate": self._lr,
            "discount_factor": self._gamma,
            "strategies": list(self._strategies),
            "max_q": round(max(all_q_values), 4) if all_q_values else 0.0,
            "min_q": round(min(all_q_values), 4) if all_q_values else 0.0,
            "mean_q": round(sum(all_q_values) / len(all_q_values), 4) if all_q_values else 0.0,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist Q-table and metadata to JSON.

        Parameters
        ----------
        path: Destination file path (parent directories are created).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "learning_rate": self._lr,
            "discount_factor": self._gamma,
            "epsilon": self._epsilon,
            "epsilon_min": self._epsilon_min,
            "epsilon_decay": self._epsilon_decay,
            "strategies": self._strategies,
            "total_updates": self._total_updates,
            "q_table": {
                state: dict(actions)
                for state, actions in self._q_table.items()
            },
            "visit_counts": {
                state: dict(counts)
                for state, counts in self._visit_counts.items()
            },
        }
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _log.info("rl_engine.saved", path=str(dest), states=len(self._q_table))

    def load(self, path: str | Path) -> None:
        """Restore Q-table from a previously saved JSON file.

        Parameters
        ----------
        path: Source file path.
        """
        src = Path(path)
        if not src.exists():
            _log.debug("rl_engine.no_file", path=str(src))
            return

        payload: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))

        self._lr = payload.get("learning_rate", self._lr)
        self._gamma = payload.get("discount_factor", self._gamma)
        self._epsilon = payload.get("epsilon", self._epsilon)
        self._epsilon_min = payload.get("epsilon_min", self._epsilon_min)
        self._epsilon_decay = payload.get("epsilon_decay", self._epsilon_decay)
        self._strategies = payload.get("strategies", self._strategies)
        self._total_updates = payload.get("total_updates", 0)

        raw_q = payload.get("q_table", {})
        self._q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in raw_q.items():
            for action, q_val in actions.items():
                self._q_table[state][action] = q_val

        raw_visits = payload.get("visit_counts", {})
        self._visit_counts = defaultdict(lambda: defaultdict(int))
        for state, counts in raw_visits.items():
            for action, count in counts.items():
                self._visit_counts[state][action] = count

        _log.info(
            "rl_engine.loaded",
            path=str(src),
            states=len(self._q_table),
            total_updates=self._total_updates,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: RLEngine | None = None


def get_rl_engine() -> RLEngine:
    """Return the module-level singleton RLEngine.

    The instance is created lazily on the first call.
    """
    global _singleton
    if _singleton is None:
        _singleton = RLEngine()
        _log.info("rl_engine.singleton_initialised")
    return _singleton
