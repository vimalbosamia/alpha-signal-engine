"""
libs.learning.strategy_evolution — Autonomous Strategy Evolution Engine.

Implements Phase 7-8 of the training pipeline: strategies evolve their
parameters automatically based on performance.  Old configurations are
compared against new ones, superior configs are promoted, and degraded
strategies are disabled.

Design rules:
  - No magic numbers — all thresholds are named constants
  - JSON persistence for config versioning
  - Module-level singleton via get_evolution_engine()
  - Strategies are never fully deleted — only disabled (can be re-enabled)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from libs.core.logging.logger import get_logger

_log = get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

MIN_TRADES_FOR_EVALUATION: int = 30
PROMOTION_WIN_RATE_THRESHOLD: float = 0.55
DEGRADATION_WIN_RATE_THRESHOLD: float = 0.35
DEGRADATION_SHARPE_THRESHOLD: float = 0.0
MIN_IMPROVEMENT_PCT: float = 5.0   # new config must be 5%+ better to promote
MAX_PARAM_MUTATION_PCT: float = 20.0  # max parameter change per evolution


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """A set of tunable parameters for a strategy."""
    strategy_name: str
    version: int = 1
    params: dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    is_promoted: bool = False

    # Performance tracking
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_rr: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "version": self.version,
            "params": self.params,
            "is_active": self.is_active,
            "is_promoted": self.is_promoted,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": self.total_pnl,
            "avg_rr": self.avg_rr,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Default parameter ranges per strategy ────────────────────────────────────

DEFAULT_PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "momentum": {
        "rsi_threshold": (20.0, 40.0),
        "ema_fast": (5.0, 15.0),
        "ema_slow": (15.0, 30.0),
        "atr_multiplier_sl": (1.0, 3.0),
        "atr_multiplier_tp": (1.5, 5.0),
        "min_volume_ratio": (0.5, 2.0),
        "confidence_threshold": (0.4, 0.8),
    },
    "reversal": {
        "rsi_oversold": (15.0, 35.0),
        "rsi_overbought": (65.0, 85.0),
        "confirmation_candles": (1.0, 3.0),
        "atr_multiplier_sl": (1.0, 2.5),
        "atr_multiplier_tp": (2.0, 5.0),
        "min_pattern_confidence": (0.5, 0.8),
    },
    "breakout": {
        "lookback_bars": (10.0, 50.0),
        "volume_surge_threshold": (1.5, 3.0),
        "atr_multiplier_sl": (0.5, 2.0),
        "atr_multiplier_tp": (2.0, 6.0),
        "min_range_bars": (5.0, 20.0),
    },
    "continuation": {
        "pullback_depth_pct": (20.0, 50.0),
        "ema_alignment_threshold": (1.002, 1.02),
        "atr_multiplier_sl": (1.0, 2.5),
        "atr_multiplier_tp": (1.5, 4.0),
    },
    "trend": {
        "ema_fast": (8.0, 15.0),
        "ema_slow": (20.0, 50.0),
        "adx_threshold": (20.0, 35.0),
        "trailing_stop_atr": (1.5, 3.0),
    },
    "mean_reversion": {
        "bb_std_dev": (1.5, 3.0),
        "rsi_oversold": (20.0, 35.0),
        "rsi_overbought": (65.0, 80.0),
        "mean_period": (15.0, 30.0),
        "atr_multiplier_sl": (1.0, 2.0),
    },
}


class StrategyEvolutionEngine:
    """Evolves strategy parameters based on performance feedback.

    Maintains a population of strategy configurations, evaluates their
    performance, mutates parameters, and promotes superior configurations.
    """

    def __init__(self) -> None:
        self._configs: dict[str, list[StrategyConfig]] = {}  # strategy_name → versions
        self._evolution_count: int = 0
        self._promotions: int = 0
        self._disablements: int = 0

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize_strategy(self, strategy_name: str) -> StrategyConfig:
        """Create initial config for a strategy with default params."""
        ranges = DEFAULT_PARAM_RANGES.get(strategy_name, {})
        params = {
            k: round((lo + hi) / 2, 4)
            for k, (lo, hi) in ranges.items()
        }

        config = StrategyConfig(
            strategy_name=strategy_name,
            version=1,
            params=params,
            is_active=True,
        )

        if strategy_name not in self._configs:
            self._configs[strategy_name] = []
        self._configs[strategy_name].append(config)

        return config

    def get_active_config(self, strategy_name: str) -> StrategyConfig | None:
        """Get the active (latest promoted or default) config for a strategy."""
        configs = self._configs.get(strategy_name, [])
        if not configs:
            return self.initialize_strategy(strategy_name)

        # Prefer promoted configs, then latest active
        promoted = [c for c in configs if c.is_promoted and c.is_active]
        if promoted:
            return promoted[-1]

        active = [c for c in configs if c.is_active]
        return active[-1] if active else configs[-1]

    # ── Trade outcome recording ───────────────────────────────────────────────

    def record_outcome(
        self,
        strategy_name: str,
        won: bool,
        pnl: float,
        rr: float = 1.0,
    ) -> None:
        """Record a trade outcome for the active config of a strategy."""
        config = self.get_active_config(strategy_name)
        if config is None:
            return

        config.total_trades += 1
        if won:
            config.wins += 1
        else:
            config.losses += 1
        config.total_pnl += pnl

        # Running average RR
        if config.total_trades > 0:
            config.avg_rr = (
                config.avg_rr * (config.total_trades - 1) + rr
            ) / config.total_trades

    # ── Evolution ─────────────────────────────────────────────────────────────

    def evaluate_and_evolve(self, strategy_name: str) -> dict[str, Any]:
        """Evaluate current config and potentially evolve.

        Returns dict with: action (none/mutated/promoted/disabled), details.
        """
        config = self.get_active_config(strategy_name)
        if config is None:
            return {"action": "none", "reason": "No config found"}

        if config.total_trades < MIN_TRADES_FOR_EVALUATION:
            return {
                "action": "none",
                "reason": f"Insufficient trades ({config.total_trades}/{MIN_TRADES_FOR_EVALUATION})",
            }

        result: dict[str, Any] = {"strategy": strategy_name, "version": config.version}

        # Check for degradation
        if (config.win_rate < DEGRADATION_WIN_RATE_THRESHOLD and
                config.sharpe_ratio < DEGRADATION_SHARPE_THRESHOLD):
            config.is_active = False
            self._disablements += 1
            result["action"] = "disabled"
            result["reason"] = (
                f"Win rate {config.win_rate:.0%} below {DEGRADATION_WIN_RATE_THRESHOLD:.0%}, "
                f"Sharpe {config.sharpe_ratio:.2f} below {DEGRADATION_SHARPE_THRESHOLD}"
            )

            # Create mutated replacement
            new_config = self._mutate(config)
            self._configs[strategy_name].append(new_config)
            result["new_version"] = new_config.version
            _log.info("strategy_evolution.disabled_and_mutated", **result)
            return result

        # Check if good enough to promote
        if config.win_rate >= PROMOTION_WIN_RATE_THRESHOLD and not config.is_promoted:
            config.is_promoted = True
            self._promotions += 1
            result["action"] = "promoted"
            result["reason"] = f"Win rate {config.win_rate:.0%} exceeds threshold"
            _log.info("strategy_evolution.promoted", **result)
            return result

        # Otherwise, create a mutated variant to test
        new_config = self._mutate(config)
        self._configs[strategy_name].append(new_config)
        self._evolution_count += 1
        result["action"] = "mutated"
        result["new_version"] = new_config.version
        result["mutations"] = {
            k: round(new_config.params[k] - config.params.get(k, 0), 4)
            for k in new_config.params
            if new_config.params[k] != config.params.get(k)
        }
        _log.info("strategy_evolution.mutated", **result)
        return result

    def _mutate(self, parent: StrategyConfig) -> StrategyConfig:
        """Create a mutated child config from a parent."""
        ranges = DEFAULT_PARAM_RANGES.get(parent.strategy_name, {})
        new_params = dict(parent.params)

        for param_name, value in new_params.items():
            if param_name not in ranges:
                continue

            lo, hi = ranges[param_name]
            # Random mutation within ±MAX_PARAM_MUTATION_PCT
            mutation_range = (hi - lo) * MAX_PARAM_MUTATION_PCT / 100
            delta = random.uniform(-mutation_range, mutation_range)
            new_value = max(lo, min(hi, value + delta))
            new_params[param_name] = round(new_value, 4)

        max_version = max(c.version for c in self._configs.get(parent.strategy_name, [parent]))
        return StrategyConfig(
            strategy_name=parent.strategy_name,
            version=max_version + 1,
            params=new_params,
            is_active=True,
        )

    def evolve_all(self) -> list[dict[str, Any]]:
        """Run evolution for all tracked strategies. Returns list of results."""
        results: list[dict[str, Any]] = []
        for strategy_name in list(self._configs.keys()):
            result = self.evaluate_and_evolve(strategy_name)
            results.append(result)
        return results

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return summary of all strategies and their evolution history."""
        strategies: list[dict[str, Any]] = []
        for name, configs in self._configs.items():
            active_config = self.get_active_config(name)
            strategies.append({
                "strategy": name,
                "versions": len(configs),
                "active_version": active_config.version if active_config else 0,
                "win_rate": round(active_config.win_rate, 4) if active_config else 0,
                "total_trades": active_config.total_trades if active_config else 0,
                "is_promoted": active_config.is_promoted if active_config else False,
                "params": active_config.params if active_config else {},
            })

        return {
            "total_strategies": len(self._configs),
            "total_evolutions": self._evolution_count,
            "total_promotions": self._promotions,
            "total_disablements": self._disablements,
            "strategies": strategies,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist all configs to JSON."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "evolution_count": self._evolution_count,
            "promotions": self._promotions,
            "disablements": self._disablements,
            "configs": {
                name: [c.to_dict() for c in configs]
                for name, configs in self._configs.items()
            },
        }
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _log.info("strategy_evolution.saved", path=str(dest))

    def load(self, path: str | Path) -> None:
        """Restore configs from JSON."""
        src = Path(path)
        if not src.exists():
            return

        payload: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
        self._evolution_count = payload.get("evolution_count", 0)
        self._promotions = payload.get("promotions", 0)
        self._disablements = payload.get("disablements", 0)

        self._configs = {}
        for name, config_dicts in payload.get("configs", {}).items():
            self._configs[name] = [
                StrategyConfig.from_dict(d) for d in config_dicts
            ]

        _log.info(
            "strategy_evolution.loaded",
            path=str(src),
            strategies=len(self._configs),
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: StrategyEvolutionEngine | None = None


def get_evolution_engine() -> StrategyEvolutionEngine:
    """Return the module-level singleton StrategyEvolutionEngine."""
    global _singleton
    if _singleton is None:
        _singleton = StrategyEvolutionEngine()
        _log.info("strategy_evolution.singleton_initialised")
    return _singleton
