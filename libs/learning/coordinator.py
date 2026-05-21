"""
libs.learning.coordinator — SelfTrainingCoordinator orchestrates all learning subsystems.

Manages 8-phase training progression from raw data collection through institutional
behaviour, dispatching trade outcomes to pattern scoring, strategy tuning, and
reinforcement learning as the agent accumulates enough trade history.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any

from libs.core.logging.logger import get_logger

log = get_logger(__name__)


# ── Phase Enum ────────────────────────────────────────────────────────────────

class TrainingPhase(IntEnum):
    DATA_COLLECTION = 1
    STATISTICAL_LEARNING = 2
    PATTERN_INTELLIGENCE = 3
    ADAPTIVE_OPTIMIZATION = 4
    REINFORCEMENT_LEARNING = 5
    PORTFOLIO_INTELLIGENCE = 6
    AUTONOMOUS_EVOLUTION = 7
    INSTITUTIONAL_BEHAVIOR = 8


# ── Default Thresholds ────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS: dict[TrainingPhase, int] = {
    TrainingPhase.DATA_COLLECTION: 0,
    TrainingPhase.STATISTICAL_LEARNING: 20,
    TrainingPhase.PATTERN_INTELLIGENCE: 50,
    TrainingPhase.ADAPTIVE_OPTIMIZATION: 100,
    TrainingPhase.REINFORCEMENT_LEARNING: 200,
    TrainingPhase.PORTFOLIO_INTELLIGENCE: 400,
    TrainingPhase.AUTONOMOUS_EVOLUTION: 800,
    TrainingPhase.INSTITUTIONAL_BEHAVIOR: 1500,
}


# ── SelfTrainingCoordinator ───────────────────────────────────────────────────

class SelfTrainingCoordinator:
    """Orchestrates all learning subsystems with 8-phase training progression.

    Parameters
    ----------
    phase_thresholds:
        Dict mapping TrainingPhase → minimum trade count to enter that phase.
        Defaults to DEFAULT_THRESHOLDS when None.
    tune_interval:
        Number of trades between automatic tune cycles (default 20).
    """

    def __init__(
        self,
        phase_thresholds: dict[TrainingPhase, int] | None = None,
        tune_interval: int = 20,
    ) -> None:
        self._thresholds: dict[TrainingPhase, int] = (
            phase_thresholds if phase_thresholds is not None else DEFAULT_THRESHOLDS
        )
        self._tune_interval = tune_interval

        self._total_trades: int = 0
        self._wins: int = 0
        self._losses: int = 0
        self._tune_cycles: int = 0
        self._current_phase: TrainingPhase = TrainingPhase.DATA_COLLECTION
        self._started_at: str = datetime.now(tz=timezone.utc).isoformat()
        self._phase_history: list[dict[str, Any]] = [
            {
                "phase": TrainingPhase.DATA_COLLECTION.name,
                "phase_number": int(TrainingPhase.DATA_COLLECTION),
                "entered_at": self._started_at,
                "trades_at_entry": 0,
            }
        ]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_phase(self) -> TrainingPhase:
        """Current training phase."""
        return self._current_phase

    @property
    def total_trades(self) -> int:
        """Total number of trades recorded so far."""
        return self._total_trades

    # ── Trade Recording ───────────────────────────────────────────────────────

    def record_trade(self, strategy: str, won: bool, pnl: float) -> None:
        """Increment trade counters.

        Parameters
        ----------
        strategy: Strategy name (unused at this level; passed for context).
        won:      True if the trade was a win.
        pnl:      Realised profit/loss.
        """
        self._total_trades += 1
        if won:
            self._wins += 1
        else:
            self._losses += 1

    def on_trade_close(
        self,
        strategy: str,
        won: bool,
        pnl: float,
        rr: float = 1.0,
        confidence: float = 0.5,
        patterns: list[str] | None = None,
        regime: str = "unknown",
        disciplined_exit: bool = True,
        regime_aligned: bool = True,
    ) -> None:
        """Process a closed trade, dispatching to active subsystems.

        Phase gating:
        - Phase 2+: PatternScoreStore
        - Phase 4+: StrategyParameterTuner
        - Phase 5+: RewardEngine

        Auto-tunes every tune_interval trades.  Phase advancement is checked
        after each trade.  Subsystem failures are caught and logged but never
        propagated.

        Parameters
        ----------
        strategy:        Name of the strategy that generated the trade.
        won:             True if the trade was profitable.
        pnl:             Realised profit/loss.
        rr:              Realised risk-reward ratio (default 1.0).
        confidence:      Signal confidence at trade entry (default 0.5).
        patterns:        List of pattern names active at entry (default empty).
        regime:          Market regime label at entry (default "unknown").
        disciplined_exit: Trade closed at planned exit (default True).
        regime_aligned:  Trade direction matched market regime (default True).
        """
        self.record_trade(strategy, won, pnl)

        active_patterns = patterns or []

        # Phase 2+: Pattern scoring
        if self._current_phase >= TrainingPhase.STATISTICAL_LEARNING:
            try:
                from libs.learning.pattern_scorer import get_pattern_store
                store = get_pattern_store()
                for pattern in active_patterns:
                    store.record(pattern=pattern, regime=regime, won=won, rr=rr)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "coordinator.pattern_scorer_error",
                    error=str(exc),
                )

        # Phase 4+: Strategy parameter tuning
        if self._current_phase >= TrainingPhase.ADAPTIVE_OPTIMIZATION:
            try:
                from libs.learning.strategy_tuner import get_strategy_tuner
                tuner = get_strategy_tuner()
                tuner.record_outcome(
                    strategy=strategy,
                    won=won,
                    confidence=confidence,
                    rr=rr,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "coordinator.strategy_tuner_error",
                    error=str(exc),
                )

        # Phase 5+: Reinforcement learning reward/penalty
        if self._current_phase >= TrainingPhase.REINFORCEMENT_LEARNING:
            try:
                from libs.learning.reward_engine import get_reward_engine
                engine = get_reward_engine()
                if won:
                    engine.reward(
                        strategy=strategy,
                        pnl=pnl,
                        rr=rr,
                        disciplined_exit=disciplined_exit,
                        regime_aligned=regime_aligned,
                    )
                else:
                    engine.penalize(
                        strategy=strategy,
                        pnl=pnl,
                        rr=rr,
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "coordinator.reward_engine_error",
                    error=str(exc),
                )

        # Auto-tune at interval
        if self._total_trades % self._tune_interval == 0:
            self._run_tune_cycle()

        # Check for phase advancement
        self.evaluate_phase()

    # ── Tune Cycle ────────────────────────────────────────────────────────────

    def _run_tune_cycle(self) -> None:
        """Run a full tune cycle: tune all strategies and apply pattern decay."""
        try:
            from libs.learning.strategy_tuner import get_strategy_tuner
            get_strategy_tuner().tune_all()
        except Exception as exc:  # noqa: BLE001
            log.warning("coordinator.tune_all_error", error=str(exc))

        try:
            from libs.learning.pattern_scorer import get_pattern_store
            get_pattern_store().apply_decay()
        except Exception as exc:  # noqa: BLE001
            log.warning("coordinator.apply_decay_error", error=str(exc))

        self._tune_cycles += 1
        log.debug(
            "coordinator.tune_cycle",
            cycle=self._tune_cycles,
            total_trades=self._total_trades,
        )

    # ── Phase Evaluation ──────────────────────────────────────────────────────

    def evaluate_phase(self) -> TrainingPhase:
        """Advance phase based on total_trades vs thresholds.

        Iterates phases from highest to lowest and promotes to the highest
        phase whose threshold has been reached.  Records phase transitions
        in phase_history.

        Returns
        -------
        The current (possibly updated) TrainingPhase.
        """
        new_phase = TrainingPhase.DATA_COLLECTION
        for phase in reversed(list(TrainingPhase)):
            threshold = self._thresholds.get(phase, 0)
            if self._total_trades >= threshold:
                new_phase = phase
                break

        if new_phase != self._current_phase:
            log.info(
                "coordinator.phase_advance",
                from_phase=self._current_phase.name,
                to_phase=new_phase.name,
                total_trades=self._total_trades,
            )
            self._current_phase = new_phase
            self._phase_history.append(
                {
                    "phase": new_phase.name,
                    "phase_number": int(new_phase),
                    "entered_at": datetime.now(tz=timezone.utc).isoformat(),
                    "trades_at_entry": self._total_trades,
                }
            )

        return self._current_phase

    # ── Training Progress Report ───────────────────────────────────────────────

    def get_training_progress(self) -> dict[str, Any]:
        """Return a comprehensive report of current training state.

        Keys
        ----
        phase, phase_number, phase_progress_pct, next_phase,
        next_phase_at_trades, total_trades, wins, losses, win_rate,
        tune_cycles, started_at, phase_history, active_subsystems,
        plus optional subsystem stats (pattern_store_stats,
        strategy_tuner_stats, reward_engine_stats).
        """
        current = self._current_phase
        phases = list(TrainingPhase)
        current_idx = phases.index(current)

        # Phase progress percentage
        current_threshold = self._thresholds.get(current, 0)
        next_phase: TrainingPhase | None = None
        next_phase_at: int | None = None

        if current_idx + 1 < len(phases):
            next_phase = phases[current_idx + 1]
            next_phase_at = self._thresholds.get(next_phase, None)

        if next_phase_at is not None and next_phase_at > current_threshold:
            trades_in_phase = self._total_trades - current_threshold
            phase_span = next_phase_at - current_threshold
            phase_progress_pct = min(100.0, round(trades_in_phase / phase_span * 100, 2))
        else:
            # Already at max phase
            phase_progress_pct = 100.0

        win_rate = (
            round(self._wins / self._total_trades, 4)
            if self._total_trades > 0
            else 0.0
        )

        report: dict[str, Any] = {
            "phase": current.name,
            "phase_number": int(current),
            "phase_progress_pct": phase_progress_pct,
            "next_phase": next_phase.name if next_phase else None,
            "next_phase_at_trades": next_phase_at,
            "total_trades": self._total_trades,
            "wins": self._wins,
            "losses": self._losses,
            "win_rate": win_rate,
            "tune_cycles": self._tune_cycles,
            "started_at": self._started_at,
            "phase_history": list(self._phase_history),
            "active_subsystems": self._active_subsystems(),
        }

        # Optional subsystem stats (best-effort)
        try:
            from libs.learning.pattern_scorer import get_pattern_store
            report["pattern_store_stats"] = get_pattern_store().get_all_stats()
        except Exception:  # noqa: BLE001
            pass

        try:
            from libs.learning.strategy_tuner import get_strategy_tuner
            report["strategy_tuner_stats"] = get_strategy_tuner().get_all_stats()
        except Exception:  # noqa: BLE001
            pass

        try:
            from libs.learning.reward_engine import get_reward_engine
            report["reward_engine_stats"] = get_reward_engine().get_stats()
        except Exception:  # noqa: BLE001
            pass

        return report

    # ── Active Subsystems ─────────────────────────────────────────────────────

    def _active_subsystems(self) -> list[str]:
        """Return the list of active subsystem names for the current phase."""
        active: list[str] = []
        phase = self._current_phase

        if phase >= TrainingPhase.STATISTICAL_LEARNING:
            active.append("pattern_scorer")
        if phase >= TrainingPhase.ADAPTIVE_OPTIMIZATION:
            active.append("strategy_tuner")
        if phase >= TrainingPhase.REINFORCEMENT_LEARNING:
            active.append("reward_engine")

        return active

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist coordinator state to a JSON file.

        Parameters
        ----------
        path: Destination file path (parent directories are created).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "total_trades": self._total_trades,
            "wins": self._wins,
            "losses": self._losses,
            "tune_cycles": self._tune_cycles,
            "current_phase": int(self._current_phase),
            "started_at": self._started_at,
            "phase_history": self._phase_history,
            "tune_interval": self._tune_interval,
            "thresholds": {
                str(int(phase)): threshold
                for phase, threshold in self._thresholds.items()
            },
        }
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.debug("coordinator.saved", path=str(dest))

    def load(self, path: str | Path) -> None:
        """Restore coordinator state from a previously saved JSON file.

        Replaces all in-memory state.

        Parameters
        ----------
        path: Source file path.
        """
        src = Path(path)
        payload: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))

        self._total_trades = payload.get("total_trades", 0)
        self._wins = payload.get("wins", 0)
        self._losses = payload.get("losses", 0)
        self._tune_cycles = payload.get("tune_cycles", 0)
        self._current_phase = TrainingPhase(payload.get("current_phase", 1))
        self._started_at = payload.get("started_at", self._started_at)
        self._phase_history = payload.get("phase_history", [])
        self._tune_interval = payload.get("tune_interval", self._tune_interval)

        raw_thresholds = payload.get("thresholds", {})
        if raw_thresholds:
            self._thresholds = {
                TrainingPhase(int(k)): v for k, v in raw_thresholds.items()
            }

        log.debug("coordinator.loaded", path=str(src))


# ── Module-level singleton ────────────────────────────────────────────────────

_coordinator: SelfTrainingCoordinator | None = None


def get_coordinator() -> SelfTrainingCoordinator:
    """Return the module-level singleton SelfTrainingCoordinator.

    The instance is created lazily on the first call using default thresholds
    and tune interval.
    """
    global _coordinator
    if _coordinator is None:
        _coordinator = SelfTrainingCoordinator()
        log.info("coordinator.singleton_initialised")
    return _coordinator
