"""
Unit tests for libs/learning/coordinator.py.

5 tests covering initial phase, phase advancement, on_trade_close integration,
training progress report structure, and auto-tune triggering.
"""
from __future__ import annotations

import pytest

from libs.learning.coordinator import (
    DEFAULT_THRESHOLDS,
    SelfTrainingCoordinator,
    TrainingPhase,
    get_coordinator,
)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestInitialPhase:
    """Coordinator starts in DATA_COLLECTION phase with zero trades."""

    def test_initial_phase_is_data_collection(self) -> None:
        # Arrange
        coord = SelfTrainingCoordinator()

        # Act / Assert
        assert coord.current_phase == TrainingPhase.DATA_COLLECTION
        assert coord.total_trades == 0


class TestPhaseAdvancement:
    """Phase advances once enough trades are recorded."""

    def test_phase_advances_after_enough_trades(self) -> None:
        # Arrange — custom thresholds: phase 2 requires 10 trades
        thresholds = {
            TrainingPhase.DATA_COLLECTION: 0,
            TrainingPhase.STATISTICAL_LEARNING: 10,
            TrainingPhase.PATTERN_INTELLIGENCE: 50,
            TrainingPhase.ADAPTIVE_OPTIMIZATION: 100,
            TrainingPhase.REINFORCEMENT_LEARNING: 200,
            TrainingPhase.PORTFOLIO_INTELLIGENCE: 400,
            TrainingPhase.AUTONOMOUS_EVOLUTION: 800,
            TrainingPhase.INSTITUTIONAL_BEHAVIOR: 1500,
        }
        coord = SelfTrainingCoordinator(phase_thresholds=thresholds)

        # Act — record 11 trades (exceeds phase 2 threshold of 10)
        for i in range(11):
            coord.on_trade_close(
                strategy="momentum",
                won=(i % 2 == 0),
                pnl=10.0 if i % 2 == 0 else -5.0,
            )

        # Assert — should have advanced to STATISTICAL_LEARNING
        assert coord.current_phase == TrainingPhase.STATISTICAL_LEARNING
        assert coord.total_trades == 11


class TestOnTradeCloseDoesNotRaise:
    """on_trade_close should not raise even if subsystems are unavailable."""

    def test_on_trade_close_triggers_all_systems(self) -> None:
        # Arrange — force a high phase so all subsystems are gated in
        thresholds = {phase: 0 for phase in TrainingPhase}
        coord = SelfTrainingCoordinator(phase_thresholds=thresholds)

        # Act / Assert — must not raise
        coord.on_trade_close(
            strategy="momentum",
            won=True,
            pnl=50.0,
            rr=2.5,
            confidence=0.75,
            patterns=["hammer", "engulfing_bull"],
            regime="trending_up",
            disciplined_exit=True,
            regime_aligned=True,
        )


class TestTrainingProgressReport:
    """get_training_progress returns a dict with all required keys."""

    _REQUIRED_KEYS = {
        "phase",
        "phase_number",
        "phase_progress_pct",
        "next_phase",
        "next_phase_at_trades",
        "total_trades",
        "wins",
        "losses",
        "win_rate",
        "tune_cycles",
        "started_at",
        "phase_history",
        "active_subsystems",
    }

    def test_training_progress_report(self) -> None:
        # Arrange
        coord = SelfTrainingCoordinator()
        coord.on_trade_close(strategy="breakout", won=True, pnl=30.0)

        # Act
        report = coord.get_training_progress()

        # Assert — all required keys present
        for key in self._REQUIRED_KEYS:
            assert key in report, f"Missing key: {key}"

        # Sanity-check values
        assert report["total_trades"] == 1
        assert report["wins"] == 1
        assert report["losses"] == 0
        assert isinstance(report["active_subsystems"], list)
        assert isinstance(report["phase_history"], list)


class TestAutoTuneAtInterval:
    """tune_cycles increments after tune_interval trades are recorded."""

    def test_auto_tune_at_interval(self) -> None:
        # Arrange — tune every 5 trades
        coord = SelfTrainingCoordinator(tune_interval=5)

        # Act — record 6 trades; first tune fires at trade 5
        for i in range(6):
            coord.on_trade_close(
                strategy="trend",
                won=(i % 2 == 0),
                pnl=10.0 if i % 2 == 0 else -5.0,
            )

        # Assert — at least one tune cycle has run
        assert coord._tune_cycles >= 1
