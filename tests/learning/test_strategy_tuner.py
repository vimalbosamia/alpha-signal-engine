"""
Unit tests for libs/learning/strategy_tuner.py.

6 tests covering default params, confidence and R:R tuning in both directions,
bounds enforcement, and JSON persistence.  All tests are fully isolated.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from libs.learning.strategy_tuner import (
    CONFIDENCE_CEILING,
    CONFIDENCE_DEFAULT,
    CONFIDENCE_FLOOR,
    MIN_TRADES_TO_TUNE,
    RR_CEILING,
    RR_DEFAULT,
    RR_FLOOR,
    TUNE_STEP,
    StrategyParameterTuner,
    StrategyParams,
    TradeRecord,
    get_strategy_tuner,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tuner_with_outcomes(
    strategy: str,
    wins: int,
    losses: int,
    win_confidence: float = 0.70,
    loss_confidence: float = 0.62,
    win_rr: float = 2.5,
    loss_rr: float = 1.1,
) -> StrategyParameterTuner:
    """Return a fresh tuner pre-loaded with *wins* + *losses* for *strategy*."""
    tuner = StrategyParameterTuner()
    for _ in range(wins):
        tuner.record_outcome(strategy, won=True, confidence=win_confidence, rr=win_rr)
    for _ in range(losses):
        tuner.record_outcome(strategy, won=False, confidence=loss_confidence, rr=loss_rr)
    return tuner


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestInitialParams:
    """get_params returns canonical defaults for an unknown strategy."""

    def test_initial_params(self) -> None:
        # Arrange
        tuner = StrategyParameterTuner()

        # Act
        params = tuner.get_params("breakout_long")

        # Assert
        assert params.min_confidence == pytest.approx(CONFIDENCE_DEFAULT)
        assert params.min_rr == pytest.approx(RR_DEFAULT)
        assert params.tune_count == 0


class TestTuneIncreasesConfidenceAfterLosses:
    """tune() raises min_confidence when win_rate < 0.40 (mostly losses)."""

    def test_tune_increases_confidence_after_losses(self) -> None:
        # Arrange — 10 losses, 0 wins → win_rate = 0.0 < 0.40
        tuner = _tuner_with_outcomes("rsi_long", wins=0, losses=10)

        # Act
        params = tuner.tune("rsi_long")

        # Assert
        assert params.min_confidence > CONFIDENCE_DEFAULT
        assert params.min_confidence == pytest.approx(CONFIDENCE_DEFAULT + TUNE_STEP)
        assert params.tune_count == 1


class TestTuneDecreasesConfidenceAfterWins:
    """tune() lowers min_confidence when win_rate > 0.60 with 12+ trades."""

    def test_tune_decreases_confidence_after_wins(self) -> None:
        # Arrange — 15 wins, 0 losses → win_rate = 1.0 > 0.60, total = 15 >= 12
        tuner = _tuner_with_outcomes("ema_cross", wins=15, losses=0)

        # Act
        params = tuner.tune("ema_cross")

        # Assert
        assert params.min_confidence <= CONFIDENCE_DEFAULT
        assert params.min_confidence == pytest.approx(CONFIDENCE_DEFAULT - TUNE_STEP * 0.5)
        assert params.tune_count == 1


class TestTuneAdjustsMinRR:
    """tune() raises min_rr when avg loss R:R is low and win_rate < 0.50."""

    def test_tune_adjusts_min_rr(self) -> None:
        # Arrange — 10 losses with rr=1.0 (< 1.5), 0 wins → win_rate = 0.0 < 0.50
        tuner = _tuner_with_outcomes(
            "macd_long",
            wins=0,
            losses=10,
            loss_rr=1.0,
        )

        # Act
        params = tuner.tune("macd_long")

        # Assert
        assert params.min_rr > RR_DEFAULT
        assert params.tune_count == 1


class TestTuneRespectsBounds:
    """Repeated tuning on a losing strategy never exceeds ceilings."""

    def test_tune_respects_bounds(self) -> None:
        # Arrange — 100 losses → would push confidence and rr way past ceiling
        # without clamping
        tuner = _tuner_with_outcomes(
            "scalp_short",
            wins=0,
            losses=100,
            loss_rr=0.5,
        )

        # Act — tune many times (each call has all 100 records available)
        for _ in range(50):
            tuner.tune("scalp_short")

        params = tuner.get_params("scalp_short")

        # Assert — values stay within defined ceilings
        assert params.min_confidence <= CONFIDENCE_CEILING
        assert params.min_rr <= RR_CEILING
        assert params.min_confidence >= CONFIDENCE_FLOOR
        assert params.min_rr >= RR_FLOOR


class TestPersistence:
    """save() + load() round-trips the full tuner state correctly."""

    def test_persistence(self, tmp_path: Path) -> None:
        # Arrange
        tuner = _tuner_with_outcomes("swing_long", wins=5, losses=5)
        tuner.tune("swing_long")
        original_params = tuner.get_params("swing_long")
        save_path = tmp_path / "tuner_state.json"

        # Act — save then load into a fresh tuner
        tuner.save(save_path)
        loaded = StrategyParameterTuner()
        loaded.load(save_path)

        # Assert — params are preserved
        loaded_params = loaded.get_params("swing_long")
        assert loaded_params.min_confidence == pytest.approx(original_params.min_confidence)
        assert loaded_params.min_rr == pytest.approx(original_params.min_rr)
        assert loaded_params.tune_count == original_params.tune_count

        # Assert — history is preserved
        assert len(loaded._history["swing_long"]) == 10

        # Assert — file is valid JSON
        raw = json.loads(save_path.read_text())
        assert "params" in raw
        assert "history" in raw
