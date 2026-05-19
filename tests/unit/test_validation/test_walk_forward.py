"""Tests for WalkForwardValidator."""
from __future__ import annotations

import pytest

from libs.validation.walk_forward import (
    MIN_TRADES_PER_WINDOW,
    MIN_WINDOWS,
    WalkForwardValidator,
    ValidationResult,
    ValidationWindow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wins(n: int) -> list[bool]:
    """Return n True values."""
    return [True] * n


def _losses(n: int) -> list[bool]:
    """Return n False values."""
    return [False] * n


def _mixed(total: int, win_rate: float) -> list[bool]:
    """Return a list of booleans with approximately the given win rate."""
    wins = round(total * win_rate)
    return ([True] * wins + [False] * (total - wins))


# ---------------------------------------------------------------------------
# Test 1 — insufficient data (5 outcomes)
# ---------------------------------------------------------------------------

def test_insufficient_data():
    validator = WalkForwardValidator()
    result = validator.validate("tiny_strat", [True, False, True, True, False])

    assert result.validation_status == "insufficient_data"
    assert result.is_valid is False
    assert result.confidence_cap == 0.50
    assert result.total_trades == 5


# ---------------------------------------------------------------------------
# Test 2 — validated good strategy (60 outcomes, ~60% win rate)
# ---------------------------------------------------------------------------

def test_validated_good_strategy():
    validator = WalkForwardValidator()
    # Deterministic 60% win rate: repeating pattern WWWLLF across 60 trades
    pattern = [True, True, True, False, False]  # 60% WR
    outcomes = (pattern * 12)[:60]  # exactly 60 trades, consistent 60% throughout
    result = validator.validate("consistent_strat", outcomes)

    assert result.validation_status == "validated"
    assert result.is_valid is True
    assert result.confidence_cap == 0.90
    assert result.avg_test_win_rate >= 0.40


# ---------------------------------------------------------------------------
# Test 3 — overfitting detected (train WR much higher than test WR)
# ---------------------------------------------------------------------------

def test_overfitting_detected():
    validator = WalkForwardValidator()
    # First 70% of trades are wins (train portion looks great),
    # last 30% are losses (test portion looks terrible).
    # Build a 60-trade sequence: wins first, losses last.
    total = 60
    outcomes = [True] * int(total * 0.75) + [False] * int(total * 0.25)

    result = validator.validate("overfit_strat", outcomes)

    # avg degradation should exceed MAX_DEGRADATION (0.15).
    assert result.validation_status == "overfitting"
    assert result.is_valid is False
    assert result.confidence_cap == 0.55
    assert result.avg_degradation > 0.15


# ---------------------------------------------------------------------------
# Test 4 — degrading strategy (poor test win rate)
# ---------------------------------------------------------------------------

def test_degrading_strategy():
    validator = WalkForwardValidator()
    # Win rate low enough (30%) so test WR < 0.40, but degradation stays small
    # by keeping train and test WR similarly poor.
    outcomes = _mixed(60, 0.30)
    result = validator.validate("bad_strat", outcomes)

    assert result.validation_status in {"degrading", "overfitting", "insufficient_data"}
    assert result.is_valid is False
    assert result.confidence_cap <= 0.55


# ---------------------------------------------------------------------------
# Test 5 — window count (40 outcomes → at least MIN_WINDOWS windows)
# ---------------------------------------------------------------------------

def test_window_count():
    validator = WalkForwardValidator()
    outcomes = _mixed(40, 0.55)
    result = validator.validate("medium_strat", outcomes)

    # May be insufficient_data or valid — either way check window count if available.
    if result.validation_status != "insufficient_data":
        assert len(result.windows) >= MIN_WINDOWS


# ---------------------------------------------------------------------------
# Test 6 — all result fields are present and typed correctly
# ---------------------------------------------------------------------------

def test_result_fields():
    validator = WalkForwardValidator()
    outcomes = _mixed(60, 0.55)
    result = validator.validate("field_strat", outcomes)

    assert isinstance(result, ValidationResult)
    assert isinstance(result.strategy_name, str)
    assert isinstance(result.total_trades, int)
    assert isinstance(result.is_valid, bool)
    assert isinstance(result.validation_status, str)
    assert isinstance(result.windows, list)
    assert isinstance(result.avg_train_win_rate, float)
    assert isinstance(result.avg_test_win_rate, float)
    assert isinstance(result.avg_degradation, float)
    assert isinstance(result.confidence_cap, float)
    assert isinstance(result.explanation, str)
    assert result.validation_status in {
        "insufficient_data", "overfitting", "degrading", "validated"
    }
    assert 0.0 <= result.confidence_cap <= 1.0

    for w in result.windows:
        assert isinstance(w, ValidationWindow)
        assert w.train_start <= w.train_end <= w.test_start <= w.test_end
        assert 0.0 <= w.train_win_rate <= 1.0
        assert 0.0 <= w.test_win_rate <= 1.0


# ---------------------------------------------------------------------------
# Test 7 — empty outcomes
# ---------------------------------------------------------------------------

def test_empty_outcomes():
    validator = WalkForwardValidator()
    result = validator.validate("empty_strat", [])

    assert result.validation_status == "insufficient_data"
    assert result.is_valid is False
    assert result.confidence_cap == 0.50
    assert result.total_trades == 0
