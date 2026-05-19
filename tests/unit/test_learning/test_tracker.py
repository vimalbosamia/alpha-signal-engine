"""
Unit tests for libs/learning/tracker.py.

8 tests covering win/loss recording, category filtering, weight suggestions,
and the safety gate.  All tests are fully isolated — no I/O, no side-effects.
"""
from __future__ import annotations

import pytest

from libs.learning.tracker import (
    PerformanceRecord,
    PerformanceTracker,
    SafetyGateResult,
    WeightUpdate,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tracker_with_outcomes(
    name: str,
    category: str,
    wins: int,
    losses: int,
    pnl_per_trade: float = 1.0,
) -> PerformanceTracker:
    """Return a fresh tracker that already has *wins* + *losses* recorded."""
    tracker = PerformanceTracker()
    for _ in range(wins):
        tracker.record_outcome(name, category, won=True, pnl=pnl_per_trade)
    for _ in range(losses):
        tracker.record_outcome(name, category, won=False, pnl=-pnl_per_trade)
    return tracker


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRecordOutcomeWin:
    """record_outcome with won=True increments the correct counters."""

    def test_record_outcome_win(self) -> None:
        # Arrange
        tracker = PerformanceTracker()

        # Act
        tracker.record_outcome("RSI", "indicator", won=True, pnl=50.0)

        # Assert
        rec = tracker.get_record("RSI", "indicator")
        assert rec is not None
        assert rec.total == 1
        assert rec.wins == 1
        assert rec.losses == 0
        assert rec.win_rate == pytest.approx(1.0)
        assert rec.avg_pnl == pytest.approx(50.0)
        assert rec.last_updated != ""


class TestRecordOutcomeLoss:
    """record_outcome with won=False increments loss counter and win_rate is correct."""

    def test_record_outcome_loss(self) -> None:
        # Arrange — record one win then one loss
        tracker = PerformanceTracker()
        tracker.record_outcome("MACD", "indicator", won=True, pnl=100.0)

        # Act
        tracker.record_outcome("MACD", "indicator", won=False, pnl=-40.0)

        # Assert
        rec = tracker.get_record("MACD", "indicator")
        assert rec is not None
        assert rec.total == 2
        assert rec.wins == 1
        assert rec.losses == 1
        assert rec.win_rate == pytest.approx(0.5)
        assert rec.avg_pnl == pytest.approx(30.0)  # (100 - 40) / 2


class TestGetRecordMissing:
    """get_record returns None for an entity that has never been recorded."""

    def test_get_record_missing(self) -> None:
        # Arrange
        tracker = PerformanceTracker()

        # Act
        result = tracker.get_record("unknown_strategy", "strategy")

        # Assert
        assert result is None


class TestGetAllByCategory:
    """get_all with a category filter returns only matching records."""

    def test_get_all_by_category(self) -> None:
        # Arrange — record entities across two different categories
        tracker = PerformanceTracker()
        tracker.record_outcome("breakout_long", "strategy", won=True)
        tracker.record_outcome("RSI", "indicator", won=False)
        tracker.record_outcome("EMA", "indicator", won=True)

        # Act
        indicators = tracker.get_all(category="indicator")

        # Assert — only the two indicator records are returned
        assert len(indicators) == 2
        names = {r.name for r in indicators}
        assert names == {"RSI", "EMA"}
        assert all(r.category == "indicator" for r in indicators)


class TestSuggestWeightIncrease:
    """suggest_weight_updates proposes an increase when win-rate > 60%."""

    def test_suggest_weight_increase(self) -> None:
        # Arrange — 65% win-rate over 20 trades (13 wins, 7 losses)
        tracker = _tracker_with_outcomes("RSI", "indicator", wins=13, losses=7)
        current_weights = {"RSI": 0.40}

        # Act
        updates = tracker.suggest_weight_updates(current_weights, min_trades=20)

        # Assert
        assert len(updates) == 1
        upd = updates[0]
        assert upd.name == "RSI"
        assert upd.new_weight > upd.old_weight
        assert upd.new_weight == pytest.approx(0.40 * 1.10)  # +10%
        assert "increase" in upd.reason


class TestSuggestWeightDecrease:
    """suggest_weight_updates proposes a decrease when win-rate < 35%."""

    def test_suggest_weight_decrease(self) -> None:
        # Arrange — 30% win-rate over 20 trades (6 wins, 14 losses)
        tracker = _tracker_with_outcomes("SMA", "indicator", wins=6, losses=14)
        current_weights = {"SMA": 0.50}

        # Act
        updates = tracker.suggest_weight_updates(current_weights, min_trades=20)

        # Assert
        assert len(updates) == 1
        upd = updates[0]
        assert upd.name == "SMA"
        assert upd.new_weight < upd.old_weight
        assert upd.new_weight == pytest.approx(0.50 * 0.80)  # -20%
        assert "decrease" in upd.reason


class TestSuggestNoChangeInsufficientData:
    """suggest_weight_updates emits no suggestion when trade count < min_trades."""

    def test_suggest_no_change_insufficient_data(self) -> None:
        # Arrange — only 5 trades recorded (below default min_trades=20)
        tracker = _tracker_with_outcomes("breakout_long", "strategy", wins=4, losses=1)
        current_weights = {"breakout_long": 0.30}

        # Act
        updates = tracker.suggest_weight_updates(current_weights, min_trades=20)

        # Assert — no suggestion because data is insufficient
        assert updates == []


class TestSafetyGateBlocksLargeChange:
    """safety_gate rejects a proposed weight change that exceeds 30%."""

    def test_safety_gate_blocks_large_change(self) -> None:
        # Arrange — old weight 0.40, new weight 0.60 → 50% increase (> 30%)
        tracker = PerformanceTracker()
        # Populate tracker with enough entities to pass the "3 entities" check.
        for entity, cat in [("A", "strategy"), ("B", "strategy"), ("C", "strategy")]:
            for _ in range(25):
                tracker.record_outcome(entity, cat, won=True)

        old_weights = {"A": 0.40, "B": 0.30, "C": 0.30}
        # A's weight jumps from 0.40 → 0.60: delta 0.20, which is 50% of 0.40
        new_weights = {"A": 0.60, "B": 0.20, "C": 0.20}

        # Act
        result = tracker.safety_gate(new_weights, old_weights)

        # Assert
        assert isinstance(result, SafetyGateResult)
        assert result.approved is False
        assert len(result.checks_failed) >= 1
        # At least one failure must reference the offending entity
        assert any("A" in f for f in result.checks_failed)
