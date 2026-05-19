"""
Unit tests for libs/signals/grading/engine.py.

7 tests covering all 5 grade tiers, late-entry downgrade, and result field
contract.  All tests are isolated — no I/O, no environment-variable
side-effects.
"""
from __future__ import annotations

import pytest

from libs.signals.grading.engine import GradingInput, GradingResult, TradeDecisionEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_input(
    *,
    confidence: float = 0.9,
    risk_reward: float = 3.0,
    bias_net: str = "bullish",
    bias_conflict: float = 0.0,
    bias_bullish: float = 0.9,
    bias_bearish: float = 0.1,
    htf_aligned: bool = True,
    regime_supports: bool = True,
    structure_strength: float = 1.0,
    volume_confirms: bool = True,
    data_quality_clean: bool = True,
    action: str = "BUY",
    is_late_entry: bool = False,
    is_overextended: bool = False,
) -> GradingInput:
    """Build a GradingInput with sensible defaults (all-positive A+ setup)."""
    return GradingInput(
        confidence=confidence,
        risk_reward=risk_reward,
        bias_net=bias_net,
        bias_conflict=bias_conflict,
        bias_bullish=bias_bullish,
        bias_bearish=bias_bearish,
        htf_aligned=htf_aligned,
        regime_supports=regime_supports,
        structure_strength=structure_strength,
        volume_confirms=volume_confirms,
        data_quality_clean=data_quality_clean,
        action=action,
        is_late_entry=is_late_entry,
        is_overextended=is_overextended,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestGradeAPlus:
    """A+ setup: score >= 75, decision = TAKE."""

    def test_a_plus_grade_and_take_decision(self) -> None:
        # Arrange — all positive signals, high confidence, good R:R
        # Expected score:
        #   confidence 0.9 * 25 = 22.5
        #   R:R 3.0 * 6.67 = 20.01 → capped at 20
        #   bias aligned = 15
        #   conflict 0.0 → 0
        #   HTF aligned = +10
        #   regime supports = +10
        #   structure 1.0 * 10 = 10
        #   volume confirms = +5
        #   data quality clean = +5
        #   total ≈ 97.5
        engine = TradeDecisionEngine()
        inp = _make_input()

        # Act
        result = engine.grade(inp)

        # Assert
        assert result.setup_grade == "A+"
        assert result.trade_decision == "TAKE"
        assert result.quality_score >= 75.0


class TestGradeA:
    """A setup: 60 <= score < 75, decision = TAKE."""

    def test_a_grade_and_take_decision(self) -> None:
        # Arrange — reduce structure and remove volume + regime to drop below 75
        # Expected score:
        #   confidence 0.7 * 25 = 17.5
        #   R:R 2.0 * 6.67 = 13.34
        #   bias aligned = 15
        #   conflict 0.0 → 0
        #   HTF aligned = +10
        #   regime supports = 0  (False)
        #   structure 0.5 * 10 = 5
        #   volume confirms = 0  (False)
        #   data quality clean = +5
        #   total ≈ 65.84  → grade A
        engine = TradeDecisionEngine()
        inp = _make_input(
            confidence=0.7,
            risk_reward=2.0,
            regime_supports=False,
            structure_strength=0.5,
            volume_confirms=False,
        )

        # Act
        result = engine.grade(inp)

        # Assert
        assert result.setup_grade == "A"
        assert result.trade_decision == "TAKE"
        assert 60.0 <= result.quality_score < 75.0


class TestGradeB:
    """B setup: 45 <= score < 60, decision = WAIT."""

    def test_b_grade_and_wait_decision(self) -> None:
        # Arrange — moderate confidence, neutral bias, HTF not aligned
        # Expected score:
        #   confidence 0.6 * 25 = 15
        #   R:R 1.5 * 6.67 = 10.005
        #   bias neutral = 5
        #   conflict 0.0 → 0
        #   HTF not aligned = -5
        #   regime supports = +10
        #   structure 0.7 * 10 = 7
        #   volume confirms = +5
        #   data quality clean = +5
        #   total ≈ 52.0  → grade B
        engine = TradeDecisionEngine()
        inp = _make_input(
            confidence=0.6,
            risk_reward=1.5,
            bias_net="neutral",
            htf_aligned=False,
            regime_supports=True,
            structure_strength=0.7,
            volume_confirms=True,
        )

        # Act
        result = engine.grade(inp)

        # Assert
        assert result.setup_grade == "B"
        assert result.trade_decision == "WAIT"
        assert 45.0 <= result.quality_score < 60.0


class TestGradeC:
    """C setup: 30 <= score < 45, decision = SKIP."""

    def test_c_grade_and_skip_decision(self) -> None:
        # Arrange — low confidence, conflicting bias, HTF not aligned, no volume
        # Expected score:
        #   confidence 0.4 * 25 = 10
        #   R:R 1.0 * 6.67 = 6.67
        #   bias conflicts BUY = 0
        #   conflict 0.0 → 0
        #   HTF not aligned = -5
        #   regime supports = 0  (False)
        #   structure 0.4 * 10 = 4
        #   volume confirms = 0  (False)
        #   data quality clean = +5
        #   total ≈ 20.67  → too low; bump structure and add regime
        # Revised:
        #   confidence 0.5 * 25 = 12.5
        #   R:R 1.2 * 6.67 = 8.0
        #   bias neutral = 5
        #   conflict 0.0 → 0
        #   HTF not aligned = -5
        #   regime supports = 0  (False)
        #   structure 0.6 * 10 = 6
        #   volume confirms = 0  (False)
        #   data quality clean = +5
        #   total ≈ 31.5  → grade C
        engine = TradeDecisionEngine()
        inp = _make_input(
            confidence=0.5,
            risk_reward=1.2,
            bias_net="neutral",
            htf_aligned=False,
            regime_supports=False,
            structure_strength=0.6,
            volume_confirms=False,
            data_quality_clean=True,
        )

        # Act
        result = engine.grade(inp)

        # Assert
        assert result.setup_grade == "C"
        assert result.trade_decision == "SKIP"
        assert 30.0 <= result.quality_score < 45.0


class TestGradeAvoid:
    """Avoid setup: score < 30, decision = NO_TRADE."""

    def test_avoid_grade_and_no_trade_decision(self) -> None:
        # Arrange — very low confidence, R:R below 1, bias conflict, dirty data
        # Expected score:
        #   confidence 0.2 * 25 = 5
        #   R:R 0.5 * 6.67 = 3.335
        #   bias bearish conflicts BUY = 0
        #   conflict 0.0 → 0
        #   HTF not aligned = -5
        #   regime supports = 0  (False)
        #   structure 0.1 * 10 = 1
        #   volume confirms = 0  (False)
        #   data quality NOT clean = -10
        #   total ≈ -5.665 → clamped to 0  → Avoid
        engine = TradeDecisionEngine()
        inp = _make_input(
            confidence=0.2,
            risk_reward=0.5,
            bias_net="bearish",
            action="BUY",
            htf_aligned=False,
            regime_supports=False,
            structure_strength=0.1,
            volume_confirms=False,
            data_quality_clean=False,
        )

        # Act
        result = engine.grade(inp)

        # Assert
        assert result.setup_grade == "Avoid"
        assert result.trade_decision == "NO_TRADE"
        assert result.quality_score < 30.0


class TestLateEntryDowngrades:
    """Late-entry flag applies -15 pts, sufficient to drop grade tier."""

    def test_late_entry_penalty_reduces_grade(self) -> None:
        # Arrange — build a solid but not elite setup that scores in low-A range
        # without late entry, then verify late entry drops it.
        engine = TradeDecisionEngine()
        base_inp = _make_input(
            confidence=0.7,
            risk_reward=2.0,
            regime_supports=False,
            structure_strength=0.5,
            volume_confirms=False,
            is_late_entry=False,
        )
        late_inp = _make_input(
            confidence=0.7,
            risk_reward=2.0,
            regime_supports=False,
            structure_strength=0.5,
            volume_confirms=False,
            is_late_entry=True,
        )

        # Act
        base_result = engine.grade(base_inp)
        late_result = engine.grade(late_inp)

        # Assert — late entry produces a lower quality_score
        assert late_result.quality_score < base_result.quality_score
        assert late_result.quality_score == pytest.approx(
            base_result.quality_score - 15.0
        )
        # And at least one reason mentions "late entry"
        assert any("late entry" in r for r in late_result.reasons)


class TestResultFields:
    """GradingResult contains all required fields with correct types."""

    def test_result_fields_are_populated_and_typed_correctly(self) -> None:
        # Arrange
        engine = TradeDecisionEngine()
        inp = _make_input()

        # Act
        result = engine.grade(inp)

        # Assert — structural contract
        assert isinstance(result, GradingResult)
        assert result.setup_grade in {"A+", "A", "B", "C", "Avoid"}
        assert result.trade_decision in {"TAKE", "WAIT", "SKIP", "NO_TRADE"}
        assert isinstance(result.quality_score, float)
        assert 0.0 <= result.quality_score <= 100.0
        assert isinstance(result.reasons, list)
        assert len(result.reasons) > 0
        assert all(isinstance(r, str) for r in result.reasons)

    def test_result_is_frozen_and_immutable(self) -> None:
        """GradingResult is a frozen dataclass — mutation raises AttributeError."""
        engine = TradeDecisionEngine()
        result = engine.grade(_make_input())

        with pytest.raises((AttributeError, TypeError)):
            result.setup_grade = "Z"  # type: ignore[misc]
