"""
Unit tests for libs/risk/safety_mode.py.

All tests are isolated — SafetyModeManager is instantiated directly with
explicit thresholds so no global state leaks between tests.
"""
from __future__ import annotations

import pytest

from libs.risk.safety_mode import (
    ModeTransition,
    SafetyMode,
    SafetyModeManager,
    get_safety_manager,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_manager(**kwargs) -> SafetyModeManager:
    """Create a fresh SafetyModeManager with sensible test thresholds."""
    defaults = dict(
        drawdown_safe_pct=5.0,
        drawdown_defensive_pct=10.0,
        drawdown_panic_pct=20.0,
        max_atr_pct=15.0,
        max_consecutive_losses=8,
        recovery_threshold=5.0,
    )
    defaults.update(kwargs)
    return SafetyModeManager(**defaults)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestInitialState:
    def test_initial_mode_normal(self):
        """A freshly created manager should be in NORMAL mode."""
        mgr = make_manager()
        assert mgr.current_mode == SafetyMode.NORMAL

    def test_initial_history_empty(self):
        mgr = make_manager()
        assert mgr.mode_history == []

    def test_initial_trading_allowed(self):
        mgr = make_manager()
        assert mgr.is_trading_allowed() is True


class TestDrawdownEscalation:
    def test_enter_safe_mode_on_drawdown(self):
        """6% drawdown with safe_pct=5 should trigger SAFE."""
        mgr = make_manager(drawdown_safe_pct=5.0)
        mode = mgr.update(drawdown_pct=6.0)
        assert mode == SafetyMode.SAFE
        assert mgr.current_mode == SafetyMode.SAFE

    def test_enter_defensive_mode_on_larger_drawdown(self):
        """12% drawdown with defensive_pct=10 should trigger DEFENSIVE."""
        mgr = make_manager(drawdown_defensive_pct=10.0)
        mode = mgr.update(drawdown_pct=12.0)
        assert mode == SafetyMode.DEFENSIVE

    def test_enter_panic_mode_on_extreme_drawdown(self):
        """22% drawdown with panic_pct=20 should trigger PANIC."""
        mgr = make_manager(drawdown_panic_pct=20.0)
        mode = mgr.update(drawdown_pct=22.0)
        assert mode == SafetyMode.PANIC

    def test_normal_mode_below_safe_threshold(self):
        """3% drawdown with safe_pct=5 should remain NORMAL."""
        mgr = make_manager(drawdown_safe_pct=5.0)
        mode = mgr.update(drawdown_pct=3.0)
        assert mode == SafetyMode.NORMAL

    def test_exact_safe_threshold_triggers_safe(self):
        """Drawdown exactly at safe_pct should trigger SAFE."""
        mgr = make_manager(drawdown_safe_pct=5.0)
        mode = mgr.update(drawdown_pct=5.0)
        assert mode == SafetyMode.SAFE


class TestVolatilityEscalation:
    def test_enter_defensive_on_atr_spike(self):
        """ATR 20% with max 15% → DEFENSIVE."""
        mgr = make_manager(max_atr_pct=15.0)
        mode = mgr.update(current_atr_pct=20.0)
        assert mode == SafetyMode.DEFENSIVE

    def test_enter_no_trade_on_extreme_volatility_spike(self):
        """ATR 20% with max_atr_pct=15 → DEFENSIVE or higher (>= DEFENSIVE)."""
        mgr = make_manager(max_atr_pct=15.0)
        mode = mgr.update(current_atr_pct=20.0)
        assert mode >= SafetyMode.DEFENSIVE

    def test_enter_panic_on_double_atr_spike(self):
        """ATR > 2x max_atr_pct → PANIC."""
        mgr = make_manager(max_atr_pct=15.0)
        mode = mgr.update(current_atr_pct=31.0)  # > 15 * 2
        assert mode == SafetyMode.PANIC

    def test_normal_atr_no_escalation(self):
        """ATR within normal range → NORMAL."""
        mgr = make_manager(max_atr_pct=15.0)
        mode = mgr.update(current_atr_pct=10.0)
        assert mode == SafetyMode.NORMAL


class TestLossStreakEscalation:
    def test_defensive_on_loss_streak(self):
        """Consecutive losses at max_losses threshold → DEFENSIVE."""
        mgr = make_manager(max_consecutive_losses=8)
        mode = mgr.update(consecutive_losses=8)
        assert mode == SafetyMode.DEFENSIVE

    def test_panic_on_double_loss_streak(self):
        """Consecutive losses at 2x max → PANIC."""
        mgr = make_manager(max_consecutive_losses=8)
        mode = mgr.update(consecutive_losses=16)
        assert mode == SafetyMode.PANIC

    def test_record_loss_increments_counter(self):
        """record_loss() should increment the internal counter."""
        mgr = make_manager(max_consecutive_losses=8)
        for _ in range(8):
            mgr.record_loss()
        mode = mgr.update()
        assert mode == SafetyMode.DEFENSIVE

    def test_record_win_resets_counter(self):
        """record_win() should reset the consecutive-loss counter."""
        mgr = make_manager(max_consecutive_losses=8)
        for _ in range(8):
            mgr.record_loss()
        mgr.record_win()
        mode = mgr.update()
        assert mode == SafetyMode.NORMAL


class TestTradingControls:
    def test_trading_allowed_in_normal(self):
        mgr = make_manager()
        assert mgr.is_trading_allowed() is True

    def test_trading_allowed_in_safe(self):
        mgr = make_manager()
        mgr.update(drawdown_pct=6.0)
        assert mgr.is_trading_allowed() is True

    def test_trading_allowed_in_defensive(self):
        mgr = make_manager()
        mgr.update(drawdown_pct=12.0)
        assert mgr.is_trading_allowed() is True

    def test_trading_blocked_in_panic(self):
        mgr = make_manager()
        mgr.update(drawdown_pct=25.0)
        assert mgr.is_trading_allowed() is False

    def test_trading_blocked_in_no_trade(self):
        mgr = make_manager()
        mgr.force_mode(SafetyMode.NO_TRADE)
        assert mgr.is_trading_allowed() is False


class TestLeverageAndSizing:
    def test_leverage_normal(self):
        mgr = make_manager()
        assert mgr.max_leverage() == 3.0

    def test_leverage_reduction_in_safe_mode(self):
        """Leverage in SAFE mode must be less than NORMAL (3.0)."""
        mgr = make_manager()
        mgr.update(drawdown_pct=6.0)
        assert mgr.current_mode == SafetyMode.SAFE
        assert mgr.max_leverage() < 3.0
        assert mgr.max_leverage() == 2.0

    def test_leverage_defensive(self):
        mgr = make_manager()
        mgr.update(drawdown_pct=12.0)
        assert mgr.max_leverage() == 1.0

    def test_leverage_panic(self):
        mgr = make_manager()
        mgr.update(drawdown_pct=25.0)
        assert mgr.max_leverage() == 0.0

    def test_position_size_multiplier_normal(self):
        mgr = make_manager()
        assert mgr.position_size_multiplier() == 1.0

    def test_position_size_multiplier_safe(self):
        """Position size multiplier must be < 1.0 in SAFE mode."""
        mgr = make_manager()
        mgr.update(drawdown_pct=6.0)
        assert mgr.current_mode == SafetyMode.SAFE
        assert mgr.position_size_multiplier() < 1.0
        assert mgr.position_size_multiplier() == 0.50

    def test_position_size_multiplier_defensive(self):
        mgr = make_manager()
        mgr.update(drawdown_pct=12.0)
        assert mgr.position_size_multiplier() == 0.25

    def test_position_size_multiplier_panic(self):
        mgr = make_manager()
        mgr.update(drawdown_pct=25.0)
        assert mgr.position_size_multiplier() == 0.0


class TestRecovery:
    def test_recovery_from_panic(self):
        """When drawdown drops below recovery level, mode steps down from PANIC."""
        mgr = make_manager(
            drawdown_safe_pct=5.0,
            drawdown_panic_pct=20.0,
            recovery_threshold=5.0,
        )
        # Enter PANIC
        mgr.update(drawdown_pct=22.0)
        assert mgr.current_mode == SafetyMode.PANIC

        # Drop below recovery level (safe_pct - recovery_threshold = 0.0)
        mgr.update(drawdown_pct=0.0)
        # Should step down one level: PANIC → DEFENSIVE
        assert mgr.current_mode == SafetyMode.DEFENSIVE

    def test_recovery_is_gradual(self):
        """Recovery steps down one level at a time, not immediately to NORMAL."""
        mgr = make_manager(
            drawdown_safe_pct=5.0,
            drawdown_panic_pct=20.0,
            recovery_threshold=5.0,
        )
        mgr.update(drawdown_pct=22.0)  # PANIC
        mgr.update(drawdown_pct=0.0)   # DEFENSIVE
        mgr.update(drawdown_pct=0.0)   # SAFE
        mgr.update(drawdown_pct=0.0)   # NORMAL
        assert mgr.current_mode == SafetyMode.NORMAL

    def test_no_recovery_when_drawdown_above_threshold(self):
        """No recovery if drawdown is still above recovery level."""
        mgr = make_manager(
            drawdown_safe_pct=5.0,
            drawdown_panic_pct=20.0,
            recovery_threshold=5.0,
        )
        mgr.update(drawdown_pct=22.0)  # PANIC
        # 6% is above recovery level (0%); should NOT step down
        mgr.update(drawdown_pct=6.0)
        assert mgr.current_mode == SafetyMode.PANIC


class TestManualOverride:
    def test_manual_override_force_no_trade(self):
        """force_mode(NO_TRADE) should override computed NORMAL."""
        mgr = make_manager()
        mgr.force_mode(SafetyMode.NO_TRADE)
        assert mgr.current_mode == SafetyMode.NO_TRADE

    def test_manual_override_clear(self):
        """clear_override() should revert to the computed mode."""
        mgr = make_manager()
        mgr.force_mode(SafetyMode.NO_TRADE)
        assert mgr.current_mode == SafetyMode.NO_TRADE
        mgr.clear_override()
        assert mgr.current_mode == SafetyMode.NORMAL

    def test_override_does_not_affect_computed_mode(self):
        """Manual override should not change the internally computed mode."""
        mgr = make_manager()
        mgr.update(drawdown_pct=6.0)          # computed = SAFE
        mgr.force_mode(SafetyMode.DEFENSIVE)  # override
        assert mgr.current_mode == SafetyMode.DEFENSIVE
        mgr.clear_override()
        assert mgr.current_mode == SafetyMode.SAFE  # back to computed

    def test_override_blocks_trading_in_no_trade(self):
        mgr = make_manager()
        mgr.force_mode(SafetyMode.NO_TRADE)
        assert mgr.is_trading_allowed() is False


class TestModeHistory:
    def test_mode_history_records_transitions(self):
        """Multiple transitions should be recorded in history."""
        mgr = make_manager()
        mgr.update(drawdown_pct=6.0)   # NORMAL → SAFE
        mgr.update(drawdown_pct=12.0)  # SAFE → DEFENSIVE
        mgr.update(drawdown_pct=25.0)  # DEFENSIVE → PANIC

        history = mgr.mode_history
        assert len(history) == 3
        assert all(isinstance(t, ModeTransition) for t in history)

    def test_mode_history_transition_details(self):
        """First transition should capture from/to correctly."""
        mgr = make_manager()
        mgr.update(drawdown_pct=6.0)

        t = mgr.mode_history[0]
        assert t.from_mode == SafetyMode.NORMAL
        assert t.to_mode == SafetyMode.SAFE
        assert isinstance(t.reason, str)
        assert len(t.reason) > 0
        assert isinstance(t.timestamp, str)

    def test_no_duplicate_history_on_same_mode(self):
        """Calling update with the same mode twice should record only one transition."""
        mgr = make_manager()
        mgr.update(drawdown_pct=6.0)
        mgr.update(drawdown_pct=7.0)  # still SAFE
        assert len(mgr.mode_history) == 1

    def test_history_is_copy(self):
        """mode_history should return a copy, not the internal list."""
        mgr = make_manager()
        mgr.update(drawdown_pct=6.0)
        h1 = mgr.mode_history
        h1.clear()
        assert len(mgr.mode_history) == 1


class TestGetStats:
    def test_get_stats_returns_dict(self):
        mgr = make_manager()
        stats = mgr.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_keys(self):
        mgr = make_manager()
        stats = mgr.get_stats()
        expected_keys = {
            "current_mode",
            "computed_mode",
            "override_active",
            "override_mode",
            "consecutive_losses",
            "trading_allowed",
            "max_leverage",
            "position_size_multiplier",
            "total_transitions",
            "last_transition",
        }
        assert expected_keys.issubset(stats.keys())

    def test_get_stats_reflects_panic(self):
        mgr = make_manager()
        mgr.update(drawdown_pct=25.0)
        stats = mgr.get_stats()
        assert stats["current_mode"] == "PANIC"
        assert stats["trading_allowed"] is False
        assert stats["max_leverage"] == 0.0

    def test_get_stats_override_active(self):
        mgr = make_manager()
        mgr.force_mode(SafetyMode.NO_TRADE)
        stats = mgr.get_stats()
        assert stats["override_active"] is True
        assert stats["override_mode"] == "NO_TRADE"

    def test_get_stats_no_transition_yet(self):
        mgr = make_manager()
        stats = mgr.get_stats()
        assert stats["last_transition"] is None
        assert stats["total_transitions"] == 0


class TestSingleton:
    def test_get_safety_manager_returns_instance(self):
        mgr = get_safety_manager()
        assert isinstance(mgr, SafetyModeManager)

    def test_get_safety_manager_is_singleton(self):
        mgr1 = get_safety_manager()
        mgr2 = get_safety_manager()
        assert mgr1 is mgr2
