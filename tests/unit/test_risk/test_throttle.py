"""
Unit tests for libs/risk/throttle.py.

6 tests covering: initial normal state, cautious after 3 losses, defensive
after 5 losses, halted after 8 losses, win resets consecutive counter,
and result field contract.  All tests are isolated.
"""
from __future__ import annotations

import pytest

from libs.risk.throttle import RiskThrottleEngine, ThrottleState


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> RiskThrottleEngine:
    return RiskThrottleEngine()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _record_losses(engine: RiskThrottleEngine, count: int) -> None:
    for _ in range(count):
        engine.record_loss()


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestRiskThrottleEngine:

    def test_normal_initial(self, engine: RiskThrottleEngine) -> None:
        """Fresh engine should start in normal / unthrottled state."""
        state = engine.get_state()

        assert isinstance(state, ThrottleState)
        assert state.throttle_level == "normal"
        assert state.is_throttled is False
        assert state.confidence_penalty == 0.0
        assert state.max_signals_per_cycle > 0

    def test_cautious_after_3(self, engine: RiskThrottleEngine) -> None:
        """3 consecutive losses should trigger cautious throttle."""
        _record_losses(engine, 3)
        state = engine.get_state()

        assert state.throttle_level == "cautious"
        assert state.is_throttled is True
        assert state.confidence_penalty < 0.0
        assert state.max_signals_per_cycle <= 5

    def test_defensive_after_5(self, engine: RiskThrottleEngine) -> None:
        """5 consecutive losses should escalate to defensive throttle."""
        _record_losses(engine, 5)
        state = engine.get_state()

        assert state.throttle_level == "defensive"
        assert state.is_throttled is True
        assert state.confidence_penalty <= -0.10
        assert state.max_signals_per_cycle <= 2

    def test_halted_after_8(self, engine: RiskThrottleEngine) -> None:
        """8 consecutive losses should halt all new signals."""
        _record_losses(engine, 8)
        state = engine.get_state()

        assert state.throttle_level == "halted"
        assert state.is_throttled is True
        assert state.max_signals_per_cycle == 0

    def test_win_resets_consecutive_counter(
        self, engine: RiskThrottleEngine
    ) -> None:
        """A win after losses should reset the consecutive counter to normal."""
        _record_losses(engine, 4)
        assert engine.get_state().throttle_level == "cautious"

        engine.record_win()
        state = engine.get_state()

        assert state.throttle_level == "normal"
        assert state.is_throttled is False

    def test_result_fields(self, engine: RiskThrottleEngine) -> None:
        """ThrottleState should expose all required fields with correct types."""
        state = engine.get_state()

        assert isinstance(state.is_throttled, bool)
        assert isinstance(state.throttle_level, str)
        assert isinstance(state.max_signals_per_cycle, int)
        assert isinstance(state.confidence_penalty, float)
        assert isinstance(state.reason, str)
        assert state.throttle_level in {"normal", "cautious", "defensive", "halted"}


class TestThrottleStateImmutability:

    def test_state_is_frozen(self, engine: RiskThrottleEngine) -> None:
        """ThrottleState should be immutable (frozen dataclass)."""
        state = engine.get_state()
        with pytest.raises((AttributeError, TypeError)):
            state.throttle_level = "halted"  # type: ignore[misc]


class TestThrottleReset:

    def test_reset_clears_all_counters(self, engine: RiskThrottleEngine) -> None:
        """reset() should clear consecutive and hourly loss counters."""
        _record_losses(engine, 6)
        assert engine.get_state().throttle_level == "defensive"

        engine.reset()
        assert engine.get_state().throttle_level == "normal"

    def test_record_win_only_clears_consecutive(
        self, engine: RiskThrottleEngine
    ) -> None:
        """record_win() only resets consecutive losses, not the hourly counter."""
        # 3 losses → cautious
        _record_losses(engine, 3)
        assert engine.get_state().throttle_level == "cautious"

        # A win brings consecutive back to 0
        engine.record_win()
        assert engine.get_state().throttle_level == "normal"

        # But more losses can escalate again
        _record_losses(engine, 5)
        assert engine.get_state().throttle_level == "defensive"
