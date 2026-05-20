"""Tests for bias flip auto-exit logic — document (4) section 9."""
import pytest
from libs.paper_trading.bias_exit import (
    compute_bias_status,
    should_exit_on_bias_flip,
    BIAS_ADVERSE_THRESHOLD,
)


class TestComputeBiasStatus:
    def test_long_bullish_is_aligned(self):
        assert compute_bias_status("LONG", "bullish") == "ALIGNED"

    def test_long_bearish_is_conflict(self):
        assert compute_bias_status("LONG", "bearish") == "CONFLICT"

    def test_long_neutral_is_warning(self):
        assert compute_bias_status("LONG", "neutral") == "WARNING"

    def test_short_bearish_is_aligned(self):
        assert compute_bias_status("SHORT", "bearish") == "ALIGNED"

    def test_short_bullish_is_conflict(self):
        assert compute_bias_status("SHORT", "bullish") == "CONFLICT"

    def test_short_neutral_is_warning(self):
        assert compute_bias_status("SHORT", "neutral") == "WARNING"

    def test_flat_anything_is_no_position(self):
        assert compute_bias_status("FLAT", "bullish") == "NO_POSITION"


class TestShouldExitOnBiasFlip:
    def test_spot_long_bearish_first_check_no_exit(self):
        """First adverse check: increment count, don't exit."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="LONG", market_mode="SPOT",
            current_bias="bearish", bias_adverse_count=0,
        )
        assert exit_now is False
        assert new_count == 1

    def test_spot_long_bearish_second_check_exits(self):
        """Second consecutive adverse check: exit."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="LONG", market_mode="SPOT",
            current_bias="bearish", bias_adverse_count=1,
        )
        assert exit_now is True
        assert new_count == 2

    def test_spot_long_bullish_resets_count(self):
        """Bias re-aligns: reset counter."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="LONG", market_mode="SPOT",
            current_bias="bullish", bias_adverse_count=1,
        )
        assert exit_now is False
        assert new_count == 0

    def test_spot_long_neutral_no_increment(self):
        """SPOT neutral = manage only, don't increment."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="LONG", market_mode="SPOT",
            current_bias="neutral", bias_adverse_count=0,
        )
        assert exit_now is False
        assert new_count == 0

    def test_futures_long_bearish_second_check_exits(self):
        """FUTURES LONG: bearish for 2 checks → CLOSE_LONG."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="LONG", market_mode="FUTURES",
            current_bias="bearish", bias_adverse_count=1,
        )
        assert exit_now is True
        assert new_count == 2

    def test_futures_short_bullish_second_check_exits(self):
        """FUTURES SHORT: bullish for 2 checks → CLOSE_SHORT."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="SHORT", market_mode="FUTURES",
            current_bias="bullish", bias_adverse_count=1,
        )
        assert exit_now is True
        assert new_count == 2

    def test_futures_short_bearish_resets(self):
        """FUTURES SHORT: bias aligns (bearish) → reset."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="SHORT", market_mode="FUTURES",
            current_bias="bearish", bias_adverse_count=1,
        )
        assert exit_now is False
        assert new_count == 0

    def test_futures_short_neutral_no_exit(self):
        """FUTURES neutral = manage only, no forced exit."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="SHORT", market_mode="FUTURES",
            current_bias="neutral", bias_adverse_count=0,
        )
        assert exit_now is False
        assert new_count == 0

    def test_flat_direction_never_exits(self):
        exit_now, new_count = should_exit_on_bias_flip(
            direction="FLAT", market_mode="SPOT",
            current_bias="bearish", bias_adverse_count=5,
        )
        assert exit_now is False
        assert new_count == 0

    def test_threshold_is_2(self):
        assert BIAS_ADVERSE_THRESHOLD == 2
