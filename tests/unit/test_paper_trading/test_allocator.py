"""
Unit tests for libs/paper_trading/allocator.py — CapitalAllocator.

Tests follow TDD RED → GREEN pattern.
All tests are fully isolated (no external dependencies).
"""
from __future__ import annotations

import pytest

from libs.paper_trading.allocator import CapitalAllocator

# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def allocator() -> CapitalAllocator:
    return CapitalAllocator()


# ── kelly_fraction ─────────────────────────────────────────────────────────────

class TestKellyFraction:
    def test_kelly_fraction_positive_edge(self, allocator: CapitalAllocator) -> None:
        """60% WR, 2:1 ratio → raw kelly = 0.6 - 0.4/2 = 0.4, capped at MAX_KELLY_FRACTION=0.25."""
        result = allocator.kelly_fraction(win_rate=0.6, avg_win_loss_ratio=2.0)
        assert result == pytest.approx(0.25, rel=1e-6)

    def test_kelly_fraction_no_edge(self, allocator: CapitalAllocator) -> None:
        """40% WR, 1:1 ratio → kelly = 0.4 - 0.6/1 = -0.2, negative."""
        result = allocator.kelly_fraction(win_rate=0.4, avg_win_loss_ratio=1.0)
        assert result == pytest.approx(-0.2, rel=1e-6)

    def test_kelly_fraction_capped(self, allocator: CapitalAllocator) -> None:
        """Extreme edge: 90% WR, 5:1 → kelly = 0.9 - 0.1/5 = 0.88, capped at 0.25."""
        result = allocator.kelly_fraction(win_rate=0.9, avg_win_loss_ratio=5.0)
        assert result == pytest.approx(0.25, rel=1e-6)

    def test_kelly_fraction_invalid_ratio_zero(self, allocator: CapitalAllocator) -> None:
        """avg_win_loss_ratio = 0 → return -1.0."""
        result = allocator.kelly_fraction(win_rate=0.6, avg_win_loss_ratio=0.0)
        assert result == -1.0

    def test_kelly_fraction_invalid_ratio_negative(self, allocator: CapitalAllocator) -> None:
        """avg_win_loss_ratio < 0 → return -1.0."""
        result = allocator.kelly_fraction(win_rate=0.6, avg_win_loss_ratio=-1.0)
        assert result == -1.0


# ── position_size ──────────────────────────────────────────────────────────────

class TestPositionSize:
    def test_position_size_cold_start(self, allocator: CapitalAllocator) -> None:
        """Phase 1 (trade_count=5): fixed 5% of 1667 ≈ 83.35, above MIN_TRADE_SIZE."""
        result = allocator.position_size(
            bot_capital=1667.0,
            trade_count=5,
            win_rate=0.6,
            avg_win_loss_ratio=2.0,
        )
        assert result == pytest.approx(83.35, rel=1e-2)

    def test_position_size_minimum(self, allocator: CapitalAllocator) -> None:
        """Phase 1 (trade_count=5): 5% of 150 = 7.5, below $10 minimum → 0."""
        result = allocator.position_size(
            bot_capital=150.0,
            trade_count=5,
            win_rate=0.6,
            avg_win_loss_ratio=2.0,
        )
        assert result == 0.0

    def test_position_size_learning_phase(self, allocator: CapitalAllocator) -> None:
        """Phase 2 (trade_count=20): quarter-Kelly sizing."""
        # kelly = 0.6 - 0.4/2 = 0.4 → capped to 0.25; quarter = 0.25 * 0.25 * 10_000 = 625
        result = allocator.position_size(
            bot_capital=10_000.0,
            trade_count=20,
            win_rate=0.6,
            avg_win_loss_ratio=2.0,
        )
        capped_kelly = 0.25  # MAX_KELLY_FRACTION
        expected = min(capped_kelly * 0.25 * 10_000.0, 10_000.0 * 0.15)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_position_size_full_phase(self, allocator: CapitalAllocator) -> None:
        """Phase 3 (trade_count=50): half-Kelly sizing."""
        # kelly = 0.4 → capped to 0.25; half = 0.25 * 0.5 * 10_000 = 1250
        result = allocator.position_size(
            bot_capital=10_000.0,
            trade_count=50,
            win_rate=0.6,
            avg_win_loss_ratio=2.0,
        )
        capped_kelly = 0.25
        expected = min(capped_kelly * 0.5 * 10_000.0, 10_000.0 * 0.15)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_position_size_negative_kelly_uses_floor(self, allocator: CapitalAllocator) -> None:
        """Negative kelly in phase 2/3 → 3% floor for paper learning."""
        result = allocator.position_size(
            bot_capital=10_000.0,
            trade_count=20,
            win_rate=0.4,
            avg_win_loss_ratio=1.0,
        )
        # 3% of 10,000 = 300, capped at 15% max = 1500
        assert result == 300.0

    def test_position_size_phase2_small_capital_below_minimum(self, allocator: CapitalAllocator) -> None:
        """Phase 2 result below MIN_TRADE_SIZE → 0."""
        # kelly = 0.01 (tiny edge), quarter = 0.01 * 0.25 * 100 = 0.25 → below $10
        result = allocator.position_size(
            bot_capital=100.0,
            trade_count=15,
            win_rate=0.51,
            avg_win_loss_ratio=1.0,  # kelly = 0.51 - 0.49 = 0.02
        )
        # quarter-kelly = 0.02 * 0.25 * 100 = 0.5 < MIN_TRADE_SIZE
        assert result == 0.0


# ── rebalance ─────────────────────────────────────────────────────────────────

class TestRebalance:
    def test_rebalance_allocations(self, allocator: CapitalAllocator) -> None:
        """6 active bots: allocations sum to total_capital, best bot > worst bot."""
        bot_sharpes = {
            "bot_a": 2.5,
            "bot_b": 1.8,
            "bot_c": 1.2,
            "bot_d": 0.9,
            "bot_e": 0.5,
            "bot_f": 0.2,
        }
        total_capital = 100_000.0
        result = allocator.rebalance(
            total_capital=total_capital,
            bot_sharpes=bot_sharpes,
            paused_bots=None,
        )

        # All bots are present
        assert set(result.keys()) == set(bot_sharpes.keys())

        # Sum equals total capital (allow small float rounding)
        assert sum(result.values()) == pytest.approx(total_capital, rel=1e-6)

        # Best Sharpe bot gets more capital than worst Sharpe bot
        assert result["bot_a"] > result["bot_f"]

        # All allocations are non-negative
        for allocation in result.values():
            assert allocation >= 0.0

    def test_rebalance_paused_bot(self, allocator: CapitalAllocator) -> None:
        """Paused bot gets 0.0; remaining capital redistributed among active bots."""
        bot_sharpes = {
            "bot_a": 2.0,
            "bot_b": 1.5,
            "bot_c": 1.0,
        }
        paused_bots = {"bot_c"}
        total_capital = 30_000.0

        result = allocator.rebalance(
            total_capital=total_capital,
            bot_sharpes=bot_sharpes,
            paused_bots=paused_bots,
        )

        # Paused bot gets zero
        assert result["bot_c"] == 0.0

        # Active bots share all the capital
        assert result["bot_a"] + result["bot_b"] == pytest.approx(total_capital, rel=1e-6)

        # Better Sharpe gets more
        assert result["bot_a"] > result["bot_b"]

    def test_rebalance_fewer_bots_than_weights(self, allocator: CapitalAllocator) -> None:
        """Fewer active bots than weight slots — remaining weights pooled to active bots."""
        bot_sharpes = {"bot_x": 3.0, "bot_y": 1.0}
        total_capital = 50_000.0
        result = allocator.rebalance(
            total_capital=total_capital,
            bot_sharpes=bot_sharpes,
            paused_bots=None,
        )
        assert sum(result.values()) == pytest.approx(total_capital, rel=1e-6)
        assert result["bot_x"] > result["bot_y"]

    def test_rebalance_all_paused(self, allocator: CapitalAllocator) -> None:
        """All bots paused → all get 0.0."""
        bot_sharpes = {"bot_a": 1.0, "bot_b": 0.5}
        paused_bots = {"bot_a", "bot_b"}
        result = allocator.rebalance(
            total_capital=10_000.0,
            bot_sharpes=bot_sharpes,
            paused_bots=paused_bots,
        )
        assert result["bot_a"] == 0.0
        assert result["bot_b"] == 0.0
