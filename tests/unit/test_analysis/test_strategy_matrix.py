"""
Unit tests for libs/analysis/regime/strategy_matrix.py.

5 tests covering trending allowances, choppy blocking, climactic blocking,
breakout allowances, and unknown regime behaviour.  All tests are isolated —
no I/O, no environment-variable side-effects.
"""
from __future__ import annotations

import pytest

from libs.analysis.regime.strategy_matrix import RegimeStrategyMatrix, StrategyActivation


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def matrix() -> RegimeStrategyMatrix:
    return RegimeStrategyMatrix()


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestRegimeStrategyMatrix:

    def test_trending_allows_ema(self, matrix: RegimeStrategyMatrix) -> None:
        """ema_crossover should be allowed in trending_up regime."""
        result = matrix.check("ema_crossover", "trending_up")

        assert isinstance(result, StrategyActivation)
        assert result.is_allowed is True
        assert result.strategy_name == "ema_crossover"
        assert result.reason != ""

    def test_choppy_blocks_most(self, matrix: RegimeStrategyMatrix) -> None:
        """ranging_high_vol (choppy) should block non-range strategies."""
        # ema_crossover is NOT in ranging_high_vol — only range_fade and
        # rsi_mean_reversion are allowed
        result = matrix.check("ema_crossover", "ranging_high_vol")

        assert result.is_allowed is False
        assert result.confidence_modifier == 0.0
        assert "ema_crossover" in result.reason

        # Also verify a trend-following strategy is blocked
        result2 = matrix.check("trend_following", "ranging_high_vol")
        assert result2.is_allowed is False

    def test_climactic_blocks_all(self, matrix: RegimeStrategyMatrix) -> None:
        """No strategy should be allowed in climactic regime."""
        strategies = [
            "ema_crossover",
            "trend_following",
            "range_fade",
            "volume_breakout",
            "hammer_reversal",
            "some_random_strategy",
        ]
        for strategy in strategies:
            result = matrix.check(strategy, "climactic")
            assert result.is_allowed is False, (
                f"Expected {strategy} to be blocked in climactic regime"
            )
            assert result.confidence_modifier == 0.0
            assert "climactic" in result.reason.lower()

    def test_breakout_allows_volume(self, matrix: RegimeStrategyMatrix) -> None:
        """volume_breakout should be allowed in breakout regime."""
        result = matrix.check("volume_breakout", "breakout")

        assert result.is_allowed is True
        assert result.strategy_name == "volume_breakout"
        # Also check atr_breakout
        result2 = matrix.check("atr_breakout", "breakout")
        assert result2.is_allowed is True

    def test_unknown_allows_reversals(self, matrix: RegimeStrategyMatrix) -> None:
        """Only reversal strategies should be allowed in unknown regime."""
        # Reversal strategies are allowed
        hammer = matrix.check("hammer_reversal", "unknown")
        assert hammer.is_allowed is True

        shooting = matrix.check("shooting_star_reversal", "unknown")
        assert shooting.is_allowed is True

        # Non-reversal strategies are blocked
        trend = matrix.check("trend_following", "unknown")
        assert trend.is_allowed is False

        ema = matrix.check("ema_crossover", "unknown")
        assert ema.is_allowed is False


class TestStrategyActivationResult:
    """StrategyActivation result contract checks."""

    def test_result_is_frozen(self, matrix: RegimeStrategyMatrix) -> None:
        """StrategyActivation should be immutable (frozen dataclass)."""
        result = matrix.check("ema_crossover", "trending_up")
        with pytest.raises((AttributeError, TypeError)):
            result.is_allowed = False  # type: ignore[misc]

    def test_trend_following_gets_confidence_boost_in_trending(
        self, matrix: RegimeStrategyMatrix
    ) -> None:
        """trend_following in trending_up should receive a positive confidence boost."""
        result = matrix.check("trend_following", "trending_up")
        assert result.is_allowed is True
        assert result.confidence_modifier > 0.0

    def test_momentum_continuation_gets_boost_in_trending_down(
        self, matrix: RegimeStrategyMatrix
    ) -> None:
        """momentum_continuation in trending_down should receive a confidence boost."""
        result = matrix.check("momentum_continuation", "trending_down")
        assert result.is_allowed is True
        assert result.confidence_modifier > 0.0

    def test_no_boost_for_non_trending_strategies(
        self, matrix: RegimeStrategyMatrix
    ) -> None:
        """Strategies not in the boost set get zero modifier even when allowed."""
        result = matrix.check("ema_crossover", "trending_up")
        assert result.is_allowed is True
        assert result.confidence_modifier == 0.0

    def test_get_allowed_returns_set(self, matrix: RegimeStrategyMatrix) -> None:
        """get_allowed() should return the correct set for a known regime."""
        allowed = matrix.get_allowed("breakout")
        assert "volume_breakout" in allowed
        assert "atr_breakout" in allowed
        assert "ema_crossover" not in allowed

    def test_get_allowed_unknown_regime_returns_empty(
        self, matrix: RegimeStrategyMatrix
    ) -> None:
        """get_allowed() should return an empty set for unrecognised regimes."""
        result = matrix.get_allowed("nonexistent_regime")
        assert result == set()
