"""
Unit tests for libs/validation/monte_carlo.py.

Tests are deterministic via fixed random seeds where outcome variance matters.
"""
from __future__ import annotations

import random

import pytest

from libs.validation.monte_carlo import MonteCarloEngine, MonteCarloResult


# ---------------------------------------------------------------------------
# Test 1 — all-winner strategy should be robust
# ---------------------------------------------------------------------------

def test_all_winners_robust():
    """A strategy that always wins +5% should be flagged as robust."""
    random.seed(42)
    engine = MonteCarloEngine()
    trade_returns = [5.0] * 50  # every trade gains 5%

    result = engine.simulate(trade_returns, n_simulations=500, n_trades_per_sim=50)

    assert isinstance(result, MonteCarloResult)
    assert result.simulations == 500
    assert result.probability_of_profit == pytest.approx(1.0, abs=0.0)
    assert result.median_return_pct > 0
    assert result.is_robust is True
    assert "ROBUST" in result.explanation


# ---------------------------------------------------------------------------
# Test 2 — all-loser strategy should NOT be robust
# ---------------------------------------------------------------------------

def test_all_losers_not_robust():
    """A strategy that always loses -3% should never be flagged as robust."""
    random.seed(42)
    engine = MonteCarloEngine()
    trade_returns = [-3.0] * 50  # every trade loses 3%

    result = engine.simulate(trade_returns, n_simulations=500, n_trades_per_sim=50)

    assert result.probability_of_profit == pytest.approx(0.0, abs=0.0)
    assert result.median_return_pct < 0
    assert result.is_robust is False
    assert "NOT ROBUST" in result.explanation


# ---------------------------------------------------------------------------
# Test 3 — mixed returns produce a reasonable probability of profit
# ---------------------------------------------------------------------------

def test_probability_reasonable():
    """
    A strategy with a roughly 60/40 win-rate should produce a probability
    of profit between 0.4 and 1.0 across simulated paths.
    """
    random.seed(7)
    engine = MonteCarloEngine()
    # 60 wins (+2%) and 40 losses (-1%) — net-positive expectancy
    trade_returns = [2.0] * 60 + [-1.0] * 40

    result = engine.simulate(trade_returns, n_simulations=1000, n_trades_per_sim=100)

    assert 0.4 <= result.probability_of_profit <= 1.0
    assert result.p5_return_pct < result.p95_return_pct
    assert result.max_drawdown_median >= 0.0


# ---------------------------------------------------------------------------
# Test 4 — empty trade_returns returns a safe zero result
# ---------------------------------------------------------------------------

def test_empty_returns():
    """Passing an empty list should return a safe, not-robust zero result."""
    engine = MonteCarloEngine()

    result = engine.simulate([], n_simulations=100, n_trades_per_sim=50)

    assert result.simulations == 0
    assert result.median_return_pct == 0.0
    assert result.probability_of_profit == 0.0
    assert result.is_robust is False
    assert "No trade returns" in result.explanation
