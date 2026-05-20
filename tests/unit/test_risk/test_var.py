"""
Unit tests for libs/risk/var.py (Value at Risk engine).
"""
from __future__ import annotations

import pytest

from libs.risk.var import VaREngine, VaRResult


# ---------------------------------------------------------------------------
# Test 1 — consistently positive returns produce very low VaR
# ---------------------------------------------------------------------------

def test_positive_returns_low_var():
    """
    A series of only positive daily returns should produce a near-zero
    VaR and be flagged as acceptable.
    """
    engine = VaREngine()
    daily_returns = [0.5, 1.0, 0.3, 0.8, 1.2, 0.6, 0.9, 0.4, 1.1, 0.7]

    result = engine.calculate(daily_returns, portfolio_value=10_000.0)

    assert isinstance(result, VaRResult)
    assert result.var_95 >= 0.0
    assert result.daily_risk_pct < 3.0
    assert result.is_acceptable is True
    assert "ACCEPTABLE" in result.explanation


# ---------------------------------------------------------------------------
# Test 2 — highly volatile returns produce a high VaR
# ---------------------------------------------------------------------------

def test_volatile_returns_high_var():
    """
    Extreme negative returns should push VaR above the 3% threshold and
    flag the result as not acceptable.
    """
    engine = VaREngine()
    # Mostly large losses
    daily_returns = [-5.0, -8.0, -3.5, -6.0, -4.5, -7.0, -9.0, -2.5, -5.5, -4.0]

    result = engine.calculate(daily_returns, portfolio_value=10_000.0)

    assert result.var_95 > 0.0
    assert result.daily_risk_pct >= 3.0
    assert result.is_acceptable is False
    assert "HIGH" in result.explanation


# ---------------------------------------------------------------------------
# Test 3 — CVaR should be >= VaR (it captures the tail beyond VaR)
# ---------------------------------------------------------------------------

def test_cvar_worse_than_var():
    """
    CVaR (expected shortfall) must always be >= VaR_95 because it averages
    losses in the worst tail, which is at least as bad as the VaR cut-off.
    """
    engine = VaREngine()
    daily_returns = [
        1.0, -2.0, 0.5, -5.0, 0.3, -1.0, 2.0, -8.0, -0.5, -3.0,
        1.5, -4.0, 0.2, -6.0, -0.8, 3.0, -2.5, 1.2, -1.8, -7.0,
    ]

    result = engine.calculate(daily_returns, portfolio_value=10_000.0)

    assert result.cvar_95 >= result.var_95 - 0.01  # tiny tolerance for rounding


# ---------------------------------------------------------------------------
# Test 4 — empty returns returns a safe zero result
# ---------------------------------------------------------------------------

def test_empty_returns():
    """Passing an empty list should return zero risk and be acceptable."""
    engine = VaREngine()

    result = engine.calculate([], portfolio_value=10_000.0)

    assert result.var_95 == 0.0
    assert result.var_99 == 0.0
    assert result.cvar_95 == 0.0
    assert result.daily_risk_pct == 0.0
    assert result.is_acceptable is True
    assert "No daily returns" in result.explanation
