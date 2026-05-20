"""
Unit tests for DerivativesPressureEngine.

All tests are synchronous and exercise assess() with various funding rate,
OI, and long/short ratio inputs. No external I/O.
"""
from __future__ import annotations

import pytest

from libs.analysis.macro.derivatives import DerivativesPressure, DerivativesPressureEngine


# ── Fixture ────────────────────────────────────────────────────────────────────


@pytest.fixture
def engine() -> DerivativesPressureEngine:
    return DerivativesPressureEngine()


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_normal_funding(engine: DerivativesPressureEngine) -> None:
    """Funding rate=0.01% (0.0001) → neutral, no confidence penalty."""
    result = engine.assess(funding_rate=0.0001)

    assert result.funding_sentiment == "neutral"
    assert result.confidence_adjustment == 0.0
    assert result.liquidation_risk == "low"


def test_high_funding_bullish_pressure(engine: DerivativesPressureEngine) -> None:
    """Funding rate=0.08% → bullish_pressure, -5% confidence adjustment."""
    result = engine.assess(funding_rate=0.0008)

    assert result.funding_sentiment == "bullish_pressure"
    assert result.funding_rate == 0.0008
    assert result.confidence_adjustment == pytest.approx(-0.05)


def test_negative_funding_bearish_pressure(engine: DerivativesPressureEngine) -> None:
    """Funding rate=-0.06% → bearish_pressure, -5% confidence adjustment."""
    result = engine.assess(funding_rate=-0.0006)

    assert result.funding_sentiment == "bearish_pressure"
    assert result.confidence_adjustment == pytest.approx(-0.05)


def test_oi_increasing_with_price_drop(engine: DerivativesPressureEngine) -> None:
    """
    OI increasing while price drops (negative oi_change_pct convention)
    → liquidation_risk=high, -10% confidence.
    """
    # Negative oi_change_pct signals OI growing while price falls
    result = engine.assess(funding_rate=0.0001, oi_change_pct=-5.0, long_short_ratio=1.0)

    assert result.liquidation_risk == "high"
    assert result.open_interest_trend == "increasing"
    assert result.confidence_adjustment == pytest.approx(-0.10)


def test_overcrowded_longs(engine: DerivativesPressureEngine) -> None:
    """Long/short ratio=2.5 → overcrowded longs reversal risk, -5% confidence."""
    result = engine.assess(funding_rate=0.0001, oi_change_pct=0.0, long_short_ratio=2.5)

    assert result.confidence_adjustment == pytest.approx(-0.05)
    assert "overcrowded longs" in result.explanation or "reversal" in result.explanation


def test_result_fields(engine: DerivativesPressureEngine) -> None:
    """All DerivativesPressure fields are present and within valid ranges."""
    result = engine.assess(funding_rate=0.0003, oi_change_pct=1.0, long_short_ratio=1.2)

    assert isinstance(result, DerivativesPressure)
    assert isinstance(result.funding_rate, float)
    assert result.funding_sentiment in {"bearish_pressure", "neutral", "bullish_pressure"}
    assert result.open_interest_trend in {"increasing", "stable", "decreasing"}
    assert result.liquidation_risk in {"low", "moderate", "high"}
    assert -0.10 <= result.confidence_adjustment <= 0.0
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0
