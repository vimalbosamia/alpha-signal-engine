"""
Unit tests for SymbolEligibilityEngine.

All inputs are synthetic — no external I/O.
"""
from __future__ import annotations

import pytest

from libs.analysis.filters.symbol_eligibility import (
    SymbolEligibility,
    SymbolEligibilityEngine,
)

# Default engine with standard institutional thresholds
ENGINE = SymbolEligibilityEngine(
    min_volume_usd=5_000_000,
    max_spread_pct=0.5,
    min_atr_pct=0.1,
    max_atr_pct=20.0,
)


class TestEligibleSymbol:
    def test_eligible_good_symbol(self):
        """All criteria met → eligible with reason 'ok'."""
        result = ENGINE.check("BTCUSDT", volume_24h=100_000_000, spread_pct=0.05, atr_pct=2.5)

        assert result.is_eligible is True
        assert result.reason == "ok"
        assert result.liquidity_ok is True
        assert result.volatility_ok is True
        assert result.spread_quality == "good"
        assert result.symbol == "BTCUSDT"
        assert result.volume_24h == 100_000_000


class TestRejectionRules:
    def test_low_volume_rejected(self):
        """volume_24h below minimum → ineligible with reason 'low volume'."""
        result = ENGINE.check("SHITUSDT", volume_24h=1_000, spread_pct=0.05, atr_pct=2.5)

        assert result.is_eligible is False
        assert result.reason == "low volume"
        assert result.liquidity_ok is False

    def test_wide_spread_rejected(self):
        """spread above max → ineligible with reason 'wide spread'."""
        result = ENGINE.check("ILLIQUSDT", volume_24h=50_000_000, spread_pct=1.0, atr_pct=2.5)

        assert result.is_eligible is False
        assert result.reason == "wide spread"
        assert result.liquidity_ok is False

    def test_dead_market_rejected(self):
        """ATR below min → ineligible with reason 'dead market'."""
        result = ENGINE.check("FLATUSDT", volume_24h=50_000_000, spread_pct=0.05, atr_pct=0.01)

        assert result.is_eligible is False
        assert result.reason == "dead market"
        assert result.volatility_ok is False

    def test_too_volatile_rejected(self):
        """ATR above max → ineligible with reason 'too volatile'."""
        result = ENGINE.check("MOONUSDT", volume_24h=50_000_000, spread_pct=0.05, atr_pct=50.0)

        assert result.is_eligible is False
        assert result.reason == "too volatile"
        assert result.volatility_ok is False


class TestSpreadQualityLevels:
    def test_spread_quality_levels(self):
        """Spread buckets: <0.1% → good, <0.3% → fair, else → poor."""
        # Arrange
        cases = [
            (0.05, "good"),
            (0.09, "good"),
            (0.1, "fair"),
            (0.29, "fair"),
            (0.3, "poor"),
            (0.49, "poor"),
        ]
        for spread, expected_quality in cases:
            result = ENGINE.check("SYM", volume_24h=50_000_000, spread_pct=spread, atr_pct=2.0)
            assert result.spread_quality == expected_quality, (
                f"spread={spread} → expected {expected_quality!r}, got {result.spread_quality!r}"
            )

    def test_result_is_immutable(self):
        """SymbolEligibility is a frozen dataclass."""
        result = ENGINE.check("BTCUSDT", volume_24h=100_000_000, spread_pct=0.05, atr_pct=2.5)
        with pytest.raises((AttributeError, TypeError)):
            result.is_eligible = False  # type: ignore[misc]
