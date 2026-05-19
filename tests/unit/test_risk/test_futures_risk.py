"""
Unit tests for libs/risk/futures.py — FuturesRiskEngine.

All tests are isolated — FuturesRiskEngine is constructed directly with
no external dependencies or environment variables.

Test IDs:
  1. liq_price_long          — liquidation price formula for LONG
  2. liq_price_short         — liquidation price formula for SHORT
  3. high_leverage_extreme   — leverage >= 50 → EXTREME risk
  4. low_leverage_moderate   — leverage 10 → MODERATE risk
  5. stop_too_close_reject   — stop at/beyond liquidation → rejected
  6. result_fields           — FuturesRiskAssessment has all required fields
  7. max_loss_calculation    — max_loss_before_stop arithmetic
"""
from __future__ import annotations

import pytest

from libs.risk.futures import (
    MAINTENANCE_MARGIN_RATE,
    FuturesRiskAssessment,
    FuturesRiskEngine,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_engine() -> FuturesRiskEngine:
    return FuturesRiskEngine()


def liq_long(entry: float, leverage: float) -> float:
    """Expected liquidation price for a LONG position."""
    return entry * (1 - 1 / leverage + MAINTENANCE_MARGIN_RATE)


def liq_short(entry: float, leverage: float) -> float:
    """Expected liquidation price for a SHORT position."""
    return entry * (1 + 1 / leverage - MAINTENANCE_MARGIN_RATE)


# ── Test 1: Liquidation price — LONG ─────────────────────────────────────────

class TestLiqPriceLong:
    """Liquidation price formula is applied correctly for LONG direction."""

    def test_liq_price_long_10x(self) -> None:
        # entry=100, leverage=10
        # liq = 100 * (1 - 1/10 + 0.004) = 100 * 0.904 = 90.4
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=10.0,
            margin_type="isolated",
            direction="long",
            stop_loss=85.0,
        )
        expected = liq_long(100.0, 10.0)
        assert result.estimated_liquidation_price == pytest.approx(expected, rel=1e-6)

    def test_liq_price_long_20x(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=50_000.0,
            leverage=20.0,
            margin_type="isolated",
            direction="long",
            stop_loss=47_000.0,
        )
        expected = liq_long(50_000.0, 20.0)
        assert result.estimated_liquidation_price == pytest.approx(expected, rel=1e-6)

    def test_liq_is_below_entry_for_long(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=200.0,
            leverage=5.0,
            margin_type="isolated",
            direction="long",
            stop_loss=180.0,
        )
        assert result.estimated_liquidation_price < 200.0


# ── Test 2: Liquidation price — SHORT ────────────────────────────────────────

class TestLiqPriceShort:
    """Liquidation price formula is applied correctly for SHORT direction."""

    def test_liq_price_short_10x(self) -> None:
        # entry=100, leverage=10
        # liq = 100 * (1 + 1/10 - 0.004) = 100 * 1.096 = 109.6
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=10.0,
            margin_type="isolated",
            direction="short",
            stop_loss=105.0,
        )
        expected = liq_short(100.0, 10.0)
        assert result.estimated_liquidation_price == pytest.approx(expected, rel=1e-6)

    def test_liq_price_short_20x(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=50_000.0,
            leverage=20.0,
            margin_type="isolated",
            direction="short",
            stop_loss=51_000.0,
        )
        expected = liq_short(50_000.0, 20.0)
        assert result.estimated_liquidation_price == pytest.approx(expected, rel=1e-6)

    def test_liq_is_above_entry_for_short(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=200.0,
            leverage=5.0,
            margin_type="isolated",
            direction="short",
            stop_loss=210.0,
        )
        assert result.estimated_liquidation_price > 200.0


# ── Test 3: High leverage → EXTREME risk ─────────────────────────────────────

class TestHighLeverageExtreme:
    """Leverage >= 50 or buffer < 2% is classified as EXTREME risk."""

    def test_leverage_50_is_extreme(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=50.0,
            margin_type="isolated",
            direction="long",
            stop_loss=97.0,
        )
        assert result.liquidation_risk == "EXTREME"

    def test_leverage_above_50_is_extreme(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=100.0,
            margin_type="isolated",
            direction="long",
            stop_loss=99.5,
        )
        assert result.liquidation_risk == "EXTREME"

    def test_buffer_below_2pct_is_extreme(self) -> None:
        # leverage=200 → buffer will be very small (< 2%)
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=200.0,
            margin_type="isolated",
            direction="long",
            stop_loss=99.9,
        )
        assert result.liquidation_risk == "EXTREME"


# ── Test 4: Low leverage → MODERATE risk ──────────────────────────────────────

class TestLowLeverageModerate:
    """Leverage = 10 (>=10, <20) with adequate buffer → MODERATE risk."""

    def test_leverage_10_is_moderate(self) -> None:
        # leverage=10: buffer = (1/10 - MMR) * 100 = (0.1 - 0.004) * 100 = 9.6%
        # 9.6% < 10% → MODERATE by buffer condition
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=10.0,
            margin_type="isolated",
            direction="long",
            stop_loss=85.0,
        )
        assert result.liquidation_risk == "MODERATE"

    def test_leverage_below_10_with_good_buffer_is_low(self) -> None:
        # leverage=3: buffer = (1/3 - 0.004) * 100 ≈ 32.9% → LOW
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=3.0,
            margin_type="isolated",
            direction="long",
            stop_loss=60.0,
        )
        assert result.liquidation_risk == "LOW"


# ── Test 5: Stop too close → rejection ───────────────────────────────────────

class TestStopTooCloseReject:
    """Stop at or beyond liquidation price should trigger rejection."""

    def test_stop_beyond_liquidation_long_is_rejected(self) -> None:
        # leverage=10, long: liq ≈ 90.4
        # set stop_loss = 89.0 (below liquidation) → reject
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=10.0,
            margin_type="isolated",
            direction="long",
            stop_loss=89.0,
        )
        assert result.should_reject is True
        assert result.rejection_reason != ""

    def test_stop_at_liquidation_long_is_rejected(self) -> None:
        # stop exactly at liq price — edge case
        engine = make_engine()
        liq = liq_long(100.0, 10.0)
        result = engine.assess(
            entry_price=100.0,
            leverage=10.0,
            margin_type="isolated",
            direction="long",
            stop_loss=liq,
        )
        assert result.should_reject is True

    def test_stop_beyond_liquidation_short_is_rejected(self) -> None:
        # leverage=10, short: liq ≈ 109.6
        # set stop_loss = 112.0 (above liquidation for short) → reject
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=10.0,
            margin_type="isolated",
            direction="short",
            stop_loss=112.0,
        )
        assert result.should_reject is True

    def test_leverage_above_50_is_rejected(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=51.0,
            margin_type="isolated",
            direction="long",
            stop_loss=96.0,
        )
        assert result.should_reject is True

    def test_buffer_below_2pct_is_rejected(self) -> None:
        # leverage=100 → buffer ≈ 0.6% → < 2% → reject
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=100.0,
            margin_type="isolated",
            direction="long",
            stop_loss=99.5,
        )
        assert result.should_reject is True


# ── Test 6: Result fields ─────────────────────────────────────────────────────

class TestResultFields:
    """FuturesRiskAssessment must expose all documented fields with correct types."""

    def test_all_fields_present(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=5.0,
            margin_type="isolated",
            direction="long",
            stop_loss=90.0,
        )
        assert isinstance(result, FuturesRiskAssessment)
        assert isinstance(result.leverage, float)
        assert isinstance(result.margin_type, str)
        assert isinstance(result.estimated_liquidation_price, float)
        assert isinstance(result.liquidation_buffer_percent, float)
        assert isinstance(result.liquidation_risk, str)
        assert isinstance(result.max_loss_before_stop, float)
        assert isinstance(result.should_reject, bool)
        assert isinstance(result.rejection_reason, str)

    def test_liquidation_risk_is_valid_label(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=5.0,
            margin_type="cross",
            direction="long",
            stop_loss=90.0,
        )
        assert result.liquidation_risk in {"LOW", "MODERATE", "HIGH", "EXTREME"}

    def test_margin_type_preserved(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=5.0,
            margin_type="cross",
            direction="long",
            stop_loss=90.0,
        )
        assert result.margin_type == "cross"

    def test_leverage_preserved(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=7.5,
            margin_type="isolated",
            direction="long",
            stop_loss=85.0,
        )
        assert result.leverage == pytest.approx(7.5)

    def test_assessment_is_immutable(self) -> None:
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=5.0,
            margin_type="isolated",
            direction="long",
            stop_loss=90.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.leverage = 99.0  # type: ignore[misc]

    def test_funding_rate_accepted(self) -> None:
        # funding_rate is an optional param; must not raise
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=5.0,
            margin_type="isolated",
            direction="long",
            stop_loss=90.0,
            funding_rate=0.0001,
        )
        assert isinstance(result, FuturesRiskAssessment)


# ── Test 7: Max loss calculation ──────────────────────────────────────────────

class TestMaxLossCalculation:
    """max_loss_before_stop = stop_distance_pct * leverage."""

    def test_max_loss_long_arithmetic(self) -> None:
        # entry=100, stop=95 → stop_dist = 5%
        # leverage=10 → max_loss = 5 * 10 = 50%
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=10.0,
            margin_type="isolated",
            direction="long",
            stop_loss=95.0,
        )
        assert result.max_loss_before_stop == pytest.approx(50.0, rel=1e-4)

    def test_max_loss_short_arithmetic(self) -> None:
        # entry=100, stop=105 → stop_dist = 5%
        # leverage=10 → max_loss = 5 * 10 = 50%
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=10.0,
            margin_type="isolated",
            direction="short",
            stop_loss=105.0,
        )
        assert result.max_loss_before_stop == pytest.approx(50.0, rel=1e-4)

    def test_max_loss_rejects_above_80pct(self) -> None:
        # entry=100, stop=90 → stop_dist=10%, leverage=10 → max_loss=100% → reject
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=10.0,
            margin_type="isolated",
            direction="long",
            stop_loss=90.0,
        )
        assert result.max_loss_before_stop > 80.0
        assert result.should_reject is True

    def test_max_loss_scales_with_leverage(self) -> None:
        engine = make_engine()
        r5 = engine.assess(
            entry_price=100.0,
            leverage=5.0,
            margin_type="isolated",
            direction="long",
            stop_loss=97.0,
        )
        r10 = engine.assess(
            entry_price=100.0,
            leverage=10.0,
            margin_type="isolated",
            direction="long",
            stop_loss=97.0,
        )
        assert r10.max_loss_before_stop == pytest.approx(
            r5.max_loss_before_stop * 2, rel=1e-4
        )

    def test_buffer_percent_formula(self) -> None:
        # For LONG, entry=100, leverage=10:
        # liq = 100 * (1 - 0.1 + 0.004) = 90.4
        # buffer = abs(100 - 90.4) / 100 * 100 = 9.6%
        engine = make_engine()
        result = engine.assess(
            entry_price=100.0,
            leverage=10.0,
            margin_type="isolated",
            direction="long",
            stop_loss=85.0,
        )
        expected_liq = liq_long(100.0, 10.0)
        expected_buffer = abs(100.0 - expected_liq) / 100.0 * 100.0
        assert result.liquidation_buffer_percent == pytest.approx(expected_buffer, rel=1e-6)
