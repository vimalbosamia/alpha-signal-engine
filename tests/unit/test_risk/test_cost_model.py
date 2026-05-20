"""
Unit tests for libs/risk/cost_model.py.

5 tests covering crypto spot, stock, futures cost schedules, rejection logic,
and result field completeness.  All tests are fully isolated — no I/O.
"""
from __future__ import annotations

import pytest

from libs.risk.cost_model import CostModelEngine, TradingCosts


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def engine() -> CostModelEngine:
    return CostModelEngine()


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCryptoSpotCosts:
    """Crypto spot round-trip cost should be ~0.36% (spread+slip+commission × 2 sides)."""

    def test_crypto_spot_costs(self, engine: CostModelEngine) -> None:
        # Arrange — standard BTC spot trade
        # Per side: spread 0.05 + slippage 0.03 + commission 0.10 = 0.18%
        # Round trip: 0.18 × 2 = 0.36%

        # Act
        result = engine.assess(
            symbol="BTCUSDT",
            asset_class="crypto",
            entry_price=50_000,
            expected_move_pct=1.0,
        )

        # Assert
        assert result.spread_pct == pytest.approx(0.05)
        assert result.slippage_pct == pytest.approx(0.03)
        assert result.commission_pct == pytest.approx(0.10)
        assert result.total_cost_pct == pytest.approx(0.36)


class TestStockCosts:
    """Stock round-trip cost should be lower (0.04% RT) due to zero commission."""

    def test_stock_costs(self, engine: CostModelEngine) -> None:
        # Arrange — standard equity trade
        # Per side: spread 0.01 + slippage 0.01 + commission 0.00 = 0.02%
        # Round trip: 0.02 × 2 = 0.04%

        # Act
        result = engine.assess(
            symbol="AAPL",
            asset_class="stock",
            entry_price=180,
            expected_move_pct=0.5,
        )

        # Assert
        assert result.spread_pct == pytest.approx(0.01)
        assert result.slippage_pct == pytest.approx(0.01)
        assert result.commission_pct == pytest.approx(0.00)
        assert result.total_cost_pct == pytest.approx(0.04)
        assert result.total_cost_pct < 0.36  # definitely cheaper than crypto spot


class TestFuturesCosts:
    """Crypto futures round-trip cost should be lower than spot (~0.16% RT)."""

    def test_futures_costs(self, engine: CostModelEngine) -> None:
        # Arrange — BTC perpetual futures
        # Per side: spread 0.02 + slippage 0.02 + commission 0.04 = 0.08%
        # Round trip: 0.08 × 2 = 0.16%

        # Act
        result = engine.assess(
            symbol="BTCUSDT-PERP",
            asset_class="crypto",
            entry_price=50_000,
            expected_move_pct=1.0,
            is_futures=True,
        )

        # Assert
        assert result.spread_pct == pytest.approx(0.02)
        assert result.slippage_pct == pytest.approx(0.02)
        assert result.commission_pct == pytest.approx(0.04)
        assert result.total_cost_pct == pytest.approx(0.16)
        assert result.total_cost_pct < 0.36  # cheaper than spot


class TestRejectTinyMove:
    """A 0.1% expected move on crypto spot (costs ~0.36%) should be rejected."""

    def test_reject_tiny_move(self, engine: CostModelEngine) -> None:
        # Arrange — tiny move that cannot cover round-trip costs
        # min_profitable_move = 0.36 × 1.5 = 0.54%; expected 0.10% → reject

        # Act
        result = engine.assess(
            symbol="ETHUSDT",
            asset_class="crypto",
            entry_price=2_000,
            expected_move_pct=0.10,
        )

        # Assert
        assert result.should_reject is True
        assert result.min_profitable_move_pct > result.total_cost_pct
        assert result.min_profitable_move_pct == pytest.approx(0.36 * 1.5)
        assert "REJECTED" in result.explanation


class TestResultFields:
    """TradingCosts dataclass exposes all required fields with correct types."""

    def test_result_fields(self, engine: CostModelEngine) -> None:
        # Arrange
        result = engine.assess(
            symbol="SOLUSDT",
            asset_class="crypto",
            entry_price=100,
            expected_move_pct=2.0,
        )

        # Assert — all fields present and typed correctly
        assert isinstance(result, TradingCosts)
        assert isinstance(result.spread_pct, float)
        assert isinstance(result.slippage_pct, float)
        assert isinstance(result.commission_pct, float)
        assert isinstance(result.total_cost_pct, float)
        assert isinstance(result.min_profitable_move_pct, float)
        assert isinstance(result.should_reject, bool)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0
        # A 2% move on crypto spot should be accepted
        assert result.should_reject is False
        assert "ACCEPTED" in result.explanation
