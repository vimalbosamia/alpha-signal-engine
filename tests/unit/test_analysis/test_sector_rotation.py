"""
Unit tests for SectorRotationEngine.

All tests are synchronous and purely deterministic — no external calls.
"""
from __future__ import annotations

import pytest

from libs.analysis.macro.sector_rotation import SectorRotationEngine, SectorStrength


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> SectorRotationEngine:
    return SectorRotationEngine()


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_tech_follows_qqq_down(engine: SectorRotationEngine) -> None:
    """AAPL (tech) + QQQ -2% → weak sector, negative adjustment."""
    result = engine.assess(symbol="AAPL", asset_class="stock", qqq_change_pct=-2.0)

    assert result.sector == "tech"
    assert result.strength == "weak"
    assert result.confidence_adjustment < 0


def test_crypto_follows_btc_down(engine: SectorRotationEngine) -> None:
    """ETHUSDT (crypto) + BTC -3% → weak sector, negative adjustment."""
    result = engine.assess(symbol="ETHUSDT", asset_class="crypto", btc_change_pct=-3.0)

    assert result.sector == "crypto"
    assert result.strength == "weak"
    assert result.confidence_adjustment < 0


def test_rising_market_strong(engine: SectorRotationEngine) -> None:
    """SPY +1.5% for a general stock → strong sector, positive adjustment."""
    result = engine.assess(symbol="JPM", asset_class="stock", spy_change_pct=1.5)

    assert result.strength == "strong"
    assert result.confidence_adjustment > 0


def test_flat_market_neutral(engine: SectorRotationEngine) -> None:
    """SPY 0% for a general stock → neutral sector, zero adjustment."""
    result = engine.assess(symbol="GE", asset_class="stock", spy_change_pct=0.0)

    assert result.strength == "neutral"
    assert result.confidence_adjustment == 0.0


def test_result_fields(engine: SectorRotationEngine) -> None:
    """All SectorStrength fields are present and within valid ranges."""
    result = engine.assess(symbol="MSFT", asset_class="stock", qqq_change_pct=0.5)

    assert isinstance(result, SectorStrength)
    assert isinstance(result.sector, str)
    assert len(result.sector) > 0
    assert result.strength in {"strong", "neutral", "weak"}
    assert -0.05 <= result.confidence_adjustment <= 0.05
