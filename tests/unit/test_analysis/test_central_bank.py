"""
Unit tests for CentralBankEngine.

All tests are synchronous and purely deterministic — no external calls.
"""
from __future__ import annotations

import pytest

from libs.analysis.macro.central_bank import CentralBankEngine, MonetaryEnvironment


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> CentralBankEngine:
    return CentralBankEngine()


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_hiking_is_tightening(engine: CentralBankEngine) -> None:
    """Fed hiking → tightening regime, risk_off, negative confidence adjustment."""
    result = engine.assess(fed_rate=5.25, rate_direction="hiking")

    assert result.regime == "tightening"
    assert result.risk_appetite == "risk_off"
    assert result.confidence_adjustment < 0


def test_cutting_is_easing(engine: CentralBankEngine) -> None:
    """Fed cutting → easing regime, risk_on, positive confidence adjustment."""
    result = engine.assess(fed_rate=4.0, rate_direction="cutting")

    assert result.regime == "easing"
    assert result.risk_appetite == "risk_on"
    assert result.confidence_adjustment > 0


def test_hold_high_rate(engine: CentralBankEngine) -> None:
    """Hold at 5.5% (above 5% threshold) → tightening regime."""
    result = engine.assess(fed_rate=5.5, rate_direction="hold")

    assert result.regime == "tightening"
    assert result.confidence_adjustment < 0


def test_strong_dollar_hurts_crypto(engine: CentralBankEngine) -> None:
    """DXY rising → additional negative confidence adjustment vs neutral DXY."""
    neutral_result = engine.assess(fed_rate=3.0, rate_direction="hold", dxy_trend="neutral")
    rising_dxy_result = engine.assess(fed_rate=3.0, rate_direction="hold", dxy_trend="rising")

    assert rising_dxy_result.confidence_adjustment < neutral_result.confidence_adjustment


def test_result_fields(engine: CentralBankEngine) -> None:
    """All MonetaryEnvironment fields are present and within valid ranges."""
    result = engine.assess(fed_rate=4.5, rate_direction="hold", treasury_10y=4.0, dxy_trend="neutral")

    assert isinstance(result, MonetaryEnvironment)
    assert result.regime in {"tightening", "neutral", "easing"}
    assert result.risk_appetite in {"risk_on", "risk_off", "neutral"}
    assert -0.10 <= result.confidence_adjustment <= 0.05
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0
