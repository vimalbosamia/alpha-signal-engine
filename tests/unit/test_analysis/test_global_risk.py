"""
Unit tests for GlobalRiskEngine.

All tests are synchronous and exercise assess() with various combinations
of VIX, drawdown, and flag inputs. No external I/O.
"""
from __future__ import annotations

import pytest

from libs.analysis.macro.global_risk import GlobalRiskAssessment, GlobalRiskEngine


# ── Fixture ────────────────────────────────────────────────────────────────────


@pytest.fixture
def engine() -> GlobalRiskEngine:
    return GlobalRiskEngine()


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_calm_market(engine: GlobalRiskEngine) -> None:
    """VIX=15, no flags → risk_level=low, zero confidence adjustment."""
    result = engine.assess(vix=15.0, btc_drawdown_pct=0.0, spy_drawdown_pct=0.0)

    assert result.risk_level == "low"
    assert result.confidence_adjustment == 0.0
    assert result.should_reduce_exposure is False
    assert result.risk_score == 0.0


def test_high_vix(engine: GlobalRiskEngine) -> None:
    """VIX=45 → score=35, risk_level=elevated, negative confidence adjustment."""
    result = engine.assess(vix=45.0)

    # VIX > 40 adds +35 → elevated (25 <= score < 50)
    assert result.risk_level == "elevated"
    assert result.risk_score == 35.0
    assert result.confidence_adjustment < 0.0
    assert result.should_reduce_exposure is False


def test_btc_crash(engine: GlobalRiskEngine) -> None:
    """BTC drawdown=15% → score=25, risk_level=elevated (or higher)."""
    result = engine.assess(btc_drawdown_pct=15.0)

    # BTC drawdown > 10% adds +25 → elevated
    assert result.risk_score == 25.0
    assert result.risk_level == "elevated"
    assert result.confidence_adjustment < 0.0
    assert any("BTC" in f for f in result.factors)


def test_crisis_extreme(engine: GlobalRiskEngine) -> None:
    """Banking crisis + geopolitical → score=50, extreme risk, reduce_exposure=True."""
    result = engine.assess(is_banking_crisis=True, is_geopolitical_risk=True)

    # +30 banking + +20 geopolitical = 50 → high level (50 <= score < 75)
    assert result.risk_score == 50.0
    assert result.risk_level == "high"
    assert result.should_reduce_exposure is True
    assert result.confidence_adjustment == -0.10

    # With VIX spike on top → extreme
    result_extreme = engine.assess(
        vix=55.0,
        is_banking_crisis=True,
        is_geopolitical_risk=True,
    )
    assert result_extreme.risk_level == "extreme"
    assert result_extreme.confidence_adjustment == -0.15
    assert result_extreme.should_reduce_exposure is True


def test_result_fields(engine: GlobalRiskEngine) -> None:
    """All GlobalRiskAssessment fields are present and within valid ranges."""
    result = engine.assess(vix=35.0, btc_drawdown_pct=8.0, spy_drawdown_pct=4.0)

    assert isinstance(result, GlobalRiskAssessment)
    assert isinstance(result.risk_score, float)
    assert 0.0 <= result.risk_score <= 100.0
    assert result.risk_level in {"low", "elevated", "high", "extreme"}
    assert -0.15 <= result.confidence_adjustment <= 0.0
    assert isinstance(result.should_reduce_exposure, bool)
    assert isinstance(result.factors, list)
    assert len(result.factors) > 0
