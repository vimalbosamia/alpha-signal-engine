"""
Unit tests for SentimentFilter.

All tests are synchronous and test assess() only.
No external API calls are made.
"""
from __future__ import annotations

import pytest

from libs.analysis.macro.sentiment import SentimentFilter, SentimentResult


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def filter_() -> SentimentFilter:
    return SentimentFilter()


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_extreme_fear(filter_: SentimentFilter) -> None:
    """Fear & Greed value=15 → extreme_fear, negative adjustment, reduce_size=True."""
    result = filter_.assess(fear_greed_value=15, volatility_pct=1.0)

    assert result.fear_greed_label == "extreme_fear"
    assert result.confidence_adjustment < 0
    assert result.should_reduce_size is True


def test_fear(filter_: SentimentFilter) -> None:
    """Fear & Greed value=30 → fear, negative adjustment, reduce_size=False."""
    result = filter_.assess(fear_greed_value=30, volatility_pct=1.0)

    assert result.fear_greed_label == "fear"
    assert result.confidence_adjustment < 0
    assert result.should_reduce_size is False


def test_neutral(filter_: SentimentFilter) -> None:
    """Fear & Greed value=50 → neutral, zero adjustment."""
    result = filter_.assess(fear_greed_value=50, volatility_pct=1.0)

    assert result.fear_greed_label == "neutral"
    assert result.confidence_adjustment == 0.0


def test_greed(filter_: SentimentFilter) -> None:
    """Fear & Greed value=70 → greed, negative adjustment."""
    result = filter_.assess(fear_greed_value=70, volatility_pct=1.0)

    assert result.fear_greed_label == "greed"
    assert result.confidence_adjustment < 0


def test_extreme_greed(filter_: SentimentFilter) -> None:
    """Fear & Greed value=90 → extreme_greed, negative adjustment, reduce_size=True."""
    result = filter_.assess(fear_greed_value=90, volatility_pct=1.0)

    assert result.fear_greed_label == "extreme_greed"
    assert result.confidence_adjustment < 0
    assert result.should_reduce_size is True


def test_high_volatility(filter_: SentimentFilter) -> None:
    """Volatility=3.0% → volatility_level=high, extra negative adjustment vs neutral."""
    neutral_result = filter_.assess(fear_greed_value=50, volatility_pct=1.0)
    high_vol_result = filter_.assess(fear_greed_value=50, volatility_pct=3.0)

    assert high_vol_result.volatility_level == "high"
    assert high_vol_result.confidence_adjustment < neutral_result.confidence_adjustment


def test_extreme_volatility(filter_: SentimentFilter) -> None:
    """Volatility=5.0% → volatility_level=extreme, reduce_size=True."""
    result = filter_.assess(fear_greed_value=50, volatility_pct=5.0)

    assert result.volatility_level == "extreme"
    assert result.should_reduce_size is True


def test_result_fields(filter_: SentimentFilter) -> None:
    """All SentimentResult fields are present and within valid ranges."""
    result = filter_.assess(fear_greed_value=45, volatility_pct=2.5)

    assert isinstance(result, SentimentResult)
    assert isinstance(result.fear_greed_value, int)
    assert 0 <= result.fear_greed_value <= 100
    assert result.fear_greed_label in {
        "extreme_fear", "fear", "neutral", "greed", "extreme_greed"
    }
    assert result.volatility_level in {"low", "normal", "high", "extreme"}
    assert -0.15 <= result.confidence_adjustment <= 0.05
    assert isinstance(result.should_reduce_size, bool)
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0
