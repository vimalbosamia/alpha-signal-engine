"""
Unit tests for BullBearBiasEngine.

Tests follow TDD: write tests first, then implement to pass them.
Each test targets a specific contract of BiasInput / BiasResult / BullBearBiasEngine.
"""
from __future__ import annotations

import pytest

from libs.analysis.bias.engine import BiasInput, BiasResult, BullBearBiasEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> BullBearBiasEngine:
    return BullBearBiasEngine()


def _bullish_input(**overrides) -> BiasInput:
    """All-bullish baseline input."""
    defaults = dict(
        indicator_bullish=0.85,
        indicator_bearish=0.10,
        structure_bias="bullish",
        structure_strength=0.80,
        candle_bullish_count=7,
        candle_bearish_count=2,
        candle_total=9,
        regime_supports_direction=True,
        volume_confirms=True,
        htf_bias="bullish",
    )
    defaults.update(overrides)
    return BiasInput(**defaults)


def _bearish_input(**overrides) -> BiasInput:
    """All-bearish baseline input."""
    defaults = dict(
        indicator_bullish=0.10,
        indicator_bearish=0.85,
        structure_bias="bearish",
        structure_strength=0.80,
        candle_bullish_count=2,
        candle_bearish_count=7,
        candle_total=9,
        regime_supports_direction=True,
        volume_confirms=True,
        htf_bias="bearish",
    )
    defaults.update(overrides)
    return BiasInput(**defaults)


# ── Test 1: Strong bullish ────────────────────────────────────────────────────

class TestStrongBullish:
    def test_net_bias_is_bullish(self, engine):
        result = engine.score(_bullish_input())
        assert result.net_bias == "bullish"

    def test_bullish_score_exceeds_threshold(self, engine):
        result = engine.score(_bullish_input())
        assert result.bullish_score > 0.6

    def test_conflict_is_low(self, engine):
        result = engine.score(_bullish_input())
        assert result.conflict_score < 0.3

    def test_explanation_mentions_bullish(self, engine):
        result = engine.score(_bullish_input())
        assert "bullish" in result.explanation.lower()


# ── Test 2: Strong bearish ────────────────────────────────────────────────────

class TestStrongBearish:
    def test_net_bias_is_bearish(self, engine):
        result = engine.score(_bearish_input())
        assert result.net_bias == "bearish"

    def test_bearish_score_exceeds_threshold(self, engine):
        result = engine.score(_bearish_input())
        assert result.bearish_score > 0.6


# ── Test 3: Mixed signals → neutral ──────────────────────────────────────────

class TestMixedSignalsNeutral:
    def test_net_bias_is_neutral(self, engine):
        inp = BiasInput(
            indicator_bullish=0.50,
            indicator_bearish=0.50,
            structure_bias="neutral",
            structure_strength=0.50,
            candle_bullish_count=5,
            candle_bearish_count=5,
            candle_total=10,
            regime_supports_direction=True,
            volume_confirms=False,
            htf_bias="neutral",
        )
        result = engine.score(inp)
        assert result.net_bias == "neutral"

    def test_conflict_is_high(self, engine):
        inp = BiasInput(
            indicator_bullish=0.50,
            indicator_bearish=0.50,
            structure_bias="neutral",
            structure_strength=0.50,
            candle_bullish_count=5,
            candle_bearish_count=5,
            candle_total=10,
            regime_supports_direction=True,
            volume_confirms=False,
            htf_bias="neutral",
        )
        result = engine.score(inp)
        assert result.conflict_score > 0.3


# ── Test 4: HTF conflict reduces confidence ───────────────────────────────────

class TestHtfConflict:
    def test_htf_bearish_reduces_bullish_score(self, engine):
        aligned = engine.score(_bullish_input(htf_bias="bullish"))
        conflicted = engine.score(_bullish_input(htf_bias="bearish"))
        assert conflicted.bullish_score < aligned.bullish_score

    def test_htf_bearish_increases_conflict(self, engine):
        aligned = engine.score(_bullish_input(htf_bias="bullish"))
        conflicted = engine.score(_bullish_input(htf_bias="bearish"))
        assert conflicted.conflict_score > aligned.conflict_score


# ── Test 5: Result has all fields in valid ranges ─────────────────────────────

class TestResultFields:
    def test_all_score_fields_present(self, engine):
        result = engine.score(_bullish_input())
        assert hasattr(result, "bullish_score")
        assert hasattr(result, "bearish_score")
        assert hasattr(result, "neutral_score")
        assert hasattr(result, "conflict_score")
        assert hasattr(result, "net_bias")
        assert hasattr(result, "explanation")

    def test_bullish_score_in_range(self, engine):
        result = engine.score(_bullish_input())
        assert 0.0 <= result.bullish_score <= 1.0

    def test_bearish_score_in_range(self, engine):
        result = engine.score(_bullish_input())
        assert 0.0 <= result.bearish_score <= 1.0

    def test_neutral_score_in_range(self, engine):
        result = engine.score(_bullish_input())
        assert 0.0 <= result.neutral_score <= 1.0

    def test_conflict_score_in_range(self, engine):
        result = engine.score(_bullish_input())
        assert 0.0 <= result.conflict_score <= 1.0

    def test_net_bias_is_valid_string(self, engine):
        result = engine.score(_bullish_input())
        assert result.net_bias in ("bullish", "bearish", "neutral")

    def test_explanation_is_non_empty_string(self, engine):
        result = engine.score(_bullish_input())
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    def test_result_is_frozen(self, engine):
        result = engine.score(_bullish_input())
        with pytest.raises((AttributeError, TypeError)):
            result.bullish_score = 0.99  # type: ignore[misc]
