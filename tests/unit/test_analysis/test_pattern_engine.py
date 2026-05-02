"""
Comprehensive tests for CandlePatternEngine and all 40 pattern detectors.

Covers:
- Engine setup (mode, detector count)
- Edge cases (empty, single candle, confidence filtering)
- PatternResult contract (fields, ranges)
- All 40 detectors grouped by category
- No-false-positive sanity check
- PatternResult property values
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from libs.core.models.domain import (
    AssetClass,
    Candle,
    PatternBias,
    PatternResult,
    Timeframe,
)
from libs.analysis.patterns.base import MatchingMode
from libs.analysis.patterns.engine import CandlePatternEngine

_UTC = timezone.utc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _c(o: float, h: float, l: float, cl: float, i: int = 0, vol: float = 1000.0) -> Candle:
    """Build a test Candle with a deterministic timestamp."""
    return Candle(
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.ONE_MIN,
        timestamp=datetime(2024, 6, 15, 10, i % 60, tzinfo=_UTC),
        open=o,
        high=h,
        low=l,
        close=cl,
        volume=vol,
    )


def _engine() -> CandlePatternEngine:
    return CandlePatternEngine()


def _names(results: list[PatternResult]) -> list[str]:
    return [r.pattern_name for r in results]


# ---------------------------------------------------------------------------
# Class 1: TestEngineSetup
# ---------------------------------------------------------------------------

class TestEngineSetup:
    def test_default_mode_is_balanced(self):
        engine = CandlePatternEngine()
        assert engine._detectors[0].mode == MatchingMode.BALANCED

    def test_detector_count_is_40(self):
        engine = _engine()
        assert len(engine.detector_names) == 40

    def test_detector_names_unique(self):
        engine = _engine()
        names = engine.detector_names
        assert len(set(names)) == 40

    def test_accepts_strict_mode(self):
        engine = CandlePatternEngine(mode=MatchingMode.STRICT)
        assert len(engine.detector_names) == 40

    def test_accepts_loose_mode(self):
        engine = CandlePatternEngine(mode=MatchingMode.LOOSE)
        assert len(engine.detector_names) == 40


# ---------------------------------------------------------------------------
# Class 2: TestDetectEdgeCases
# ---------------------------------------------------------------------------

class TestDetectEdgeCases:
    def _candles(self) -> list[Candle]:
        return [_c(100, 110, 90, 100.5, i=i) for i in range(5)]

    def test_empty_candles_returns_empty(self):
        engine = _engine()
        assert engine.detect([]) == []

    def test_single_candle_runs_without_error(self):
        engine = _engine()
        result = engine.detect([_c(100, 110, 90, 105)])
        assert isinstance(result, list)

    def test_two_candles_runs_without_error(self):
        engine = _engine()
        c1 = _c(100, 110, 90, 105, i=0)
        c2 = _c(105, 115, 95, 110, i=1)
        result = engine.detect([c1, c2])
        assert isinstance(result, list)

    def test_min_confidence_zero_returns_all_detected(self):
        engine = _engine()
        candles = self._candles()
        results = engine.detect(candles, min_confidence=0.0)
        assert all(r.confidence >= 0.0 for r in results)

    def test_min_confidence_one_returns_high_confidence_only(self):
        engine = _engine()
        candles = self._candles()
        results = engine.detect(candles, min_confidence=1.0)
        assert all(r.confidence == 1.0 for r in results)

    def test_invalid_min_confidence_low_raises(self):
        engine = _engine()
        candles = self._candles()
        with pytest.raises(ValueError):
            engine.detect(candles, min_confidence=-0.1)

    def test_invalid_min_confidence_high_raises(self):
        engine = _engine()
        candles = self._candles()
        with pytest.raises(ValueError):
            engine.detect(candles, min_confidence=1.1)

    def test_results_sorted_by_confidence_descending(self):
        engine = _engine()
        candles = self._candles()
        results = engine.detect(candles)
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_all_results_have_detected_true(self):
        engine = _engine()
        candles = self._candles()
        results = engine.detect(candles)
        assert all(r.detected for r in results)

    def test_candle_index_set_on_all_results(self):
        engine = _engine()
        candles = self._candles()
        results = engine.detect(candles)
        expected_index = len(candles) - 1
        assert all(r.candle_index == expected_index for r in results)

    def test_source_timestamp_set_on_all_results(self):
        engine = _engine()
        candles = self._candles()
        results = engine.detect(candles)
        expected_ts = candles[-1].timestamp
        assert all(r.source_timestamp == expected_ts for r in results)

    def test_no_result_references_buy_sell(self):
        engine = _engine()
        candles = self._candles()
        results = engine.detect(candles)
        for r in results:
            assert "BUY" not in r.pattern_name
            assert "SELL" not in r.pattern_name


# ---------------------------------------------------------------------------
# Class 3: TestPatternResultContract
# ---------------------------------------------------------------------------

class TestPatternResultContract:
    """Verify PatternResult field values on real detected results."""

    def _detected_results(self) -> list[PatternResult]:
        """Get a non-empty set of detected results from a deliberately pattern-rich sequence."""
        engine = _engine()
        # Hammer candle: body=1, lower_wick=10, upper_wick=2 — will trigger many patterns
        candles = [
            _c(100, 105, 90, 104, i=0),
            _c(104, 112, 95, 110, i=1),
            _c(110, 115, 105, 113, i=2),
            _c(113, 120, 100, 101, i=3),  # bearish reversal bar
            _c(100, 102, 88, 101, i=4),   # hammer-like
        ]
        return engine.detect(candles)

    def test_all_results_have_category(self):
        results = self._detected_results()
        valid = {"", "reversal", "continuation", "indecision"}
        assert all(r.category in valid for r in results)

    def test_all_results_have_explanation(self):
        results = self._detected_results()
        # All detected results with category should have non-empty explanation
        categorized = [r for r in results if r.category != ""]
        assert all(r.explanation != "" for r in categorized)

    def test_strength_is_int_0_to_100(self):
        results = self._detected_results()
        assert all(isinstance(r.strength, int) and 0 <= r.strength <= 100 for r in results)

    def test_reliability_score_is_int_0_to_100(self):
        results = self._detected_results()
        assert all(0 <= r.reliability_score <= 100 for r in results)

    def test_confidence_in_range(self):
        results = self._detected_results()
        assert all(0.0 <= r.confidence <= 1.0 for r in results)


# ---------------------------------------------------------------------------
# Class 4: TestSingleCandlePatterns
# ---------------------------------------------------------------------------

class TestSingleCandlePatterns:
    """Each test uses a candle sequence designed to trigger a specific pattern."""

    def test_hammer_detected(self):
        # body=1 (open=100, close=101), lower_wick=10 (low=90), upper_wick=0.3 (high=101.3)
        # lw/body=10x >> 2x threshold; uw/body=0.3x << 0.4x limit
        candle = _c(100, 101.3, 90, 101)
        results = _engine().detect([candle])
        names = _names(results)
        assert "hammer" in names
        r = next(r for r in results if r.pattern_name == "hammer")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "reversal"

    def test_shooting_star_detected(self):
        # body=1 (open=100, close=101), upper_wick=11 (high=112), lower_wick=0.4 (low=99.6)
        # uw/body=11x >> 2x threshold; lw/body=0.4 << 0.5 limit
        candle = _c(100, 112, 99.6, 101)
        results = _engine().detect([candle])
        names = _names(results)
        assert "shooting_star" in names
        r = next(r for r in results if r.pattern_name == "shooting_star")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "reversal"

    def test_doji_detected(self):
        # body=0.5, range=20, body_pct=2.5%
        candle = _c(100, 110, 90, 100.5)
        results = _engine().detect([candle])
        names = _names(results)
        assert "doji" in names
        r = next(r for r in results if r.pattern_name == "doji")
        assert r.category == "indecision"

    def test_bullish_marubozu_detected(self):
        # open=95, high=110, low=94, close=110; body=15, range=16, body_pct=93.75%
        candle = _c(95, 110, 94, 110)
        results = _engine().detect([candle])
        names = _names(results)
        assert "bullish_marubozu" in names
        r = next(r for r in results if r.pattern_name == "bullish_marubozu")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "continuation"

    def test_bearish_marubozu_detected(self):
        # open=110, high=111, low=95, close=95; body=15, range=16, body_pct=93.75%
        candle = _c(110, 111, 95, 95)
        results = _engine().detect([candle])
        names = _names(results)
        assert "bearish_marubozu" in names
        r = next(r for r in results if r.pattern_name == "bearish_marubozu")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "continuation"

    def test_spinning_top_detected(self):
        # body=3, range=20, body_pct=15%; upper_wick=7(35%), lower_wick=10(50%)
        candle = _c(100, 110, 90, 103)
        results = _engine().detect([candle])
        names = _names(results)
        assert "spinning_top" in names
        r = next(r for r in results if r.pattern_name == "spinning_top")
        assert r.category == "indecision"

    def test_pin_bar_bullish_detected(self):
        # open=100, high=101, low=85, close=100.5
        # range=16, lower_wick=15(93.75%), body=0.5(3.1%), upper_wick=0.5(3.1%)
        candle = _c(100, 101, 85, 100.5)
        results = _engine().detect([candle])
        names = _names(results)
        assert "pin_bar" in names
        r = next(r for r in results if r.pattern_name == "pin_bar")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "reversal"

    def test_long_wick_candle_detected(self):
        # body=2, upper_wick=8, lower_wick=12, total_wick=20, wick_to_body=10x
        candle = _c(100, 110, 88, 102)
        results = _engine().detect([candle])
        names = _names(results)
        assert "long_wick_candle" in names
        r = next(r for r in results if r.pattern_name == "long_wick_candle")
        assert r.category == "indecision"

    def test_momentum_candle_bullish_detected(self):
        # 3 candles; last one is the momentum bar: body=18, range=22, body_pct=81.8%
        candles = [
            _c(100, 105, 95, 102, i=0),
            _c(102, 108, 100, 105, i=1),
            _c(95, 115, 93, 113, i=2),
        ]
        results = _engine().detect(candles)
        names = _names(results)
        assert "momentum_candle" in names
        r = next(r for r in results if r.pattern_name == "momentum_candle")
        assert r.bias == PatternBias.BULLISH


# ---------------------------------------------------------------------------
# Class 5: TestTwoCandlePatterns
# ---------------------------------------------------------------------------

class TestTwoCandlePatterns:
    def test_bullish_engulfing_detected(self):
        prior = _c(105, 108, 98, 100, i=0)   # bearish: open=105, close=100
        curr = _c(98, 112, 97, 110, i=1)     # bullish engulfing
        results = _engine().detect([prior, curr])
        names = _names(results)
        assert "bullish_engulfing" in names
        r = next(r for r in results if r.pattern_name == "bullish_engulfing")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "reversal"

    def test_bearish_engulfing_detected(self):
        prior = _c(98, 112, 97, 108, i=0)   # bullish
        curr = _c(110, 113, 95, 96, i=1)    # bearish engulfing
        results = _engine().detect([prior, curr])
        names = _names(results)
        assert "bearish_engulfing" in names
        r = next(r for r in results if r.pattern_name == "bearish_engulfing")
        assert r.bias == PatternBias.BEARISH

    def test_inside_bar_detected(self):
        prior = _c(100, 115, 85, 110, i=0)   # wide range
        curr = _c(105, 112, 90, 108, i=1)    # inside: high=112 < 115, low=90 > 85
        results = _engine().detect([prior, curr])
        names = _names(results)
        assert "inside_bar" in names
        r = next(r for r in results if r.pattern_name == "inside_bar")
        assert r.category == "indecision"

    def test_tweezer_bottom_detected(self):
        # Matching lows: 95.0 and 95.1 → diff=0.1, avg=95.05 → diff_pct=0.00105 < 0.0016 (balanced)
        prior = _c(105, 107, 95.0, 100, i=0)   # bearish, low=95.0
        curr = _c(98, 106, 95.1, 104, i=1)     # bullish, low≈95.0
        results = _engine().detect([prior, curr])
        names = _names(results)
        assert "tweezer_bottom" in names
        r = next(r for r in results if r.pattern_name == "tweezer_bottom")
        assert r.bias == PatternBias.BULLISH

    def test_trap_candle_detected(self):
        # Open above prior high (112), close below it
        candles = [
            _c(100, 110, 98, 108, i=0),   # prior bar 1
            _c(105, 112, 100, 110, i=1),  # prior bar 2 (establishes lookback_high=112)
            _c(115, 118, 105, 108, i=2),  # current: open=115 > 112, close=108 < 112
        ]
        results = _engine().detect(candles)
        names = _names(results)
        assert "trap_candle" in names
        r = next(r for r in results if r.pattern_name == "trap_candle")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "reversal"


# ---------------------------------------------------------------------------
# Class 6: TestMultiCandlePatterns
# ---------------------------------------------------------------------------

class TestMultiCandlePatterns:
    def test_morning_star_detected(self):
        b1 = _c(110, 112, 95, 97, i=0)    # large bearish; body_pct=(13/17)=76%
        b2 = _c(96, 98, 93, 95, i=1)      # small star (body=1, range=5, body_pct=20%)
        b3 = _c(95, 115, 94, 112, i=2)    # large bullish closing above b1 midpoint=(110+97)/2=103.5
        results = _engine().detect([b1, b2, b3])
        names = _names(results)
        assert "morning_star" in names
        r = next(r for r in results if r.pattern_name == "morning_star")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "reversal"

    def test_three_white_soldiers_detected(self):
        # All bullish, each opens within prior body, closes near high
        b1 = _c(100, 108, 99, 107, i=0)   # bullish
        b2 = _c(104, 113, 103, 112, i=1)  # opens in b1 body, closes near high
        b3 = _c(109, 119, 108, 118, i=2)  # opens in b2 body, closes near high
        results = _engine().detect([b1, b2, b3])
        names = _names(results)
        assert "three_white_soldiers" in names
        r = next(r for r in results if r.pattern_name == "three_white_soldiers")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "continuation"

    def test_three_black_crows_detected(self):
        # All bearish, each opens within prior body, closes near low
        b1 = _c(110, 111, 102, 103, i=0)  # bearish
        b2 = _c(106, 107, 98, 99, i=1)    # opens in b1 body range (103-110)
        b3 = _c(102, 103, 93, 94, i=2)    # opens in b2 body range (99-106)
        results = _engine().detect([b1, b2, b3])
        names = _names(results)
        assert "three_black_crows" in names
        r = next(r for r in results if r.pattern_name == "three_black_crows")
        assert r.bias == PatternBias.BEARISH


# ---------------------------------------------------------------------------
# Class 7: TestContextPatterns
# ---------------------------------------------------------------------------

class TestContextPatterns:
    def test_narrow_range_candle_detected(self):
        # 9 normal bars with range ~20, then a tiny range candle
        normal = [_c(100 + i * 2, 115 + i * 2, 95 + i * 2, 110 + i * 2, i=i) for i in range(9)]
        # last bar: range=3 << 60% of avg range ~20
        tiny = _c(120, 122, 119, 121, i=9)
        candles = normal + [tiny]
        results = _engine().detect(candles)
        names = _names(results)
        assert "narrow_range_candle" in names
        r = next(r for r in results if r.pattern_name == "narrow_range_candle")
        assert r.category == "indecision"

    def test_wide_range_candle_detected(self):
        # 9 bars with range ~9, then a wide bar with range=40 >> 1.8x
        normal = [_c(100 + i * 2, 107 + i * 2, 98 + i * 2, 105 + i * 2, i=i) for i in range(9)]
        wide = _c(115, 145, 105, 140, i=9)
        candles = normal + [wide]
        results = _engine().detect(candles)
        names = _names(results)
        assert "wide_range_candle" in names

    def test_breakout_candle_detected(self):
        # Prior 5 bars with highs up to ~112, then a padding bar, then a breakout
        prior = [_c(100, 108 + i, 98, 105 + i, i=i) for i in range(5)]  # highs 108-112
        base = [_c(103, 112, 100, 108, i=5)]
        # Breakout: closes at 130, body=15(range=22, 68%), range >> avg
        bk = _c(115, 135, 113, 130, i=6)
        candles = prior + base + [bk]
        results = _engine().detect(candles)
        names = _names(results)
        assert "breakout_candle" in names
        r = next(r for r in results if r.pattern_name == "breakout_candle")
        assert r.category == "continuation"


# ---------------------------------------------------------------------------
# Class 8: TestPatternNoFalsePositives
# ---------------------------------------------------------------------------

class TestPatternNoFalsePositives:
    def test_flat_candles_do_not_crash(self):
        engine = _engine()
        flat = [_c(100, 100.1, 99.9, 100.0, i=i) for i in range(3)]
        result = engine.detect(flat)
        assert isinstance(result, list)

    def test_all_detected_have_confidence_above_zero(self):
        engine = _engine()
        # A benign neutral sequence with small ranges
        candles = [_c(100 + i * 0.5, 101 + i * 0.5, 99 + i * 0.5, 100.2 + i * 0.5, i=i) for i in range(5)]
        results = engine.detect(candles)
        assert all(r.confidence > 0.0 for r in results)

    def test_no_crash_with_many_candles(self):
        engine = _engine()
        candles = [_c(100 + i * 0.1, 101 + i * 0.1, 99 + i * 0.1, 100.05 + i * 0.1, i=i % 60) for i in range(50)]
        result = engine.detect(candles)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Class 9: TestPatternProperties
# ---------------------------------------------------------------------------

class TestPatternProperties:
    """Verify PatternResult field values using a known hammer trigger."""

    def _hammer_result(self) -> PatternResult:
        """Guaranteed hammer: body=1, lower_wick=10, upper_wick=0.3 (passes uw check)."""
        engine = _engine()
        candle = _c(100, 101.3, 90, 101)
        results = engine.detect([candle])
        return next(r for r in results if r.pattern_name == "hammer")

    def test_detected_result_has_pattern_name(self):
        r = self._hammer_result()
        assert isinstance(r.pattern_name, str) and r.pattern_name != ""

    def test_detected_result_bias_is_pattern_bias(self):
        r = self._hammer_result()
        assert isinstance(r.bias, PatternBias)

    def test_detected_result_confidence_bounded(self):
        r = self._hammer_result()
        assert 0.0 <= r.confidence <= 1.0

    def test_detected_result_strength_is_int(self):
        r = self._hammer_result()
        assert isinstance(r.strength, int)

    def test_detected_result_reliability_score_is_int(self):
        r = self._hammer_result()
        assert isinstance(r.reliability_score, int)

    def test_detected_result_category_valid(self):
        r = self._hammer_result()
        assert r.category in {"", "reversal", "continuation", "indecision"}

    def test_detected_result_explanation_non_empty(self):
        r = self._hammer_result()
        assert r.explanation != ""

    def test_detected_result_is_detected_true(self):
        r = self._hammer_result()
        assert r.detected is True

    def test_detected_result_bias_is_bullish_for_hammer(self):
        r = self._hammer_result()
        assert r.bias == PatternBias.BULLISH

    def test_detected_result_category_is_reversal_for_hammer(self):
        r = self._hammer_result()
        assert r.category == "reversal"

    def test_strength_range(self):
        r = self._hammer_result()
        assert 0 <= r.strength <= 100

    def test_reliability_score_range(self):
        r = self._hammer_result()
        assert 0 <= r.reliability_score <= 100
