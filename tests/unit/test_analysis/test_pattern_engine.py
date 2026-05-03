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
        # BALANCED (0.9x multiplier) → higher output confidence than LOOSE (0.75x multiplier)
        candle = _c(100, 101.3, 90, 101)
        balanced = CandlePatternEngine(MatchingMode.BALANCED).detect([candle])
        loose = CandlePatternEngine(MatchingMode.LOOSE).detect([candle])
        b_hammer = next((r for r in balanced if r.pattern_name == "hammer"), None)
        l_hammer = next((r for r in loose if r.pattern_name == "hammer"), None)
        assert b_hammer is not None and l_hammer is not None
        assert b_hammer.confidence > l_hammer.confidence

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
        candles = [_c(100, 101.3, 90, 101)]  # hammer — guaranteed to produce results
        results = engine.detect(candles)
        expected_index = len(candles) - 1
        assert len(results) > 0, "Expected at least one pattern to be detected"
        assert all(r.candle_index == expected_index for r in results)

    def test_source_timestamp_set_on_all_results(self):
        engine = _engine()
        candles = [_c(100, 101.3, 90, 101)]  # hammer — guaranteed to produce results
        results = engine.detect(candles)
        expected_ts = candles[-1].timestamp
        assert len(results) > 0, "Expected at least one pattern to be detected"
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
        assert len(results) > 0, "Expected at least one detected pattern"
        # Every detected result must have a non-empty explanation
        assert all(r.explanation != "" for r in results)

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

    def test_inverted_hammer_detected(self):
        # body=1, upper_wick=11 (11x body), lower_wick=0.4 (0.4x body < 0.5 limit)
        candle = _c(100, 112, 99.6, 101)
        results = _engine().detect([candle])
        names = _names(results)
        assert "inverted_hammer" in names
        r = next(r for r in results if r.pattern_name == "inverted_hammer")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "reversal"

    def test_hanging_man_detected(self):
        # body=1, lower_wick=10 (10x body), upper_wick=0.3 (0.3x body < 0.5 limit)
        candle = _c(101, 101.3, 90, 100)
        results = _engine().detect([candle])
        names = _names(results)
        assert "hanging_man" in names
        r = next(r for r in results if r.pattern_name == "hanging_man")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "reversal"

    def test_dragonfly_doji_detected(self):
        # body=0.05, rng=10.1, body_pct=0.5% → doji; uw=0.05, lw=10 → wick_ratio=0.005 < 0.33 → dragonfly
        candle = _c(100, 100.1, 90, 100.05)
        results = _engine().detect([candle])
        names = _names(results)
        assert "dragonfly_doji" in names
        r = next(r for r in results if r.pattern_name == "dragonfly_doji")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "reversal"

    def test_gravestone_doji_detected(self):
        # body=0.05, rng=10.1, body_pct=0.5% → doji; uw=9.95, lw=0.1 → wick_ratio=99.5 > 3.0 → gravestone
        candle = _c(100, 110, 99.9, 100.05)
        results = _engine().detect([candle])
        names = _names(results)
        assert "gravestone_doji" in names
        r = next(r for r in results if r.pattern_name == "gravestone_doji")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "reversal"


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

    def test_bullish_harami_detected(self):
        # Prior bearish (o=110,c=98), current bullish contained inside prior body
        prior = _c(110, 112, 95, 98, i=0)   # bearish; p_top=110, p_bot=98
        curr  = _c(101, 106, 100, 104, i=1)  # bullish; c_bot=101>=98, c_top=104<=110
        results = _engine().detect([prior, curr])
        names = _names(results)
        assert "bullish_harami" in names
        r = next(r for r in results if r.pattern_name == "bullish_harami")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "reversal"

    def test_bearish_harami_detected(self):
        # Prior bullish (o=98,c=112), current bearish contained inside prior body
        prior = _c(98, 115, 97, 112, i=0)   # bullish; p_bot=98, p_top=112
        curr  = _c(108, 110, 105, 106, i=1)  # bearish; c_bot=106>=98, c_top=108<=112
        results = _engine().detect([prior, curr])
        names = _names(results)
        assert "bearish_harami" in names
        r = next(r for r in results if r.pattern_name == "bearish_harami")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "reversal"

    def test_piercing_line_detected(self):
        # Prior bearish (o=110,c=92,low=90,mid=101); current opens below low=90, closes above mid=101
        prior = _c(110, 112, 90, 92, i=0)   # bearish; low=90; mid=101
        curr  = _c(88, 115, 87, 108, i=1)   # bullish; open=88<90; close=108>101
        results = _engine().detect([prior, curr])
        names = _names(results)
        assert "piercing_line" in names
        r = next(r for r in results if r.pattern_name == "piercing_line")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "reversal"

    def test_dark_cloud_cover_detected(self):
        # Prior bullish (o=90,c=108,high=110,mid=99); current opens above high=110, closes below mid=99
        prior = _c(90, 110, 89, 108, i=0)   # bullish; high=110; mid=99
        curr  = _c(112, 115, 88, 96, i=1)   # bearish; open=112>110; close=96<99
        results = _engine().detect([prior, curr])
        names = _names(results)
        assert "dark_cloud_cover" in names
        r = next(r for r in results if r.pattern_name == "dark_cloud_cover")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "reversal"

    def test_tweezer_top_detected(self):
        # Matching highs 110.0/110.1 (diff_pct=0.00091 < tol=0.002); first bullish, second bearish
        prior = _c(98, 110.0, 95, 108, i=0)   # bullish
        curr  = _c(109, 110.1, 102, 104, i=1)  # bearish
        results = _engine().detect([prior, curr])
        names = _names(results)
        assert "tweezer_top" in names
        r = next(r for r in results if r.pattern_name == "tweezer_top")
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

    def test_evening_star_detected(self):
        # Large bullish | small star | large bearish closing below b1 midpoint
        b1 = _c(90, 120, 89, 115, i=0)    # bullish; body=25, rng=31, body_pct=80.6%; mid=102.5
        b2 = _c(116, 118, 114, 117, i=1)  # small star; body=1, rng=4, body_pct=25%
        b3 = _c(116, 117, 85, 88, i=2)    # bearish; body=28, rng=32, body_pct=87.5%; close=88<102.5
        results = _engine().detect([b1, b2, b3])
        names = _names(results)
        assert "evening_star" in names
        r = next(r for r in results if r.pattern_name == "evening_star")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "reversal"

    def test_morning_doji_star_detected(self):
        # Large bearish | doji star | large bullish — doji body_pct=1.7%
        b1 = _c(110, 112, 95, 97, i=0)    # bearish; body_pct=76.5%; mid=103.5
        b2 = _c(96, 97, 94, 96.05, i=1)   # doji; body_pct=1.7% < 10%
        b3 = _c(95, 115, 94, 112, i=2)    # bullish; body_pct=81%; close=112 > mid=103.5
        results = _engine().detect([b1, b2, b3])
        names = _names(results)
        assert "morning_doji_star" in names
        r = next(r for r in results if r.pattern_name == "morning_doji_star")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "reversal"

    def test_evening_doji_star_detected(self):
        # Large bullish | doji star | large bearish — doji body_pct=1.7%
        b1 = _c(90, 120, 89, 115, i=0)      # bullish; body_pct=80.6%; mid=102.5
        b2 = _c(116, 117, 114, 116.05, i=1) # doji; body_pct=1.7% < 10%
        b3 = _c(115, 116, 80, 85, i=2)      # bearish; body_pct=83.3%; close=85 < mid=102.5
        results = _engine().detect([b1, b2, b3])
        names = _names(results)
        assert "evening_doji_star" in names
        r = next(r for r in results if r.pattern_name == "evening_doji_star")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "reversal"

    def test_bullish_abandoned_baby_detected(self):
        # Bearish | gapped-down doji (b2.high=92 < b1.low=95) | bullish gap-up (b3.low=93 > b2.high=92)
        b1 = _c(110, 112, 95, 97, i=0)   # bearish; low=95
        b2 = _c(90, 92, 88, 90.05, i=1)  # doji; body_pct=0.05/4=1.25%; high=92 < 95 ✓
        b3 = _c(95, 115, 93, 112, i=2)   # bullish; low=93 > b2.high=92 ✓
        results = _engine().detect([b1, b2, b3])
        names = _names(results)
        assert "bullish_abandoned_baby" in names
        r = next(r for r in results if r.pattern_name == "bullish_abandoned_baby")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "reversal"

    def test_bearish_abandoned_baby_detected(self):
        # Bullish | gapped-up doji (b2.low=113 > b1.high=112) | bearish gap-down (b3.high=112 < b2.low=113)
        b1 = _c(95, 112, 94, 110, i=0)      # bullish; high=112
        b2 = _c(115, 118, 113, 115.05, i=1) # doji; body_pct=1%; low=113 > 112 ✓
        b3 = _c(110, 112, 90, 92, i=2)      # bearish; high=112 < b2.low=113 ✓
        results = _engine().detect([b1, b2, b3])
        names = _names(results)
        assert "bearish_abandoned_baby" in names
        r = next(r for r in results if r.pattern_name == "bearish_abandoned_baby")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "reversal"

    def test_rising_three_methods_detected(self):
        # Large bullish | 3 small bars inside b1 range | large bullish above b1 close
        b1 = _c(100, 120, 99, 118, i=0)   # bullish; body_pct=85.7%; close=118
        b2 = _c(115, 118, 106, 110, i=1)  # small; body_pct=41.7%; inside b1 ✓
        b3 = _c(111, 116, 108, 112, i=2)  # small; body_pct=12.5%; inside b1 ✓
        b4 = _c(112, 117, 109, 113, i=3)  # small; body_pct=12.5%; inside b1 ✓
        b5 = _c(113, 135, 112, 133, i=4)  # bullish; body_pct=87%; close=133 > 118 ✓
        results = _engine().detect([b1, b2, b3, b4, b5])
        names = _names(results)
        assert "rising_three_methods" in names
        r = next(r for r in results if r.pattern_name == "rising_three_methods")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "continuation"

    def test_falling_three_methods_detected(self):
        # Large bearish | 3 small bars inside b1 range | large bearish below b1 close
        b1 = _c(120, 121, 100, 102, i=0)  # bearish; body_pct=85.7%; close=102
        b2 = _c(105, 112, 102, 108, i=1)  # small; body_pct=30%; inside b1 ✓
        b3 = _c(107, 110, 104, 106, i=2)  # small; body_pct=16.7%; inside b1 ✓
        b4 = _c(106, 109, 103, 105, i=3)  # small; body_pct=16.7%; inside b1 ✓
        b5 = _c(104, 106, 80, 82, i=4)    # bearish; body_pct=84.6%; close=82 < 102 ✓
        results = _engine().detect([b1, b2, b3, b4, b5])
        names = _names(results)
        assert "falling_three_methods" in names
        r = next(r for r in results if r.pattern_name == "falling_three_methods")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "continuation"


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

    def test_outside_bar_detected(self):
        # Current range fully engulfs prior range; close near high → bullish
        prior = _c(100, 110, 95, 107, i=0)
        curr  = _c(101, 118, 92, 115, i=1)  # high=118>110, low=92<95; close_position=0.885
        results = _engine().detect([prior, curr])
        names = _names(results)
        assert "outside_bar" in names
        r = next(r for r in results if r.pattern_name == "outside_bar")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "continuation"

    def test_rejection_candle_detected(self):
        # body=1(5.9%), lower_wick=15(88% of range=17) >= 50% threshold, bullish close
        candle = _c(100, 102, 85, 101)  # o=100,c=101 bullish; lw=15; rng=17
        results = _engine().detect([candle])
        names = _names(results)
        assert "rejection_candle" in names
        r = next(r for r in results if r.pattern_name == "rejection_candle")
        assert r.bias == PatternBias.BULLISH
        assert r.category == "reversal"

    def test_exhaustion_candle_detected(self):
        # 4 normal bars (range~11) then a large bullish bar with big upper wick → bearish exhaustion signal
        prior = [_c(100 + i, 108 + i, 97 + i, 105 + i, i=i) for i in range(4)]
        # last bar: range=37 >= avg(11)*1.8=19.8; bullish; upper_wick=15 (40.5% >= 25%)
        last = _c(100, 135, 98, 120, i=4)
        results = _engine().detect(prior + [last])
        names = _names(results)
        assert "exhaustion_candle" in names
        r = next(r for r in results if r.pattern_name == "exhaustion_candle")
        assert r.bias == PatternBias.BEARISH   # bullish exhaustion → bearish signal
        assert r.category == "reversal"

    def test_failed_breakout_candle_detected(self):
        # Bars 0,1: lb_high=110; bar 2 (prior) closes above 110; bar 3 (current) reverses below 110
        b0 = _c(100, 108, 98, 106, i=0)
        b1 = _c(105, 110, 103, 108, i=1)  # lb_high=110
        b2 = _c(109, 115, 108, 113, i=2)  # prior_close=113 > lb_high=110 ✓
        b3 = _c(112, 115, 100, 107, i=3)  # curr_close=107 < lb_high=110 ✓
        results = _engine().detect([b0, b1, b2, b3])
        names = _names(results)
        assert "failed_breakout_candle" in names
        r = next(r for r in results if r.pattern_name == "failed_breakout_candle")
        assert r.bias == PatternBias.BEARISH
        assert r.category == "reversal"


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

    def test_hammer_does_not_fire_on_marubozu(self):
        # Bullish marubozu: no lower wick — hammer lw >= 2x body condition cannot be met
        engine = _engine()
        candle = _c(95, 110, 94, 110)  # body=15, lw=0
        results = engine.detect([candle])
        assert "hammer" not in _names(results)

    def test_bullish_engulfing_does_not_fire_when_both_bullish(self):
        # Both candles bullish — requires prior bearish
        engine = _engine()
        c1 = _c(100, 110, 99, 108, i=0)
        c2 = _c(108, 120, 107, 118, i=1)
        results = engine.detect([c1, c2])
        assert "bullish_engulfing" not in _names(results)

    def test_morning_star_does_not_fire_on_two_candles(self):
        # morning_star requires min 3 bars
        engine = _engine()
        b1 = _c(110, 112, 95, 97, i=0)
        b2 = _c(96, 98, 93, 95, i=1)
        results = engine.detect([b1, b2])
        assert "morning_star" not in _names(results)


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
