"""
Unit tests for IndicatorBiasAnalyzer.

Tests follow TDD: each test targets a specific contract of IndicatorBias /
IndicatorBiasReport / IndicatorBiasAnalyzer.
"""
from __future__ import annotations

import pytest

from libs.analysis.indicators.bias import (
    IndicatorBias,
    IndicatorBiasReport,
    IndicatorBiasAnalyzer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def analyzer() -> IndicatorBiasAnalyzer:
    return IndicatorBiasAnalyzer()


# ── RSI ───────────────────────────────────────────────────────────────────────

class TestRsiBias:
    def test_rsi_oversold_is_bullish(self, analyzer):
        bias = analyzer.rsi_bias(rsi=25.0, rsi_prev=27.0)
        assert bias.direction == "bullish"
        assert bias.strength > 0.5
        assert "oversold" in bias.explanation.lower()

    def test_rsi_overbought_is_bearish(self, analyzer):
        bias = analyzer.rsi_bias(rsi=78.0, rsi_prev=75.0)
        assert bias.direction == "bearish"
        assert bias.strength > 0.0

    def test_rsi_neutral(self, analyzer):
        bias = analyzer.rsi_bias(rsi=50.0, rsi_prev=49.0)
        assert bias.direction == "neutral"


# ── MACD ──────────────────────────────────────────────────────────────────────

class TestMacdBias:
    def test_macd_bullish_crossover(self, analyzer):
        # histogram flips from negative to positive → bullish crossover
        bias = analyzer.macd_bias(
            macd_line=0.5,
            macd_signal=0.3,
            histogram=0.2,
            histogram_prev=-0.1,
        )
        assert bias.direction == "bullish"
        assert "crossover" in bias.explanation.lower()

    def test_macd_bearish(self, analyzer):
        # negative histogram and line below signal → bearish
        bias = analyzer.macd_bias(
            macd_line=-0.5,
            macd_signal=-0.2,
            histogram=-0.3,
            histogram_prev=-0.1,
        )
        assert bias.direction == "bearish"


# ── Bollinger Bands ───────────────────────────────────────────────────────────

class TestBollingerBias:
    def test_bollinger_below_lower_is_bullish(self, analyzer):
        bias = analyzer.bollinger_bias(
            close=98.0,
            bb_upper=106.0,
            bb_lower=100.0,
            bb_pct_b=-0.1,
        )
        assert bias.direction == "bullish"

    def test_bollinger_above_upper_is_bearish(self, analyzer):
        bias = analyzer.bollinger_bias(
            close=108.0,
            bb_upper=106.0,
            bb_lower=100.0,
            bb_pct_b=1.2,
        )
        assert bias.direction == "bearish"


# ── EMA ───────────────────────────────────────────────────────────────────────

class TestEmaBias:
    def test_ema_stack_bullish(self, analyzer):
        # ema_9 > ema_20 > ema_50 and close > ema_9 → bullish
        bias = analyzer.ema_bias(
            ema_9=105.0,
            ema_20=103.0,
            ema_50=100.0,
            close=107.0,
        )
        assert bias.direction == "bullish"

    def test_ema_stack_bearish(self, analyzer):
        # ema_9 < ema_20 < ema_50 and close < ema_9 → bearish
        bias = analyzer.ema_bias(
            ema_9=95.0,
            ema_20=97.0,
            ema_50=100.0,
            close=93.0,
        )
        assert bias.direction == "bearish"


# ── ADX ───────────────────────────────────────────────────────────────────────

class TestAdxBias:
    def test_adx_strong_trend(self, analyzer):
        bias = analyzer.adx_bias(adx=35.0)
        assert bias.strength > 0.6

    def test_adx_weak(self, analyzer):
        bias = analyzer.adx_bias(adx=15.0)
        assert bias.strength < 0.4


# ── Volume ────────────────────────────────────────────────────────────────────

class TestVolumeBias:
    def test_volume_bias_spike_on_bullish_candle(self, analyzer):
        bias = analyzer.volume_bias(relative_volume=2.5, is_bullish_candle=True)
        assert bias.direction == "bullish"
        assert bias.strength > 0.6


# ── Aggregate ─────────────────────────────────────────────────────────────────

class TestAggregateAllIndicators:
    def test_aggregate_bullish_inputs_returns_report(self, analyzer):
        report = analyzer.analyze_all(
            rsi=25.0,
            rsi_prev=28.0,
            macd_line=0.5,
            macd_signal=0.3,
            histogram=0.2,
            histogram_prev=-0.05,
            close=107.0,
            bb_upper=106.0,
            bb_lower=98.0,
            bb_pct_b=1.1,
            ema_9=105.0,
            ema_20=103.0,
            ema_50=100.0,
            adx=35.0,
            relative_volume=2.5,
            is_bullish_candle=True,
        )
        assert isinstance(report, IndicatorBiasReport)
        assert len(report.indicators) == 6
        assert report.net_bias in ("bullish", "bearish", "neutral")
        assert 0.0 <= report.bullish_score <= 1.0
        assert 0.0 <= report.bearish_score <= 1.0
        assert 0.0 <= report.neutral_score <= 1.0
        assert isinstance(report.explanation, str)
        assert len(report.explanation) > 0
