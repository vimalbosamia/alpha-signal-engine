"""
Unit tests for MarketContextCollector and MarketContext.

Tests cover:
  1. Extraction from a SignalOutput object
  2. Building from indicator values
  3. Merging two partial contexts
  4. to_dict() excludes None fields
"""
from __future__ import annotations

from unittest.mock import MagicMock


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_signal(
    timeframe_value: str = "1h",
    regime_value: str = "trending_up",
    confidence: float = 0.85,
    grade_value: str = "A+",
    confluence_total: float = 0.78,
    bullish_score: float = 0.70,
    bearish_score: float = 0.20,
    patterns: list | None = None,
) -> MagicMock:
    """Build a minimal SignalOutput-like mock."""
    timeframe = MagicMock()
    timeframe.value = timeframe_value

    regime = MagicMock()
    regime.value = regime_value

    grade = MagicMock()
    grade.value = grade_value

    confluence = MagicMock()
    confluence.weighted_total = confluence_total

    bias_scores = MagicMock()
    bias_scores.bullish_score = bullish_score
    bias_scores.bearish_score = bearish_score

    sig = MagicMock()
    sig.timeframe = timeframe
    sig.market_regime = regime
    sig.confidence = confidence
    sig.setup_grade = grade
    sig.confluence = confluence
    sig.bias_scores = bias_scores
    sig.patterns_detected = patterns or ["hammer", "golden_cross"]
    return sig


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_collect_from_signal_output():
    """Fields extracted from a SignalOutput are stored with correct values."""
    from libs.paper_trading.market_context import MarketContextCollector

    sig = _make_signal(
        timeframe_value="4h",
        regime_value="trending_up",
        confidence=0.90,
        grade_value="A+",
        confluence_total=0.82,
        bullish_score=0.75,
        bearish_score=0.15,
        patterns=["engulfing", "ema_cross"],
    )

    ctx = MarketContextCollector.from_signal(sig)

    assert ctx.timeframe == "4h"
    assert ctx.market_regime == "trending_up"
    assert ctx.entry_confidence == 0.90
    assert ctx.setup_grade == "A+"
    assert ctx.entry_confluence_score == 0.82
    # bias score = bullish - bearish
    assert abs(ctx.entry_bias_score - (0.75 - 0.15)) < 1e-9
    assert ctx.entry_patterns == ["engulfing", "ema_cross"]


def test_collect_with_indicators():
    """Indicator values are stored in the correct fields."""
    from libs.paper_trading.market_context import MarketContextCollector

    ctx = MarketContextCollector.from_indicators(
        rsi=62.5,
        macd_histogram=0.0032,
        ema_structure="bullish_stack",
        atr=120.0,
        atr_pct=0.18,
        bollinger_pct_b=0.65,
        adx=28.4,
        vwap_deviation_pct=0.42,
    )

    assert ctx.entry_rsi == 62.5
    assert ctx.entry_macd_histogram == 0.0032
    assert ctx.entry_ema_structure == "bullish_stack"
    assert ctx.entry_atr == 120.0
    assert ctx.entry_atr_pct == 0.18
    assert ctx.entry_bollinger_pct_b == 0.65
    assert ctx.entry_adx == 28.4
    assert ctx.entry_vwap_deviation_pct == 0.42

    # Signal fields untouched (None) in this partial
    assert ctx.timeframe is None
    assert ctx.market_regime is None


def test_merge_contexts():
    """Merging two partial contexts produces a combined context with all fields."""
    from libs.paper_trading.market_context import MarketContext, MarketContextCollector

    signal_ctx = MarketContextCollector.from_signal(_make_signal())
    indicator_ctx = MarketContextCollector.from_indicators(
        rsi=55.0,
        macd_histogram=-0.001,
        atr=90.0,
    )
    volume_ctx = MarketContextCollector.from_volume(
        relative=1.8,
        trend="increasing",
        orderflow_bias="buy_dominant",
        liquidity_state="deep",
    )

    merged = MarketContextCollector.merge(signal_ctx, indicator_ctx, volume_ctx)

    # Signal fields present
    assert merged.timeframe == "1h"
    assert merged.market_regime == "trending_up"
    assert merged.entry_confidence == 0.85

    # Indicator fields present
    assert merged.entry_rsi == 55.0
    assert merged.entry_macd_histogram == -0.001
    assert merged.entry_atr == 90.0

    # Volume fields present
    assert merged.entry_volume_relative == 1.8
    assert merged.entry_volume_trend == "increasing"
    assert merged.entry_orderflow_bias == "buy_dominant"
    assert merged.entry_liquidity_state == "deep"


def test_merge_later_overrides_earlier():
    """A later context's non-None values override earlier ones."""
    from libs.paper_trading.market_context import MarketContext, MarketContextCollector

    first = MarketContext(entry_rsi=40.0, entry_adx=20.0)
    second = MarketContext(entry_rsi=65.0)  # overrides rsi, leaves adx

    merged = MarketContextCollector.merge(first, second)

    assert merged.entry_rsi == 65.0       # overridden
    assert merged.entry_adx == 20.0       # preserved from first


def test_to_dict_excludes_none():
    """to_dict() returns only fields that have a non-None value."""
    from libs.paper_trading.market_context import MarketContextCollector

    ctx = MarketContextCollector.from_indicators(
        rsi=70.0,
        adx=35.0,
    )

    result = ctx.to_dict()

    assert "entry_rsi" in result
    assert "entry_adx" in result
    assert result["entry_rsi"] == 70.0
    assert result["entry_adx"] == 35.0

    # All other indicator / signal / volume / macro fields absent
    assert "entry_macd_histogram" not in result
    assert "timeframe" not in result
    assert "entry_volume_relative" not in result
    assert "entry_news_sentiment" not in result


def test_to_dict_full_context():
    """A fully populated context serialises all expected keys."""
    from libs.paper_trading.market_context import MarketContext

    ctx = MarketContext(
        timeframe="1d",
        market_regime="breakout",
        entry_confidence=0.91,
        entry_rsi=68.0,
        entry_fear_greed_index=75,
        entry_volume_relative=2.1,
        entry_session="london",
        leverage=3.0,
        execution_latency_ms=42,
    )

    result = ctx.to_dict()

    assert result["timeframe"] == "1d"
    assert result["market_regime"] == "breakout"
    assert result["entry_confidence"] == 0.91
    assert result["entry_rsi"] == 68.0
    assert result["entry_fear_greed_index"] == 75
    assert result["entry_volume_relative"] == 2.1
    assert result["entry_session"] == "london"
    assert result["leverage"] == 3.0
    assert result["execution_latency_ms"] == 42

    # None fields not present
    assert "entry_macd_histogram" not in result
    assert "entry_news_sentiment" not in result
