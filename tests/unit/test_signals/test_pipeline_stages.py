"""Tests for libs/signals/pipeline_stages.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

@dataclass
class _Indicators:
    rsi: float | None = 55.0
    rsi_prev: float | None = 52.0
    macd_line: float | None = 0.5
    macd_signal: float | None = 0.3
    macd_histogram: float | None = 0.2
    macd_histogram_prev: float | None = 0.1
    bb_upper: float | None = 105.0
    bb_lower: float | None = 95.0
    bb_pct_b: float | None = 0.6
    ema_9: float | None = 100.5
    ema_20: float | None = 100.0
    ema_50: float | None = 99.0
    adx: float | None = 25.0


@dataclass
class _RegimeTrend:
    value: str = "trending_up"


@dataclass
class _Regime:
    regime: _RegimeTrend = field(default_factory=lambda: _RegimeTrend("trending_up"))
    atr: float | None = 1.5


def _make_df(close: float = 100.0, rel_vol: float = 1.5, is_bullish: bool = True) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame."""
    return pd.DataFrame({
        "open": [close - 0.5],
        "high": [close + 1.0],
        "low": [close - 1.0],
        "close": [close],
        "volume": [1000.0],
        "relative_volume": [rel_vol],
        "is_bullish": [is_bullish],
    })


# ---------------------------------------------------------------------------
# Test 1 — compute_symbol_bias returns correct tuple shape
# ---------------------------------------------------------------------------

def test_compute_symbol_bias_returns_tuple():
    """compute_symbol_bias must return a 5-tuple with expected types."""
    from libs.analysis.bias.engine import BiasResult

    df = _make_df(close=100.0, rel_vol=1.5)
    indicators = _Indicators()
    regime = _Regime()

    # No HTF structure; no pattern results
    with (
        patch("libs.analysis.indicators.bias.IndicatorBiasAnalyzer.analyze_all") as mock_ind,
        patch("libs.analysis.structure.market_structure.MarketStructureAnalyzer.analyze") as mock_struct,
        patch("libs.analysis.bias.engine.BullBearBiasEngine.score") as mock_score,
    ):
        # Stub indicator bias report
        ind_report = MagicMock()
        ind_report.bullish_score = 0.7
        ind_report.bearish_score = 0.2
        mock_ind.return_value = ind_report

        # Stub structure result
        struct_result = MagicMock()
        struct_result.trend_bias = "bullish"
        struct_result.strength = 0.6
        mock_struct.return_value = struct_result

        # Stub bias result
        bias_result = MagicMock(spec=BiasResult)
        bias_result.net_bias = "bullish"
        mock_score.return_value = bias_result

        from libs.signals.pipeline_stages import compute_symbol_bias
        result = compute_symbol_bias(
            df=df,
            indicators=indicators,
            htf_structure=None,
            regime=regime,
            pattern_results=[],
        )

    assert isinstance(result, tuple)
    assert len(result) == 5

    bias, allowed_action, htf_bias_str, indicator_bias, deep_structure = result

    assert bias is bias_result
    # bullish net_bias → BUY action
    from libs.core.models.domain import SignalAction
    assert allowed_action == SignalAction.BUY

    assert htf_bias_str == "neutral"   # no htf_structure passed
    assert indicator_bias is ind_report
    assert deep_structure is struct_result


# ---------------------------------------------------------------------------
# Test 2 — apply_macro_filters normal path (no blocking)
# ---------------------------------------------------------------------------

def test_apply_macro_filters_normal():
    """apply_macro_filters returns (adj, False, '') when nothing blocks."""
    from libs.signals.pipeline_stages import apply_macro_filters

    df = _make_df(close=100.0)
    regime = _Regime(atr=1.5)

    with (
        patch("libs.analysis.macro.news.NewsImpactEngine.assess") as mock_news,
        patch("libs.analysis.macro.sentiment.SentimentFilter.assess") as mock_sent,
        patch("libs.analysis.macro.global_risk.GlobalRiskEngine.assess") as mock_risk,
    ):
        news_result = MagicMock()
        news_result.should_block = False
        news_result.total_confidence_impact = -0.05
        mock_news.return_value = news_result

        sent_result = MagicMock()
        sent_result.confidence_adjustment = -0.03
        mock_sent.return_value = sent_result

        risk_result = MagicMock()
        risk_result.risk_level = "low"
        risk_result.confidence_adjustment = 0.0
        mock_risk.return_value = risk_result

        adj, should_block, block_reason = apply_macro_filters(
            symbol="BTC/USD",
            asset_class_value="crypto",
            regime=regime,
            df=df,
        )

    assert should_block is False
    assert block_reason == ""
    assert isinstance(adj, float)
    assert pytest.approx(adj, abs=1e-6) == -0.08   # -0.05 + -0.03 + 0.0


# ---------------------------------------------------------------------------
# Test 3 — apply_macro_filters blocks on news
# ---------------------------------------------------------------------------

def test_apply_macro_filters_blocked():
    """apply_macro_filters returns (0.0, True, reason) when news blocks trading."""
    from libs.signals.pipeline_stages import apply_macro_filters

    df = _make_df(close=100.0)
    regime = _Regime(atr=2.0)

    with patch("libs.analysis.macro.news.NewsImpactEngine.assess") as mock_news:
        news_result = MagicMock()
        news_result.should_block = True
        news_result.explanation = "FOMC meeting — block all trades"
        mock_news.return_value = news_result

        adj, should_block, block_reason = apply_macro_filters(
            symbol="SPY",
            asset_class_value="equity",
            regime=regime,
            df=df,
        )

    assert should_block is True
    assert adj == 0.0
    assert "FOMC" in block_reason or block_reason != ""
