"""
Unit tests for libs/signals/explainability.py.

4 tests covering: basic build, dominant factors populated, trace has steps,
and result field contract.  All tests use lightweight stub objects instead of
the real pipeline types to keep the tests fully isolated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from libs.signals.explainability import ExplainabilityEngine, ExplanationTree


# ── Stub objects (lightweight replacements for real pipeline types) ────────────

@dataclass
class _BiasResult:
    net_bias: str = "bullish"
    bullish_score: float = 0.8
    bearish_score: float = 0.2


@dataclass
class _IndicatorBias:
    name: str
    direction: str
    strength: float
    explanation: str


@dataclass
class _IndicatorBiasReport:
    indicators: list[_IndicatorBias] = field(default_factory=list)
    net_bias: str = "bullish"


@dataclass
class _StructureEvent:
    kind: str = "BOS"
    direction: str = "bullish"
    price: float = 101.5


@dataclass
class _StructureAnalysis:
    trend_bias: str = "bullish"
    strength: float = 0.75
    events: list[_StructureEvent] = field(default_factory=list)


@dataclass
class _GradingResult:
    setup_grade: str = "A"
    quality_score: float = 72.0


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> ExplainabilityEngine:
    return ExplainabilityEngine()


def _make_full_result(engine: ExplainabilityEngine) -> ExplanationTree:
    """Build a fully populated ExplanationTree using all pipeline components."""
    bias = _BiasResult(net_bias="bullish", bullish_score=0.8, bearish_score=0.2)

    indicators = _IndicatorBiasReport(
        indicators=[
            _IndicatorBias("RSI", "bullish", 0.75, "RSI above 50 — bullish momentum"),
            _IndicatorBias("MACD", "bullish", 0.65, "MACD crossed above signal line"),
            _IndicatorBias("ADX", "bearish", 0.4, "ADX trending lower — weakening trend"),
        ],
        net_bias="bullish",
    )

    structure = _StructureAnalysis(
        trend_bias="bullish",
        strength=0.75,
        events=[_StructureEvent("BOS", "bullish", 101.5)],
    )

    grading = _GradingResult(setup_grade="A", quality_score=72.0)

    return engine.build(
        bias_result=bias,
        indicator_bias=indicators,
        structure=structure,
        regime=None,
        grading_result=grading,
        strategy_name="ema_crossover",
        action="BUY",
        signal_id="sig-001",
        symbol="AAPL",
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestExplainabilityEngine:

    def test_builds_explanation(self, engine: ExplainabilityEngine) -> None:
        """build() should return an ExplanationTree with correct identity fields."""
        result = _make_full_result(engine)

        assert isinstance(result, ExplanationTree)
        assert result.signal_id == "sig-001"
        assert result.symbol == "AAPL"
        assert result.action == "BUY"

    def test_dominant_factors_populated(self, engine: ExplainabilityEngine) -> None:
        """Bullish bias with BUY action should produce at least one dominant factor."""
        result = _make_full_result(engine)

        assert len(result.dominant_factors) >= 1
        # The directional bias entry should be present
        assert any("bullish" in f.lower() for f in result.dominant_factors)

    def test_trace_has_steps(self, engine: ExplainabilityEngine) -> None:
        """decision_trace should include bias, indicator, and final decision steps."""
        result = _make_full_result(engine)

        assert len(result.decision_trace) >= 3
        # Bias step
        assert any("Bias:" in step for step in result.decision_trace)
        # Final decision step
        assert any("Decision:" in step for step in result.decision_trace)
        # Strategy step
        assert any("Strategy:" in step for step in result.decision_trace)

    def test_result_fields(self, engine: ExplainabilityEngine) -> None:
        """ExplanationTree should expose all required fields with correct types."""
        result = _make_full_result(engine)

        assert isinstance(result.signal_id, str)
        assert isinstance(result.symbol, str)
        assert isinstance(result.action, str)
        assert isinstance(result.dominant_factors, list)
        assert isinstance(result.supporting_factors, list)
        assert isinstance(result.opposing_factors, list)
        assert isinstance(result.blocked_factors, list)
        assert isinstance(result.confidence_breakdown, dict)
        assert isinstance(result.decision_trace, list)


class TestExplanationTreeImmutability:

    def test_result_is_frozen(self, engine: ExplainabilityEngine) -> None:
        """ExplanationTree should be immutable (frozen dataclass)."""
        result = _make_full_result(engine)
        with pytest.raises((AttributeError, TypeError)):
            result.action = "SELL"  # type: ignore[misc]


class TestOptionalComponents:

    def test_build_without_indicators(self, engine: ExplainabilityEngine) -> None:
        """build() should succeed when indicator_bias is None."""
        bias = _BiasResult()
        result = engine.build(
            bias_result=bias,
            indicator_bias=None,
            structure=None,
            regime=None,
            grading_result=None,
            strategy_name="range_fade",
            action="BUY",
            signal_id="sig-002",
            symbol="BTCUSD",
        )
        assert isinstance(result, ExplanationTree)
        assert result.signal_id == "sig-002"

    def test_build_opposing_factors_captured(
        self, engine: ExplainabilityEngine
    ) -> None:
        """Indicators that oppose the action should appear in opposing_factors."""
        bias = _BiasResult(net_bias="bullish", bullish_score=0.7, bearish_score=0.3)
        indicators = _IndicatorBiasReport(
            indicators=[
                _IndicatorBias("RSI", "bearish", 0.6, "RSI overbought — bearish signal"),
            ],
            net_bias="bullish",
        )
        result = engine.build(
            bias_result=bias,
            indicator_bias=indicators,
            structure=None,
            regime=None,
            grading_result=None,
            strategy_name="pullback_continuation",
            action="BUY",
            signal_id="sig-003",
            symbol="AAPL",
        )
        assert len(result.opposing_factors) >= 1
        assert any("RSI" in f for f in result.opposing_factors)

    def test_confidence_breakdown_populated_for_aligned_indicators(
        self, engine: ExplainabilityEngine
    ) -> None:
        """Aligned indicator strengths should appear in confidence_breakdown."""
        bias = _BiasResult()
        indicators = _IndicatorBiasReport(
            indicators=[
                _IndicatorBias("MACD", "bullish", 0.8, "MACD bullish crossover"),
            ],
            net_bias="bullish",
        )
        result = engine.build(
            bias_result=bias,
            indicator_bias=indicators,
            structure=None,
            regime=None,
            grading_result=None,
            strategy_name="macd_crossover",
            action="BUY",
            signal_id="sig-004",
            symbol="AAPL",
        )
        assert "MACD" in result.confidence_breakdown
        assert result.confidence_breakdown["MACD"] == 0.8
