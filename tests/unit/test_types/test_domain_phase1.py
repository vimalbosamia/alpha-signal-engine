"""
Phase 1 — Core Types and Standard Output Contract tests.

Covers:
  - All new enumerations
  - BiasScores, EntryZone, StrategyResult, IndicatorResult,
    RiskResult, FuturesRiskResult
  - Candle.source backward compat
  - PatternResult new fields backward compat
  - SignalOutput new fields backward compat
  - SignalOutput still serialises/deserialises correctly
  - No existing passing tests broken (checked by running full suite)
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from libs.core.models.domain import (
    # Existing
    AssetClass,
    Candle,
    ConfluenceBreakdown,
    DataQualityStatus,
    MarketRegime,
    PatternBias,
    PatternResult,
    SessionType,
    SignalAction,
    SignalOutput,
    Timeframe,
    TrendDirection,
    # Phase 1 enumerations
    Direction,
    FuturesRiskResult,
    RiskLevel,
    SetupGrade,
    SignalQuality,
    SignalType,
    TradeDecision,
    TradingMode,
    ValidationStatus,
    # Phase 1 models
    BiasScores,
    EntryZone,
    IndicatorResult,
    RiskResult,
    StrategyResult,
)


# ── Enum coverage ─────────────────────────────────────────────────────────────

class TestTradingMode:
    def test_values(self):
        assert TradingMode.SPOT.value == "spot"
        assert TradingMode.FUTURES.value == "futures"

    def test_exhaustive(self):
        assert set(TradingMode) == {TradingMode.SPOT, TradingMode.FUTURES}


class TestSignalType:
    def test_spot_values(self):
        assert SignalType.SPOT_BUY.value == "SPOT_BUY"
        assert SignalType.SPOT_SELL_EXIT.value == "SPOT_SELL_EXIT"
        assert SignalType.SPOT_NO_TRADE.value == "SPOT_NO_TRADE"

    def test_futures_values(self):
        assert SignalType.FUTURES_LONG.value == "FUTURES_LONG"
        assert SignalType.FUTURES_SHORT.value == "FUTURES_SHORT"
        assert SignalType.FUTURES_NO_TRADE.value == "FUTURES_NO_TRADE"

    def test_exhaustive(self):
        assert len(SignalType) == 6


class TestDirection:
    def test_values(self):
        assert Direction.LONG.value == "LONG"
        assert Direction.SHORT.value == "SHORT"
        assert Direction.EXIT.value == "EXIT"
        assert Direction.NO_TRADE.value == "NO_TRADE"

    def test_exhaustive(self):
        assert len(Direction) == 4


class TestRiskLevel:
    def test_order_exists(self):
        levels = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.EXTREME]
        assert len(levels) == 4

    def test_exhaustive(self):
        assert len(RiskLevel) == 4


class TestSetupGrade:
    def test_a_plus_value(self):
        assert SetupGrade.A_PLUS.value == "A+"

    def test_avoid_value(self):
        assert SetupGrade.AVOID.value == "Avoid"

    def test_exhaustive(self):
        assert len(SetupGrade) == 5


class TestTradeDecision:
    def test_values(self):
        assert TradeDecision.TAKE.value == "TAKE"
        assert TradeDecision.WAIT.value == "WAIT"
        assert TradeDecision.SKIP.value == "SKIP"
        assert TradeDecision.NO_TRADE.value == "NO_TRADE"

    def test_exhaustive(self):
        assert len(TradeDecision) == 4


class TestValidationStatus:
    def test_values(self):
        assert ValidationStatus.VALIDATED.value == "validated"
        assert ValidationStatus.UNVALIDATED.value == "unvalidated"
        assert ValidationStatus.INSUFFICIENT_DATA.value == "insufficient_data"

    def test_exhaustive(self):
        assert len(ValidationStatus) == 3


class TestSignalQuality:
    def test_values(self):
        assert SignalQuality.INSTITUTIONAL.value == "institutional"
        assert SignalQuality.REJECT.value == "reject"

    def test_exhaustive(self):
        assert len(SignalQuality) == 5


# ── BiasScores ────────────────────────────────────────────────────────────────

class TestBiasScores:
    def test_defaults(self):
        bs = BiasScores()
        assert bs.bullish_score == 0.0
        assert bs.bearish_score == 0.0
        assert bs.neutral_score == 0.0
        assert bs.conflict_score == 0.0
        assert bs.upside_probability == 0.0
        assert bs.downside_probability == 0.0

    def test_dominant_bullish(self):
        bs = BiasScores(bullish_score=0.8, bearish_score=0.1)
        assert bs.dominant_bias == "bullish"

    def test_dominant_bearish(self):
        bs = BiasScores(bearish_score=0.9, bullish_score=0.1)
        assert bs.dominant_bias == "bearish"

    def test_dominant_conflict(self):
        bs = BiasScores(conflict_score=0.95)
        assert bs.dominant_bias == "conflict"

    def test_frozen(self):
        bs = BiasScores(bullish_score=0.5)
        with pytest.raises(Exception):
            bs.bullish_score = 0.1  # type: ignore[misc]

    def test_score_bounds(self):
        with pytest.raises(Exception):
            BiasScores(bullish_score=1.5)
        with pytest.raises(Exception):
            BiasScores(bearish_score=-0.1)


# ── EntryZone ─────────────────────────────────────────────────────────────────

class TestEntryZone:
    def test_mid(self):
        ez = EntryZone(low=100.0, high=110.0)
        assert ez.mid == 105.0

    def test_width_pct(self):
        ez = EntryZone(low=99.0, high=101.0)
        assert abs(ez.width_pct - 2.0) < 0.01  # (101-99)/100 * 100 = 2%

    def test_zero_mid_safety(self):
        ez = EntryZone(low=0.0, high=0.0)
        assert ez.width_pct == 0.0

    def test_frozen(self):
        ez = EntryZone(low=100.0, high=110.0)
        with pytest.raises(Exception):
            ez.low = 50.0  # type: ignore[misc]


# ── StrategyResult ────────────────────────────────────────────────────────────

class TestStrategyResult:
    def test_defaults(self):
        sr = StrategyResult(strategy_name="hammer_reversal", produced_candidate=False)
        assert sr.confidence == 0.0
        assert sr.bullish_score == 0.0
        assert sr.bearish_score == 0.0
        assert sr.reasons == []
        assert sr.failed_confirmations == []
        assert sr.required_confirmations == []
        assert sr.risk_notes == []
        assert sr.validation_status == ValidationStatus.UNVALIDATED

    def test_with_candidate(self):
        sr = StrategyResult(
            strategy_name="rsi_mean_reversion",
            produced_candidate=True,
            confidence=0.75,
            bullish_score=0.8,
            reasons=["RSI oversold", "price at support"],
            failed_confirmations=["volume below average"],
            validation_status=ValidationStatus.VALIDATED,
        )
        assert sr.produced_candidate is True
        assert sr.confidence == 0.75
        assert len(sr.reasons) == 2
        assert len(sr.failed_confirmations) == 1

    def test_frozen(self):
        sr = StrategyResult(strategy_name="test", produced_candidate=False)
        with pytest.raises(Exception):
            sr.confidence = 0.9  # type: ignore[misc]


# ── IndicatorResult ───────────────────────────────────────────────────────────

class TestIndicatorResult:
    def test_defaults(self):
        ir = IndicatorResult(name="RSI", value=32.5)
        assert ir.bias == PatternBias.NEUTRAL
        assert ir.strength == 0.0
        assert ir.explanation == ""

    def test_full(self):
        ir = IndicatorResult(
            name="MACD",
            value=0.002,
            bias=PatternBias.BULLISH,
            strength=0.7,
            explanation="histogram turned positive",
        )
        assert ir.name == "MACD"
        assert ir.bias == PatternBias.BULLISH
        assert ir.strength == 0.7

    def test_strength_bounds(self):
        with pytest.raises(Exception):
            IndicatorResult(name="X", value=0.0, strength=1.5)


# ── RiskResult ────────────────────────────────────────────────────────────────

class TestRiskResult:
    def test_minimal(self):
        rr = RiskResult(entry=100.0, stop_loss=97.0, take_profit_1=106.0)
        assert rr.risk_reward_ratio == 0.0
        assert rr.take_profit_2 is None
        assert rr.invalidation_level is None
        assert not rr.has_warnings

    def test_has_warnings(self):
        rr = RiskResult(
            entry=100.0, stop_loss=97.0, take_profit_1=106.0,
            volatility_warning="ATR spike detected",
        )
        assert rr.has_warnings

    def test_full(self):
        rr = RiskResult(
            entry=50000.0,
            stop_loss=49000.0,
            take_profit_1=52000.0,
            take_profit_2=54000.0,
            risk_reward_ratio=2.0,
            invalidation_level=48500.0,
            max_loss_percent=2.0,
            liquidity_warning="low volume session",
        )
        assert rr.take_profit_2 == 54000.0
        assert rr.invalidation_level == 48500.0
        assert rr.has_warnings

    def test_frozen(self):
        rr = RiskResult(entry=100.0, stop_loss=97.0, take_profit_1=106.0)
        with pytest.raises(Exception):
            rr.stop_loss = 90.0  # type: ignore[misc]


# ── FuturesRiskResult ─────────────────────────────────────────────────────────

class TestFuturesRiskResult:
    def test_defaults(self):
        fr = FuturesRiskResult()
        assert fr.leverage == 1.0
        assert fr.margin_type == "isolated"
        assert fr.estimated_liquidation_price == 0.0
        assert fr.liquidation_buffer_percent == 0.0
        assert fr.liquidation_risk == RiskLevel.LOW
        assert fr.funding_fee_risk == RiskLevel.LOW
        assert fr.max_loss_before_stop == 0.0

    def test_realistic_10x(self):
        fr = FuturesRiskResult(
            leverage=10.0,
            margin_type="isolated",
            estimated_liquidation_price=45000.0,
            liquidation_buffer_percent=10.0,
            liquidation_risk=RiskLevel.MODERATE,
            funding_fee_risk=RiskLevel.LOW,
            max_loss_before_stop=1000.0,
        )
        assert fr.leverage == 10.0
        assert fr.liquidation_risk == RiskLevel.MODERATE

    def test_leverage_minimum(self):
        with pytest.raises(Exception):
            FuturesRiskResult(leverage=0.5)  # below ge=1.0

    def test_extreme_leverage_risk(self):
        fr = FuturesRiskResult(
            leverage=50.0,
            liquidation_risk=RiskLevel.EXTREME,
            funding_fee_risk=RiskLevel.HIGH,
        )
        assert fr.liquidation_risk == RiskLevel.EXTREME

    def test_frozen(self):
        fr = FuturesRiskResult()
        with pytest.raises(Exception):
            fr.leverage = 5.0  # type: ignore[misc]


# ── Candle backward compat + source field ─────────────────────────────────────

class TestCandleBackwardCompat:
    def test_no_source_still_works(self):
        c = Candle(
            symbol="BTCUSDT",
            asset_class=AssetClass.CRYPTO,
            timeframe=Timeframe.FIVE_MIN,
            timestamp=datetime.now(timezone.utc),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )
        assert c.source == ""  # default

    def test_source_set(self):
        c = Candle(
            symbol="AAPL",
            asset_class=AssetClass.STOCK,
            timeframe=Timeframe.ONE_MIN,
            timestamp=datetime.now(timezone.utc),
            open=150.0,
            high=151.0,
            low=149.5,
            close=150.5,
            volume=1000.0,
            source="alpaca",
        )
        assert c.source == "alpaca"


# ── PatternResult backward compat + new fields ────────────────────────────────

class TestPatternResultBackwardCompat:
    def test_existing_fields_unchanged(self):
        pr = PatternResult(
            pattern_name="Hammer",
            detected=True,
            confidence=0.85,
            bias=PatternBias.BULLISH,
            candle_span=1,
        )
        assert pr.pattern_name == "Hammer"
        assert pr.is_actionable is True

    def test_new_fields_default(self):
        pr = PatternResult(pattern_name="Doji", detected=False, confidence=0.0)
        assert pr.category == ""
        assert pr.reliability == 0.0
        assert pr.candle_index == -1
        assert pr.source_timestamp is None

    def test_new_fields_set(self):
        ts = datetime.now(timezone.utc)
        pr = PatternResult(
            pattern_name="MorningStar",
            detected=True,
            confidence=0.9,
            bias=PatternBias.BULLISH,
            candle_span=3,
            category="reversal",
            reliability=0.62,
            candle_index=198,
            source_timestamp=ts,
        )
        assert pr.category == "reversal"
        assert pr.reliability == 0.62
        assert pr.candle_index == 198
        assert pr.source_timestamp == ts


def _pattern_result(
    confidence: float = 0.7,
    reliability: float = 0.0,
    explanation: str = "",
    category: str = "",
) -> PatternResult:
    return PatternResult(
        pattern_name="test_pattern",
        detected=True,
        confidence=confidence,
        reliability=reliability,
        explanation=explanation,
        category=category,
        bias=PatternBias.NEUTRAL,
    )


class TestPatternResultNewProperties:
    def test_strength_zero_confidence(self):
        r = _pattern_result(confidence=0.0)
        assert r.strength == 0
        assert isinstance(r.strength, int)

    def test_strength_full_confidence(self):
        r = _pattern_result(confidence=1.0)
        assert r.strength == 100
        assert isinstance(r.strength, int)

    def test_strength_partial_confidence(self):
        r = _pattern_result(confidence=0.85)
        assert r.strength == 85

    def test_strength_rounds_not_truncates(self):
        r = _pattern_result(confidence=0.999)
        assert r.strength == 100  # rounds up, not truncates to 99

    def test_reliability_score_partial(self):
        r = _pattern_result(confidence=0.7, reliability=0.72)
        assert r.reliability_score == 72
        assert isinstance(r.reliability_score, int)

    def test_reliability_score_zero(self):
        r = _pattern_result(confidence=0.7, reliability=0.0)
        assert r.reliability_score == 0

    def test_explanation_field_stores_value(self):
        r = _pattern_result(confidence=0.8, explanation="Bullish engulfing with volume confirmation")
        assert r.explanation == "Bullish engulfing with volume confirmation"

    def test_category_valid_values(self):
        for cat in ["", "reversal", "continuation", "indecision"]:
            r = _pattern_result(confidence=0.7, category=cat)
            assert r.category == cat

    def test_category_invalid_value_raises(self):
        with pytest.raises(Exception):
            _pattern_result(confidence=0.7, category="single")  # old invalid value


# ── SignalOutput backward compat + new fields ─────────────────────────────────

def _minimal_confluence() -> ConfluenceBreakdown:
    return ConfluenceBreakdown(
        pattern_score=0.7,
        structure_score=0.8,
        level_score=0.6,
        volume_score=0.7,
        regime_score=0.6,
        session_score=0.8,
        risk_score=0.7,
        data_quality_score=1.0,
        weighted_total=0.72,
    )


def _minimal_signal_output(**overrides) -> SignalOutput:
    defaults = dict(
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        strategy_name="hammer_reversal",
        timeframe=Timeframe.FIVE_MIN,
        action=SignalAction.BUY,
        confidence=0.72,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=49900.0,
        entry_zone_high=50100.0,
        stop_loss=49000.0,
        take_profit_1=52000.0,
        estimated_risk_reward=2.1,
        market_regime=MarketRegime.TRENDING_UP,
        session_status=SessionType.CONTINUOUS,
        data_quality_status=DataQualityStatus.CLEAN,
        confluence=_minimal_confluence(),
    )
    defaults.update(overrides)
    return SignalOutput(**defaults)


class TestSignalOutputBackwardCompat:
    def test_existing_fields_unchanged(self):
        sig = _minimal_signal_output()
        assert sig.action == SignalAction.BUY
        assert sig.confidence == 0.72
        assert sig.symbol == "BTCUSDT"
        assert sig.market_regime == MarketRegime.TRENDING_UP

    def test_new_fields_default_to_safe_values(self):
        sig = _minimal_signal_output()
        assert sig.trading_mode == TradingMode.SPOT
        assert sig.signal_type is None
        assert sig.direction == Direction.NO_TRADE
        assert sig.bias_scores is None
        assert sig.risk_level == RiskLevel.MODERATE
        assert sig.signal_quality == SignalQuality.MODERATE
        assert sig.validation_status == ValidationStatus.UNVALIDATED
        assert sig.setup_grade is None
        assert sig.trade_decision == TradeDecision.NO_TRADE
        assert sig.invalidation_level is None
        assert sig.failed_confirmations == []
        assert sig.risk_result is None
        assert sig.futures_risk is None
        assert sig.strategy_results == []
        assert sig.indicator_results == []

    def test_new_fields_set(self):
        bs = BiasScores(bullish_score=0.8, bearish_score=0.1)
        rr = RiskResult(entry=50000.0, stop_loss=49000.0, take_profit_1=52000.0)
        sig = _minimal_signal_output(
            trading_mode=TradingMode.SPOT,
            signal_type=SignalType.SPOT_BUY,
            direction=Direction.LONG,
            bias_scores=bs,
            risk_level=RiskLevel.LOW,
            signal_quality=SignalQuality.HIGH,
            validation_status=ValidationStatus.VALIDATED,
            setup_grade=SetupGrade.A_PLUS,
            trade_decision=TradeDecision.TAKE,
            invalidation_level=48500.0,
            risk_result=rr,
        )
        assert sig.signal_type == SignalType.SPOT_BUY
        assert sig.direction == Direction.LONG
        assert sig.bias_scores.dominant_bias == "bullish"
        assert sig.setup_grade == SetupGrade.A_PLUS
        assert sig.trade_decision == TradeDecision.TAKE
        assert sig.invalidation_level == 48500.0

    def test_futures_signal_with_futures_risk(self):
        fr = FuturesRiskResult(
            leverage=10.0,
            estimated_liquidation_price=45000.0,
            liquidation_buffer_percent=10.0,
            liquidation_risk=RiskLevel.MODERATE,
        )
        sig = _minimal_signal_output(
            trading_mode=TradingMode.FUTURES,
            signal_type=SignalType.FUTURES_LONG,
            direction=Direction.LONG,
            futures_risk=fr,
        )
        assert sig.trading_mode == TradingMode.FUTURES
        assert sig.futures_risk.leverage == 10.0
        assert sig.futures_risk.liquidation_risk == RiskLevel.MODERATE

    def test_to_display_still_works(self):
        sig = _minimal_signal_output()
        display = sig.to_display()
        assert "BUY" in display
        assert "BTCUSDT" in display

    def test_serialise_roundtrip(self):
        sig = _minimal_signal_output(
            signal_type=SignalType.SPOT_BUY,
            setup_grade=SetupGrade.A,
            trade_decision=TradeDecision.TAKE,
        )
        json_str = sig.model_dump_json()
        reloaded = SignalOutput.model_validate_json(json_str)
        assert reloaded.signal_type == SignalType.SPOT_BUY
        assert reloaded.setup_grade == SetupGrade.A
        assert reloaded.trade_decision == TradeDecision.TAKE
        assert reloaded.confidence == sig.confidence

    def test_no_trade_signal_defaults(self):
        sig = _minimal_signal_output(
            action=SignalAction.NO_TRADE,
            confidence=0.0,
        )
        assert sig.action == SignalAction.NO_TRADE
        # new fields still have safe defaults
        assert sig.trade_decision == TradeDecision.NO_TRADE


# ── Signal type / direction relationship sanity ───────────────────────────────

class TestSignalTypeDirectionMapping:
    """Verify the semantic relationship between SignalType and Direction."""

    @pytest.mark.parametrize("signal_type,expected_direction", [
        (SignalType.SPOT_BUY, Direction.LONG),
        (SignalType.FUTURES_LONG, Direction.LONG),
        (SignalType.SPOT_SELL_EXIT, Direction.EXIT),
        (SignalType.FUTURES_SHORT, Direction.SHORT),
        (SignalType.SPOT_NO_TRADE, Direction.NO_TRADE),
        (SignalType.FUTURES_NO_TRADE, Direction.NO_TRADE),
    ])
    def test_direction_matches_signal_type(self, signal_type, expected_direction):
        # Models don't enforce this automatically yet — test documents the contract
        # so future enforcement code can reference it.
        DIRECTION_MAP = {
            SignalType.SPOT_BUY: Direction.LONG,
            SignalType.SPOT_SELL_EXIT: Direction.EXIT,
            SignalType.SPOT_NO_TRADE: Direction.NO_TRADE,
            SignalType.FUTURES_LONG: Direction.LONG,
            SignalType.FUTURES_SHORT: Direction.SHORT,
            SignalType.FUTURES_NO_TRADE: Direction.NO_TRADE,
        }
        assert DIRECTION_MAP[signal_type] == expected_direction
