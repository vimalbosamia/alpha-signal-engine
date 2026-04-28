"""
Unit tests for libs/signals/output/emitter.py.

All tests are isolated — no environment-variable side-effects.
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch

import pytest

from libs.core.models.domain import (
    AssetClass,
    DataQualityReport,
    DataQualityStatus,
    MarketRegime,
    PatternBias,
    PatternResult,
    SessionState,
    SessionType,
    SignalAction,
    SignalCandidate,
    SignalOutput,
    Timeframe,
    TrendDirection,
)
from libs.signals.confluence.engine import ConfluenceEngine
from libs.signals.output.emitter import SignalEmitter


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_candidate(
    action: SignalAction = SignalAction.BUY,
    patterns: list[PatternResult] | None = None,
    quality_status: DataQualityStatus = DataQualityStatus.CLEAN,
    regime: MarketRegime = MarketRegime.TRENDING_UP,
    session_quality: float = 1.0,
    entry_low: float = 100.0,
    entry_high: float = 100.5,
    stop_loss: float = 98.0,
    tp1: float = 105.0,
) -> SignalCandidate:
    quality = DataQualityReport(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        status=quality_status,
        rows_checked=100,
    )
    session = SessionState(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        session_type=SessionType.REGULAR,
        is_tradable=True,
        quality_score=session_quality,
    )
    return SignalCandidate(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        strategy_name="test_strategy",
        proposed_action=action,
        timeframe=Timeframe.FIVE_MIN,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=entry_low,
        entry_zone_high=entry_high,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        pattern_results=patterns or [],
        regime=regime,
        quality=quality,
        session=session,
        raw_features={},
    )


def _score_candidate(candidate: SignalCandidate) -> "ConfluenceBreakdown":  # type: ignore[name-defined]
    """Score a candidate using the ConfluenceEngine."""
    from libs.signals.confluence.engine import ConfluenceEngine
    return ConfluenceEngine().score(candidate)


def _mock_settings(min_score: float = 0.65) -> MagicMock:
    """Return a MagicMock that mimics Settings with the given min_confluence_score."""
    settings = MagicMock()
    settings.signal.min_confluence_score = min_score
    return settings


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestSignalEmitterAction:
    """Tests for action determination."""

    def test_clean_candidate_with_passing_score_returns_proposed_action(
        self,
    ) -> None:
        """Clean candidate with score >= min_confluence → action matches proposed."""
        # Build a high-scoring candidate
        patterns = [
            PatternResult(
                pattern_name="bullish_engulfing",
                detected=True,
                confidence=0.9,
                bias=PatternBias.BULLISH,
            )
        ]
        candidate = make_candidate(action=SignalAction.BUY, patterns=patterns)

        # Construct a breakdown that passes (weighted_total above default 0.65)
        from libs.core.models.domain import ConfluenceBreakdown
        breakdown = ConfluenceBreakdown(
            pattern_score=1.0,
            structure_score=1.0,
            level_score=0.8,
            volume_score=0.8,
            regime_score=0.8,
            session_score=1.0,
            risk_score=0.9,
            data_quality_score=1.0,
            weighted_total=0.92,
            weights={"pattern": 0.2},
            factor_notes={},
            blocked_reasons=[],
            warning_tags=[],
        )

        emitter = SignalEmitter()
        with patch("libs.signals.output.emitter.get_settings", return_value=_mock_settings(0.65)):
            output = emitter.emit(candidate, breakdown)
        assert output.action == SignalAction.BUY

    def test_blocked_quality_final_action_is_no_trade(self) -> None:
        """BLOCKED data quality → final action = NO_TRADE."""
        candidate = make_candidate(quality_status=DataQualityStatus.BLOCKED)
        breakdown = _score_candidate(candidate)

        emitter = SignalEmitter()
        with patch("libs.signals.output.emitter.get_settings", return_value=_mock_settings(0.65)):
            output = emitter.emit(candidate, breakdown)
        assert output.action == SignalAction.NO_TRADE

    def test_low_score_below_min_confluence_gives_no_trade(self) -> None:
        """weighted_total < min_confluence_score → NO_TRADE."""
        from libs.core.models.domain import ConfluenceBreakdown
        candidate = make_candidate(action=SignalAction.BUY)
        breakdown = ConfluenceBreakdown(
            pattern_score=0.1,
            structure_score=0.1,
            level_score=0.1,
            volume_score=0.1,
            regime_score=0.1,
            session_score=0.1,
            risk_score=0.1,
            data_quality_score=1.0,
            weighted_total=0.10,   # well below default min of 0.65
            weights={},
            factor_notes={},
            blocked_reasons=[],
            warning_tags=[],
        )

        emitter = SignalEmitter()
        with patch("libs.signals.output.emitter.get_settings", return_value=_mock_settings(0.65)):
            output = emitter.emit(candidate, breakdown)
        assert output.action == SignalAction.NO_TRADE


class TestSignalEmitterRR:
    """Tests for R:R calculation."""

    def test_rr_calculated_correctly_for_buy(self) -> None:
        """R:R for BUY = (tp1 - entry_mid) / (entry_mid - stop)."""
        # entry_mid = (100 + 101) / 2 = 100.5
        # rr = (110 - 100.5) / (100.5 - 95) = 9.5 / 5.5 ≈ 1.727
        candidate = make_candidate(
            action=SignalAction.BUY,
            entry_low=100.0,
            entry_high=101.0,
            stop_loss=95.0,
            tp1=110.0,
        )
        emitter = SignalEmitter()
        rr = emitter._calc_rr(candidate)
        expected = (110.0 - 100.5) / (100.5 - 95.0)
        assert rr == pytest.approx(expected, rel=1e-5)

    def test_rr_calculated_correctly_for_sell(self) -> None:
        """R:R for SELL = (entry_mid - tp1) / (stop - entry_mid)."""
        # entry_mid = (100 + 101) / 2 = 100.5
        # rr = (100.5 - 90) / (110 - 100.5) = 10.5 / 9.5 ≈ 1.105
        candidate = make_candidate(
            action=SignalAction.SELL,
            entry_low=100.0,
            entry_high=101.0,
            stop_loss=110.0,
            tp1=90.0,
        )
        emitter = SignalEmitter()
        rr = emitter._calc_rr(candidate)
        expected = (100.5 - 90.0) / (110.0 - 100.5)
        assert rr == pytest.approx(expected, rel=1e-5)


class TestSignalEmitterOutput:
    """Tests for SignalOutput completeness."""

    def test_signal_output_has_all_required_fields_populated(self) -> None:
        """SignalOutput has all required fields populated (no None in required fields)."""
        patterns = [
            PatternResult(
                pattern_name="hammer",
                detected=True,
                confidence=0.75,
                bias=PatternBias.BULLISH,
            )
        ]
        candidate = make_candidate(action=SignalAction.BUY, patterns=patterns)
        breakdown = _score_candidate(candidate)

        emitter = SignalEmitter(agent_mode="paper", data_provider="alpaca")
        with patch("libs.signals.output.emitter.get_settings", return_value=_mock_settings(0.65)):
            output = emitter.emit(candidate, breakdown)

        assert isinstance(output, SignalOutput)
        assert output.symbol == "AAPL"
        assert output.asset_class == AssetClass.STOCK
        assert output.strategy_name == "test_strategy"
        assert output.timeframe == Timeframe.FIVE_MIN
        assert output.action in list(SignalAction)
        assert 0.0 <= output.confidence <= 1.0
        assert output.higher_tf_bias == TrendDirection.UPTREND
        assert output.entry_zone_low == 100.0
        assert output.entry_zone_high == 100.5
        assert output.stop_loss == 98.0
        assert output.take_profit_1 == 105.0
        assert output.market_regime is not None
        assert output.session_status is not None
        assert output.data_quality_status is not None
        assert output.confluence is not None
        assert isinstance(output.patterns_detected, list)
        assert output.explanation != ""
        assert output.agent_mode == "paper"
        assert output.data_provider == "alpaca"

    def test_detected_pattern_names_are_included_in_output(self) -> None:
        """Detected pattern names appear in SignalOutput.patterns_detected."""
        patterns = [
            PatternResult(
                pattern_name="morning_star",
                detected=True,
                confidence=0.8,
                bias=PatternBias.BULLISH,
            ),
            PatternResult(
                pattern_name="doji",
                detected=False,
                confidence=0.5,
                bias=PatternBias.NEUTRAL,
            ),
        ]
        candidate = make_candidate(patterns=patterns)
        breakdown = _score_candidate(candidate)

        emitter = SignalEmitter()
        with patch("libs.signals.output.emitter.get_settings", return_value=_mock_settings(0.65)):
            output = emitter.emit(candidate, breakdown)

        assert "morning_star" in output.patterns_detected
        assert "doji" not in output.patterns_detected
