"""
Unit tests for libs/signals/confluence/engine.py.

All tests are isolated — no environment-variable side-effects.
"""
from __future__ import annotations

import pytest

from libs.analysis.structure.engine import MarketStructure
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
    Timeframe,
    TrendDirection,
)
from libs.risk.engine import RiskAssessment
from libs.signals.confluence.engine import ConfluenceEngine, DEFAULT_WEIGHTS


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_candidate(
    action: SignalAction = SignalAction.BUY,
    patterns: list[PatternResult] | None = None,
    quality_status: DataQualityStatus = DataQualityStatus.CLEAN,
    regime: MarketRegime = MarketRegime.TRENDING_UP,
    session_quality: float = 1.0,
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
        strategy_name="test",
        proposed_action=action,
        timeframe=Timeframe.FIVE_MIN,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=100.0,
        entry_zone_high=100.5,
        stop_loss=98.0,
        take_profit_1=105.0,
        pattern_results=patterns or [],
        regime=regime,
        quality=quality,
        session=session,
        raw_features={},
    )


def make_uptrend_structure(
    strength: float = 0.8,
    bos: bool = False,
) -> MarketStructure:
    return MarketStructure(
        trend=TrendDirection.UPTREND,
        points=[],
        trend_strength=strength,
        break_of_structure=bos,
        swing_high=105.0,
        swing_low=99.0,
    )


def make_risk_passed() -> RiskAssessment:
    return RiskAssessment(
        passed=True,
        reward_risk=3.0,
        stop_distance_pct=2.0,
        risk_per_trade_pct=1.0,
        blocked_reasons=[],
        warnings=[],
        score=0.9,
    )


def make_risk_failed() -> RiskAssessment:
    return RiskAssessment(
        passed=False,
        reward_risk=0.5,
        stop_distance_pct=2.0,
        risk_per_trade_pct=1.0,
        blocked_reasons=["R:R too low"],
        warnings=[],
        score=0.0,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestConfluenceEngineBasic:
    """Basic smoke tests."""

    def test_no_patterns_all_none_context_returns_valid_breakdown(self) -> None:
        """No patterns + all None context → weighted_total in [0, 1], no exception."""
        engine = ConfluenceEngine()
        candidate = make_candidate(patterns=[])
        breakdown = engine.score(candidate)

        assert 0.0 <= breakdown.weighted_total <= 1.0
        assert isinstance(breakdown.factor_notes, dict)
        assert isinstance(breakdown.blocked_reasons, list)
        assert isinstance(breakdown.warning_tags, list)

    def test_all_none_inputs_returns_confluence_breakdown_no_crash(self) -> None:
        """All None inputs → returns ConfluenceBreakdown without raising."""
        engine = ConfluenceEngine()
        candidate = SignalCandidate(
            symbol="AAPL",
            asset_class=AssetClass.STOCK,
            strategy_name="test",
            proposed_action=SignalAction.BUY,
            timeframe=Timeframe.FIVE_MIN,
            higher_tf_bias=TrendDirection.UNKNOWN,
            entry_zone_low=100.0,
            entry_zone_high=100.5,
            stop_loss=98.0,
            take_profit_1=105.0,
            raw_features={},
        )
        breakdown = engine.score(
            candidate,
            structure=None,
            levels=None,
            volume=None,
            regime=None,
            risk=None,
        )
        assert 0.0 <= breakdown.weighted_total <= 1.0

    def test_weights_normalise_to_sum_one_on_construction(self) -> None:
        """Custom weights are normalised to sum to 1.0."""
        raw = {"pattern": 2.0, "structure": 2.0, "level": 1.0, "volume": 1.0,
               "regime": 1.0, "session": 1.0, "data_quality": 1.0, "risk": 1.0}
        engine = ConfluenceEngine(weights=raw)
        total = sum(engine._weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_default_weights_sum_to_one(self) -> None:
        """DEFAULT_WEIGHTS already sums to 1.0."""
        assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


class TestDataQualityScoring:
    """Data quality blocking behaviour."""

    def test_blocked_quality_sets_blocked_reasons_and_zero_total(self) -> None:
        """BLOCKED data quality → blocked_reasons non-empty, weighted_total=0.0."""
        engine = ConfluenceEngine()
        candidate = make_candidate(quality_status=DataQualityStatus.BLOCKED)
        breakdown = engine.score(candidate)

        assert breakdown.weighted_total == 0.0
        assert len(breakdown.blocked_reasons) > 0

    def test_warning_quality_adds_warning_tag(self) -> None:
        """WARNING data quality → warning_tags contains 'data_quality_warning'."""
        engine = ConfluenceEngine()
        candidate = make_candidate(quality_status=DataQualityStatus.WARNING)
        breakdown = engine.score(candidate)

        assert "data_quality_warning" in breakdown.warning_tags
        assert breakdown.weighted_total > 0.0   # not blocked

    def test_clean_quality_gives_full_dq_score(self) -> None:
        """CLEAN data quality → data_quality_score = 1.0."""
        engine = ConfluenceEngine()
        candidate = make_candidate(quality_status=DataQualityStatus.CLEAN)
        breakdown = engine.score(candidate)

        assert breakdown.data_quality_score == 1.0


class TestStructureScoring:
    """Structure alignment scoring."""

    def test_high_confidence_buy_pattern_uptrend_gives_high_score(self) -> None:
        """High-confidence BUY pattern + uptrend structure → high weighted total."""
        engine = ConfluenceEngine()
        patterns = [
            PatternResult(
                pattern_name="bullish_engulfing",
                detected=True,
                confidence=0.9,
                bias=PatternBias.BULLISH,
            )
        ]
        candidate = make_candidate(action=SignalAction.BUY, patterns=patterns)
        structure = make_uptrend_structure(strength=0.8)
        breakdown = engine.score(candidate, structure=structure)

        assert breakdown.pattern_score > 0.8   # 0.9 + 0.1 bonus = 1.0 (clamped)
        assert breakdown.structure_score == 1.0
        assert breakdown.weighted_total > 0.6

    def test_counter_trend_sell_in_strong_uptrend_gives_low_structure_score(
        self,
    ) -> None:
        """Counter-trend SELL in uptrend → structure_score == 0.2."""
        engine = ConfluenceEngine()
        candidate = make_candidate(action=SignalAction.SELL)
        structure = make_uptrend_structure(strength=0.5)
        breakdown = engine.score(candidate, structure=structure)

        assert breakdown.structure_score == pytest.approx(0.2, abs=0.01)

    def test_bos_against_buy_in_uptrend_penalises_structure_score(self) -> None:
        """Break of structure against BUY direction deducts from structure score."""
        engine = ConfluenceEngine()
        candidate = make_candidate(action=SignalAction.BUY)
        structure_with_bos = make_uptrend_structure(strength=0.5, bos=True)
        structure_no_bos = make_uptrend_structure(strength=0.5, bos=False)

        bd_bos = engine.score(candidate, structure=structure_with_bos)
        bd_no_bos = engine.score(candidate, structure=structure_no_bos)

        assert bd_bos.structure_score < bd_no_bos.structure_score


class TestPatternScoring:
    """Pattern confidence and bonus logic."""

    def test_pattern_bonus_applied_for_high_confidence_pattern(self) -> None:
        """confidence >= 0.8 triggers +0.1 bonus on pattern score."""
        engine = ConfluenceEngine()
        low_confidence_pattern = [
            PatternResult(
                pattern_name="doji",
                detected=True,
                confidence=0.70,
                bias=PatternBias.NEUTRAL,
            )
        ]
        high_confidence_pattern = [
            PatternResult(
                pattern_name="bullish_engulfing",
                detected=True,
                confidence=0.85,
                bias=PatternBias.BULLISH,
            )
        ]
        bd_low = engine.score(make_candidate(patterns=low_confidence_pattern))
        bd_high = engine.score(make_candidate(patterns=high_confidence_pattern))

        # High-confidence pattern should have a higher pattern_score
        assert bd_high.pattern_score > bd_low.pattern_score

    def test_undetected_patterns_are_ignored(self) -> None:
        """Patterns with detected=False are not included in scoring."""
        engine = ConfluenceEngine()
        patterns = [
            PatternResult(
                pattern_name="hammer",
                detected=False,
                confidence=0.95,
                bias=PatternBias.BULLISH,
            )
        ]
        breakdown = engine.score(make_candidate(patterns=patterns))
        assert breakdown.pattern_score == 0.0


class TestRiskScoring:
    """Risk assessment integration."""

    def test_risk_passed_false_gives_zero_risk_score_and_blocked(self) -> None:
        """risk.passed=False → risk_score=0.0 and candidate is blocked."""
        engine = ConfluenceEngine()
        candidate = make_candidate()
        risk = make_risk_failed()
        breakdown = engine.score(candidate, risk=risk)

        assert breakdown.risk_score == 0.0
        assert len(breakdown.blocked_reasons) > 0
        assert breakdown.weighted_total == 0.0

    def test_risk_passed_true_uses_risk_score(self) -> None:
        """risk.passed=True → risk_score matches risk.score."""
        engine = ConfluenceEngine()
        candidate = make_candidate()
        risk = make_risk_passed()
        breakdown = engine.score(candidate, risk=risk)

        assert breakdown.risk_score == pytest.approx(0.9)
