"""
Unit tests for libs/risk/engine.py.

All tests are isolated — RiskSettings is constructed directly so no
environment-variable side-effects can influence results.
"""
from __future__ import annotations

import math
from datetime import datetime
from uuid import uuid4

import pytest

from libs.core.config.settings import RiskSettings
from libs.core.models.domain import (
    AssetClass,
    MarketRegime,
    SignalAction,
    SignalCandidate,
    Timeframe,
    TrendDirection,
)
from libs.risk.engine import CostModel, RiskAssessment, RiskEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_candidate(
    action: SignalAction = SignalAction.BUY,
    asset_class: AssetClass = AssetClass.STOCK,
    entry_low: float = 100.0,
    entry_high: float = 100.5,
    stop_loss: float = 98.0,
    tp1: float = 105.0,
    regime: MarketRegime = MarketRegime.TRENDING_UP,
) -> SignalCandidate:
    return SignalCandidate(
        symbol="AAPL",
        asset_class=asset_class,
        strategy_name="test",
        proposed_action=action,
        timeframe=Timeframe.FIVE_MIN,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=entry_low,
        entry_zone_high=entry_high,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        regime=regime,
        raw_features={},
    )


def default_engine() -> RiskEngine:
    return RiskEngine(settings=RiskSettings())


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGoodBuySetup:
    """Test 1: A well-formed BUY with R:R ~2.5 and ~2% stop should pass."""

    def test_passes(self) -> None:
        # entry_mid = 100.25; stop = 98.0 → stop_dist ≈ 2.24%
        # tp1 = 105.0 → rr = (105 - 100.25) / (100.25 - 98) ≈ 2.11
        candidate = make_candidate(
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=98.0,
            tp1=106.0,  # rr = 5.75 / 2.25 ≈ 2.56
        )
        result = default_engine().assess(candidate)

        assert result.passed is True
        assert result.blocked_reasons == []
        assert result.reward_risk > 2.0
        assert result.stop_distance_pct > 0

    def test_risk_per_trade_reflects_settings(self) -> None:
        settings = RiskSettings(max_risk_per_signal_pct=1.5)
        engine = RiskEngine(settings=settings)
        candidate = make_candidate(tp1=106.0)
        result = engine.assess(candidate)
        assert result.risk_per_trade_pct == pytest.approx(1.5)


class TestLowRewardRisk:
    """Test 2: R:R below minimum should block the signal."""

    def test_blocked_when_rr_too_low(self) -> None:
        # entry_mid = 100.25; stop = 99.0; tp1 = 101.0
        # rr = (101 - 100.25) / (100.25 - 99) = 0.75 / 1.25 = 0.6
        candidate = make_candidate(
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=99.0,
            tp1=101.0,
        )
        result = default_engine().assess(candidate)

        assert result.passed is False
        assert any("Reward:Risk" in r or "R:R" in r or "ratio" in r.lower()
                   for r in result.blocked_reasons)

    def test_score_is_zero_when_blocked(self) -> None:
        candidate = make_candidate(
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=99.0,
            tp1=101.0,
        )
        result = default_engine().assess(candidate)
        assert result.score == 0.0


class TestStopTooTight:
    """Test 3: Stop distance < 0.10% for stock → blocked."""

    def test_blocked_when_stop_too_tight(self) -> None:
        # entry_mid = 100.25; stop = 100.20 → dist = 0.05%
        candidate = make_candidate(
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=100.20,
            tp1=110.0,
        )
        result = default_engine().assess(candidate)

        assert result.passed is False
        assert any("below minimum" in r for r in result.blocked_reasons)


class TestStopTooWide:
    """Test 4: Stop distance > 15% for stock → blocked."""

    def test_blocked_when_stop_too_wide(self) -> None:
        # entry_mid = 100.25; stop = 80.0 → dist ≈ 20.2%
        candidate = make_candidate(
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=80.0,
            tp1=140.0,
        )
        result = default_engine().assess(candidate)

        assert result.passed is False
        assert any("exceeds maximum" in r for r in result.blocked_reasons)


class TestSellCandidate:
    """Test 5: R:R is calculated correctly for SELL signals."""

    def test_rr_positive_for_sell(self) -> None:
        # For SELL: entry_mid=100.25; stop=103.0; tp1=95.0
        # rr = (100.25 - 95.0) / (103.0 - 100.25) = 5.25 / 2.75 ≈ 1.91
        candidate = make_candidate(
            action=SignalAction.SELL,
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=103.0,
            tp1=95.0,
            regime=MarketRegime.TRENDING_DOWN,
        )
        engine = default_engine()
        entry = engine._entry_mid(candidate)
        rr = engine._calc_reward_risk(candidate, entry)

        assert rr == pytest.approx((100.25 - 95.0) / (103.0 - 100.25), rel=1e-4)
        assert rr > 0

    def test_sell_with_good_rr_can_pass(self) -> None:
        # stop=104.0; tp1=93.0 → rr = 7.25 / 3.75 ≈ 1.93 — still under 2.0
        # Use tp1 = 90.0 → rr = 10.25 / 3.75 ≈ 2.73
        candidate = make_candidate(
            action=SignalAction.SELL,
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=103.0,
            tp1=90.0,
            regime=MarketRegime.TRENDING_DOWN,
        )
        result = default_engine().assess(candidate)
        assert result.reward_risk > 2.0


class TestNoTrade:
    """Test 6: NO_TRADE should always return reward_risk=0.0 and fail."""

    def test_no_trade_reward_risk_is_zero(self) -> None:
        candidate = make_candidate(action=SignalAction.NO_TRADE)
        engine = default_engine()
        entry = engine._entry_mid(candidate)
        assert engine._calc_reward_risk(candidate, entry) == 0.0

    def test_no_trade_is_blocked(self) -> None:
        candidate = make_candidate(action=SignalAction.NO_TRADE)
        result = default_engine().assess(candidate)
        assert result.passed is False
        assert result.reward_risk == pytest.approx(0.0)

    def test_no_trade_score_is_zero(self) -> None:
        candidate = make_candidate(action=SignalAction.NO_TRADE)
        result = default_engine().assess(candidate)
        assert result.score == 0.0


class TestCryptoAsset:
    """Test 7: Crypto uses wider stop limits and higher friction costs."""

    def test_crypto_allows_wider_stop(self) -> None:
        # 10% stop — invalid for stock, valid for crypto
        candidate = make_candidate(
            asset_class=AssetClass.CRYPTO,
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=90.0,   # ~10% stop
            tp1=130.0,        # rr ≈ 29.75 / 10.25 ≈ 2.9
        )
        result = default_engine().assess(candidate)
        # Should not be blocked due to stop distance
        assert not any("below minimum" in r or "exceeds maximum" in r
                       for r in result.blocked_reasons)

    def test_crypto_cost_model_has_higher_friction(self) -> None:
        engine = default_engine()
        stock_cost = engine._cost_model(AssetClass.STOCK)
        crypto_cost = engine._cost_model(AssetClass.CRYPTO)
        assert crypto_cost.total_friction_pct > stock_cost.total_friction_pct

    def test_crypto_cost_model_values(self) -> None:
        engine = default_engine()
        cost = engine._cost_model(AssetClass.CRYPTO)
        assert cost.slippage_pct == pytest.approx(RiskEngine.COST_CRYPTO_SLIPPAGE)
        assert cost.spread_pct == pytest.approx(RiskEngine.COST_CRYPTO_SPREAD)
        assert cost.commission_pct == pytest.approx(RiskEngine.COST_CRYPTO_COMMISSION)
        assert cost.total_friction_pct == pytest.approx(
            RiskEngine.COST_CRYPTO_SLIPPAGE
            + RiskEngine.COST_CRYPTO_SPREAD
            + RiskEngine.COST_CRYPTO_COMMISSION
        )

    def test_crypto_tight_stop_is_blocked(self) -> None:
        # 0.10% stop — valid for stock but below crypto minimum (0.20%)
        candidate = make_candidate(
            asset_class=AssetClass.CRYPTO,
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=100.15,  # ≈ 0.10% stop
            tp1=110.0,
        )
        result = default_engine().assess(candidate)
        assert result.passed is False
        assert any("below minimum" in r for r in result.blocked_reasons)


class TestHighVolRegime:
    """Test 8: High-volatility regime reduces score compared to trending."""

    def test_lower_score_in_high_vol(self) -> None:
        good_candidate = make_candidate(
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=97.0,
            tp1=110.0,
            regime=MarketRegime.TRENDING_UP,
        )
        noisy_candidate = make_candidate(
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=97.0,
            tp1=110.0,
            regime=MarketRegime.RANGING_HIGH_VOL,
        )
        engine = default_engine()
        good_result = engine.assess(good_candidate)
        noisy_result = engine.assess(noisy_candidate)

        assert noisy_result.score < good_result.score

    def test_climactic_regime_also_penalised(self) -> None:
        trending = make_candidate(
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=97.0,
            tp1=110.0,
            regime=MarketRegime.TRENDING_UP,
        )
        climactic = make_candidate(
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=97.0,
            tp1=110.0,
            regime=MarketRegime.CLIMACTIC,
        )
        engine = default_engine()
        assert engine.assess(climactic).score < engine.assess(trending).score


class TestScoreBounds:
    """Test 9: score is always clamped to [0.0, 1.0]."""

    @pytest.mark.parametrize("action,stop_loss,tp1,regime,asset_class", [
        # High-scoring scenario
        (SignalAction.BUY, 97.0, 115.0, MarketRegime.TRENDING_UP, AssetClass.STOCK),
        # Blocked → score should be 0.0
        (SignalAction.BUY, 100.20, 115.0, MarketRegime.TRENDING_UP, AssetClass.STOCK),
        # High vol + crypto accumulation of penalties
        (SignalAction.BUY, 90.0, 115.0, MarketRegime.RANGING_HIGH_VOL, AssetClass.CRYPTO),
        # NO_TRADE
        (SignalAction.NO_TRADE, 98.0, 105.0, MarketRegime.UNKNOWN, AssetClass.STOCK),
    ])
    def test_score_in_range(
        self,
        action: SignalAction,
        stop_loss: float,
        tp1: float,
        regime: MarketRegime,
        asset_class: AssetClass,
    ) -> None:
        candidate = make_candidate(
            action=action,
            stop_loss=stop_loss,
            tp1=tp1,
            regime=regime,
            asset_class=asset_class,
        )
        result = default_engine().assess(candidate)
        assert 0.0 <= result.score <= 1.0


class TestEdgeInputs:
    """Test 10: assess() never raises on pathological inputs."""

    def test_zero_entry_price(self) -> None:
        candidate = make_candidate(entry_low=0.0, entry_high=0.0, stop_loss=0.0, tp1=0.0)
        # Should not raise
        result = default_engine().assess(candidate)
        assert isinstance(result, RiskAssessment)

    def test_equal_entry_and_stop(self) -> None:
        # entry_mid = stop_loss → denominator collapses to 1e-9
        candidate = make_candidate(
            entry_low=100.0,
            entry_high=100.0,
            stop_loss=100.0,
            tp1=110.0,
        )
        result = default_engine().assess(candidate)
        assert isinstance(result, RiskAssessment)
        assert not math.isnan(result.reward_risk)
        assert not math.isinf(result.reward_risk)

    def test_negative_rr_is_handled(self) -> None:
        # tp1 below stop for a BUY — R:R will be negative → should block on R:R check
        candidate = make_candidate(
            entry_low=100.0,
            entry_high=100.5,
            stop_loss=98.0,
            tp1=97.0,   # below stop
        )
        result = default_engine().assess(candidate)
        assert result.passed is False

    def test_no_exception_on_very_large_prices(self) -> None:
        candidate = make_candidate(
            entry_low=1_000_000.0,
            entry_high=1_000_000.5,
            stop_loss=999_000.0,
            tp1=1_010_000.0,
        )
        result = default_engine().assess(candidate)
        assert isinstance(result, RiskAssessment)


class TestCostModel:
    """Additional tests for CostModel frozen dataclass."""

    def test_stock_cost_model_values(self) -> None:
        engine = default_engine()
        cost = engine._cost_model(AssetClass.STOCK)
        assert cost.slippage_pct == pytest.approx(RiskEngine.COST_STOCK_SLIPPAGE)
        assert cost.spread_pct == pytest.approx(RiskEngine.COST_STOCK_SPREAD)
        assert cost.commission_pct == pytest.approx(RiskEngine.COST_STOCK_COMMISSION)
        assert cost.total_friction_pct == pytest.approx(
            RiskEngine.COST_STOCK_SLIPPAGE
            + RiskEngine.COST_STOCK_SPREAD
            + RiskEngine.COST_STOCK_COMMISSION
        )

    def test_cost_model_is_immutable(self) -> None:
        cost = CostModel(slippage_pct=0.05, spread_pct=0.02, commission_pct=0.0)
        with pytest.raises((AttributeError, TypeError)):
            cost.slippage_pct = 0.99  # type: ignore[misc]


class TestRewardRiskCalculation:
    """Verify _calc_reward_risk arithmetic for BUY and SELL."""

    def test_buy_rr_arithmetic(self) -> None:
        candidate = make_candidate(
            action=SignalAction.BUY,
            entry_low=100.0,
            entry_high=101.0,  # mid = 100.5
            stop_loss=99.0,    # dist = 1.5
            tp1=103.5,         # profit = 3.0 → rr = 2.0
        )
        engine = default_engine()
        entry = engine._entry_mid(candidate)
        assert entry == pytest.approx(100.5)
        rr = engine._calc_reward_risk(candidate, entry)
        assert rr == pytest.approx(2.0, rel=1e-4)

    def test_sell_rr_arithmetic(self) -> None:
        candidate = make_candidate(
            action=SignalAction.SELL,
            entry_low=100.0,
            entry_high=101.0,  # mid = 100.5
            stop_loss=102.0,   # dist above = 1.5
            tp1=97.5,          # profit = 3.0 → rr = 2.0
        )
        engine = default_engine()
        entry = engine._entry_mid(candidate)
        rr = engine._calc_reward_risk(candidate, entry)
        assert rr == pytest.approx(2.0, rel=1e-4)
