"""
Unit tests for the six paper-trading bot subclasses.

Tests match CURRENT bot logic (exclusive strategy assignment, aggressive mode).
"""
from __future__ import annotations

from libs.core.models.domain import (
    AssetClass, ConfluenceBreakdown, DataQualityStatus, MarketRegime,
    SessionType, SignalAction, SignalOutput, Timeframe, TrendDirection,
)
from libs.paper_trading.bots.mean_reversion import MeanReversionBot
from libs.paper_trading.bots.momentum import MomentumBot
from libs.paper_trading.bots.reversal import ReversalBot
from libs.paper_trading.bots.scalper import ScalperBot
from libs.paper_trading.bots.swing import SwingBot

_CONFLUENCE = ConfluenceBreakdown(
    pattern_score=0.7, structure_score=0.6, level_score=0.6,
    volume_score=0.5, regime_score=0.8, session_score=0.7,
    risk_score=0.6, data_quality_score=0.9, weighted_total=0.68,
)


def _sig(**overrides) -> SignalOutput:
    defaults = dict(
        symbol="BTC/USDT", asset_class=AssetClass.CRYPTO,
        strategy_name="ema_crossover", timeframe=Timeframe.ONE_HOUR,
        action=SignalAction.BUY, confidence=0.75,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=99.0, entry_zone_high=101.0,
        stop_loss=95.0, take_profit_1=110.0, take_profit_2=120.0,
        estimated_risk_reward=2.5, market_regime=MarketRegime.TRENDING_UP,
        session_status=SessionType.CONTINUOUS,
        data_quality_status=DataQualityStatus.CLEAN,
        confluence=_CONFLUENCE, patterns_detected=["hammer"],
    )
    defaults.update(overrides)
    return SignalOutput(**defaults)


# ── MomentumBot: ema_crossover, macd_crossover ──────────────────────────────

class TestMomentumBot:
    def test_takes_ema_crossover(self):
        bot = MomentumBot()
        assert bot.should_take_signal(_sig(strategy_name="ema_crossover")) is True

    def test_takes_macd_crossover(self):
        bot = MomentumBot()
        assert bot.should_take_signal(_sig(strategy_name="macd_crossover")) is True

    def test_skips_other_strategy(self):
        bot = MomentumBot()
        assert bot.should_take_signal(_sig(strategy_name="hammer_reversal")) is False

    def test_skips_support_breakdown(self):
        bot = MomentumBot()
        assert bot.should_take_signal(_sig(strategy_name="support_breakdown")) is False


# ── ReversalBot: hammer_reversal, shooting_star_reversal ─────────────────────

class TestReversalBot:
    def test_takes_hammer(self):
        bot = ReversalBot()
        assert bot.should_take_signal(_sig(strategy_name="hammer_reversal")) is True

    def test_takes_shooting_star(self):
        bot = ReversalBot()
        assert bot.should_take_signal(_sig(strategy_name="shooting_star_reversal")) is True

    def test_skips_ema(self):
        bot = ReversalBot()
        assert bot.should_take_signal(_sig(strategy_name="ema_crossover")) is False

    def test_skips_pullback(self):
        bot = ReversalBot()
        assert bot.should_take_signal(_sig(strategy_name="pullback_continuation")) is False


# ── MeanReversionBot: pullback + rsi_mean_reversion, RANGING regime ─────────

class TestMeanReversionBot:
    def test_takes_pullback_ranging(self):
        bot = MeanReversionBot()
        assert bot.should_take_signal(_sig(
            strategy_name="pullback_continuation",
            market_regime=MarketRegime.RANGING_LOW_VOL,
        )) is True

    def test_takes_rsi_ranging(self):
        bot = MeanReversionBot()
        assert bot.should_take_signal(_sig(
            strategy_name="rsi_mean_reversion",
            market_regime=MarketRegime.RANGING_HIGH_VOL,
        )) is True

    def test_skips_trending(self):
        bot = MeanReversionBot()
        assert bot.should_take_signal(_sig(
            strategy_name="pullback_continuation",
            market_regime=MarketRegime.TRENDING_UP,
        )) is False

    def test_skips_wrong_strategy(self):
        bot = MeanReversionBot()
        assert bot.should_take_signal(_sig(
            strategy_name="ema_crossover",
            market_regime=MarketRegime.RANGING_LOW_VOL,
        )) is False


# ── ScalperBot: any strategy, M1/M5/M15, R:R >= 1.0 ────────────────────────

class TestScalperBot:
    def test_takes_short_tf(self):
        bot = ScalperBot()
        assert bot.should_take_signal(_sig(
            timeframe=Timeframe.FIVE_MIN, estimated_risk_reward=1.5,
        )) is True

    def test_takes_m15(self):
        bot = ScalperBot()
        assert bot.should_take_signal(_sig(
            timeframe=Timeframe.FIFTEEN_MIN, estimated_risk_reward=1.2,
        )) is True

    def test_skips_long_tf(self):
        bot = ScalperBot()
        assert bot.should_take_signal(_sig(
            timeframe=Timeframe.ONE_HOUR, estimated_risk_reward=2.0,
        )) is False

    def test_skips_low_rr(self):
        bot = ScalperBot()
        assert bot.should_take_signal(_sig(
            timeframe=Timeframe.FIVE_MIN, estimated_risk_reward=0.8,
        )) is False

    def test_target_tp1(self):
        assert ScalperBot()._target_exit() == "tp1"


# ── SwingBot: resistance_breakout, support_breakdown ─────────────────────────

class TestSwingBot:
    def test_takes_resistance_breakout(self):
        bot = SwingBot()
        assert bot.should_take_signal(_sig(strategy_name="resistance_breakout")) is True

    def test_takes_support_breakdown(self):
        bot = SwingBot()
        assert bot.should_take_signal(_sig(strategy_name="support_breakdown")) is True

    def test_skips_ema(self):
        bot = SwingBot()
        assert bot.should_take_signal(_sig(strategy_name="ema_crossover")) is False

    def test_skips_hammer(self):
        bot = SwingBot()
        assert bot.should_take_signal(_sig(strategy_name="hammer_reversal")) is False

    def test_target_tp2(self):
        assert SwingBot()._target_exit() == "tp2"
