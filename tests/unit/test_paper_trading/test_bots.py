"""
Unit tests for the six paper-trading bot subclasses.

Covers: MomentumBot, ReversalBot, MeanReversionBot, ScalperBot, SwingBot.
AdaptiveBot is tested in test_adaptive.py.
"""
from __future__ import annotations

import pytest

from libs.core.models.domain import (
    AssetClass,
    ConfluenceBreakdown,
    DataQualityStatus,
    MarketRegime,
    SessionType,
    SignalAction,
    SignalOutput,
    Timeframe,
    TrendDirection,
)
from libs.paper_trading.bots.mean_reversion import MeanReversionBot
from libs.paper_trading.bots.momentum import MomentumBot
from libs.paper_trading.bots.reversal import ReversalBot
from libs.paper_trading.bots.scalper import ScalperBot
from libs.paper_trading.bots.swing import SwingBot


# ── Signal factory ─────────────────────────────────────────────────────────────

_CONFLUENCE = ConfluenceBreakdown(
    pattern_score=0.7,
    structure_score=0.6,
    level_score=0.6,
    volume_score=0.5,
    regime_score=0.8,
    session_score=0.7,
    risk_score=0.6,
    data_quality_score=0.9,
    weighted_total=0.68,
)


def _sig(**overrides) -> SignalOutput:
    """Build a SignalOutput with sensible defaults; override any field."""
    defaults = dict(
        symbol="BTC/USDT",
        asset_class=AssetClass.CRYPTO,
        strategy_name="ema_crossover",
        timeframe=Timeframe.ONE_HOUR,
        action=SignalAction.BUY,
        confidence=0.75,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=99.0,
        entry_zone_high=101.0,
        stop_loss=95.0,
        take_profit_1=110.0,
        take_profit_2=120.0,
        estimated_risk_reward=2.5,
        market_regime=MarketRegime.TRENDING_UP,
        session_status=SessionType.CONTINUOUS,
        data_quality_status=DataQualityStatus.CLEAN,
        confluence=_CONFLUENCE,
        patterns_detected=["hammer"],
    )
    defaults.update(overrides)
    return SignalOutput(**defaults)


# ── MomentumBot ────────────────────────────────────────────────────────────────

class TestMomentumBot:
    def test_momentum_takes_trending_ema(self):
        """Trending regime + ema_crossover strategy + H1 → accept."""
        bot = MomentumBot()
        signal = _sig(
            strategy_name="ema_crossover",
            market_regime=MarketRegime.TRENDING_UP,
            timeframe=Timeframe.ONE_HOUR,
            confidence=0.65,
        )
        assert bot.should_take_signal(signal) is True

    def test_momentum_takes_breakout_regime(self):
        """BREAKOUT regime with matching strategy → accept."""
        bot = MomentumBot()
        signal = _sig(
            strategy_name="resistance_breakout",
            market_regime=MarketRegime.BREAKOUT,
            timeframe=Timeframe.FIFTEEN_MIN,
        )
        assert bot.should_take_signal(signal) is True

    def test_momentum_takes_high_confidence_regardless_of_strategy(self):
        """Unknown strategy but confidence >= 0.7 in a trending regime → accept."""
        bot = MomentumBot()
        signal = _sig(
            strategy_name="unknown_strategy",
            market_regime=MarketRegime.TRENDING_DOWN,
            timeframe=Timeframe.THIRTY_MIN,
            confidence=0.72,
        )
        assert bot.should_take_signal(signal) is True

    def test_momentum_skips_ranging(self):
        """RANGING_LOW_VOL regime → reject regardless of strategy."""
        bot = MomentumBot()
        signal = _sig(
            strategy_name="ema_crossover",
            market_regime=MarketRegime.RANGING_LOW_VOL,
            timeframe=Timeframe.ONE_HOUR,
        )
        assert bot.should_take_signal(signal) is False

    def test_momentum_skips_wrong_timeframe(self):
        """H4 timeframe → reject (not in M15/M30/H1)."""
        bot = MomentumBot()
        signal = _sig(
            strategy_name="ema_crossover",
            market_regime=MarketRegime.TRENDING_UP,
            timeframe=Timeframe.FOUR_HOUR,
        )
        assert bot.should_take_signal(signal) is False

    def test_momentum_skips_low_confidence_unknown_strategy(self):
        """Unknown strategy + confidence < 0.7 → reject."""
        bot = MomentumBot()
        signal = _sig(
            strategy_name="other_strategy",
            market_regime=MarketRegime.TRENDING_UP,
            timeframe=Timeframe.ONE_HOUR,
            confidence=0.65,
        )
        assert bot.should_take_signal(signal) is False


# ── ReversalBot ───────────────────────────────────────────────────────────────

class TestReversalBot:
    def test_reversal_takes_hammer(self):
        """hammer_reversal strategy + hammer pattern + non-choppy regime → accept."""
        bot = ReversalBot()
        signal = _sig(
            strategy_name="hammer_reversal",
            patterns_detected=["hammer"],
            market_regime=MarketRegime.TRENDING_DOWN,
            timeframe=Timeframe.ONE_HOUR,
        )
        assert bot.should_take_signal(signal) is True

    def test_reversal_takes_engulfing_pattern(self):
        """shooting_star_reversal + bullish_engulfing pattern → accept."""
        bot = ReversalBot()
        signal = _sig(
            strategy_name="shooting_star_reversal",
            patterns_detected=["bullish_engulfing", "volume_spike"],
            market_regime=MarketRegime.RANGING_LOW_VOL,
            timeframe=Timeframe.FIFTEEN_MIN,
        )
        assert bot.should_take_signal(signal) is True

    def test_reversal_skips_choppy(self):
        """RANGING_HIGH_VOL regime → reject (too choppy for reversals)."""
        bot = ReversalBot()
        signal = _sig(
            strategy_name="hammer_reversal",
            patterns_detected=["hammer"],
            market_regime=MarketRegime.RANGING_HIGH_VOL,
            timeframe=Timeframe.ONE_HOUR,
        )
        assert bot.should_take_signal(signal) is False

    def test_reversal_skips_wrong_timeframe(self):
        """M5 timeframe → reject (only M15/H1 allowed)."""
        bot = ReversalBot()
        signal = _sig(
            strategy_name="hammer_reversal",
            patterns_detected=["hammer"],
            market_regime=MarketRegime.TRENDING_DOWN,
            timeframe=Timeframe.FIVE_MIN,
        )
        assert bot.should_take_signal(signal) is False

    def test_reversal_skips_no_reversal_pattern(self):
        """No reversal pattern in patterns_detected → reject."""
        bot = ReversalBot()
        signal = _sig(
            strategy_name="hammer_reversal",
            patterns_detected=["volume_spike", "ema_cross"],
            market_regime=MarketRegime.TRENDING_DOWN,
            timeframe=Timeframe.ONE_HOUR,
        )
        assert bot.should_take_signal(signal) is False

    def test_reversal_skips_wrong_strategy(self):
        """Correct pattern but wrong strategy → reject."""
        bot = ReversalBot()
        signal = _sig(
            strategy_name="ema_crossover",
            patterns_detected=["pin_bar"],
            market_regime=MarketRegime.TRENDING_DOWN,
            timeframe=Timeframe.ONE_HOUR,
        )
        assert bot.should_take_signal(signal) is False


# ── MeanReversionBot ──────────────────────────────────────────────────────────

class TestMeanReversionBot:
    def test_meanrev_takes_ranging(self):
        """RANGING_LOW_VOL + pullback_continuation + M15 → accept."""
        bot = MeanReversionBot()
        signal = _sig(
            strategy_name="pullback_continuation",
            market_regime=MarketRegime.RANGING_LOW_VOL,
            timeframe=Timeframe.FIFTEEN_MIN,
        )
        assert bot.should_take_signal(signal) is True

    def test_meanrev_takes_ranging_high_vol(self):
        """RANGING_HIGH_VOL + pullback_bear_continuation + M30 → accept."""
        bot = MeanReversionBot()
        signal = _sig(
            strategy_name="pullback_bear_continuation",
            market_regime=MarketRegime.RANGING_HIGH_VOL,
            timeframe=Timeframe.THIRTY_MIN,
        )
        assert bot.should_take_signal(signal) is True

    def test_meanrev_skips_trending(self):
        """TRENDING_UP regime → reject."""
        bot = MeanReversionBot()
        signal = _sig(
            strategy_name="pullback_continuation",
            market_regime=MarketRegime.TRENDING_UP,
            timeframe=Timeframe.FIFTEEN_MIN,
        )
        assert bot.should_take_signal(signal) is False

    def test_meanrev_skips_wrong_timeframe(self):
        """H1 timeframe → reject (only M15/M30 allowed)."""
        bot = MeanReversionBot()
        signal = _sig(
            strategy_name="pullback_continuation",
            market_regime=MarketRegime.RANGING_LOW_VOL,
            timeframe=Timeframe.ONE_HOUR,
        )
        assert bot.should_take_signal(signal) is False

    def test_meanrev_skips_wrong_strategy(self):
        """Correct regime + wrong strategy → reject."""
        bot = MeanReversionBot()
        signal = _sig(
            strategy_name="ema_crossover",
            market_regime=MarketRegime.RANGING_LOW_VOL,
            timeframe=Timeframe.FIFTEEN_MIN,
        )
        assert bot.should_take_signal(signal) is False


# ── ScalperBot ────────────────────────────────────────────────────────────────

class TestScalperBot:
    def test_scalper_takes_short_tf(self):
        """M5 + R:R >= 1.5 → accept."""
        bot = ScalperBot()
        signal = _sig(
            timeframe=Timeframe.FIVE_MIN,
            estimated_risk_reward=1.8,
        )
        assert bot.should_take_signal(signal) is True

    def test_scalper_takes_m1(self):
        """M1 + R:R >= 1.5 → accept."""
        bot = ScalperBot()
        signal = _sig(
            timeframe=Timeframe.ONE_MIN,
            estimated_risk_reward=2.0,
        )
        assert bot.should_take_signal(signal) is True

    def test_scalper_skips_long_tf(self):
        """H1 timeframe → reject (only M1/M5 allowed)."""
        bot = ScalperBot()
        signal = _sig(
            timeframe=Timeframe.ONE_HOUR,
            estimated_risk_reward=2.0,
        )
        assert bot.should_take_signal(signal) is False

    def test_scalper_skips_low_rr(self):
        """M5 + R:R 1.2 (below 1.5 minimum) → reject."""
        bot = ScalperBot()
        signal = _sig(
            timeframe=Timeframe.FIVE_MIN,
            estimated_risk_reward=1.2,
        )
        assert bot.should_take_signal(signal) is False

    def test_scalper_target_exit_is_tp1(self):
        """ScalperBot always targets tp1."""
        bot = ScalperBot()
        assert bot._target_exit() == "tp1"


# ── SwingBot ──────────────────────────────────────────────────────────────────

class TestSwingBot:
    def test_swing_takes_high_tf(self):
        """H1 + R:R >= 2.5 → accept."""
        bot = SwingBot()
        signal = _sig(
            timeframe=Timeframe.ONE_HOUR,
            estimated_risk_reward=3.0,
        )
        assert bot.should_take_signal(signal) is True

    def test_swing_takes_h4(self):
        """H4 + R:R >= 2.5 → accept."""
        bot = SwingBot()
        signal = _sig(
            timeframe=Timeframe.FOUR_HOUR,
            estimated_risk_reward=2.5,
        )
        assert bot.should_take_signal(signal) is True

    def test_swing_skips_short_tf(self):
        """M5 timeframe → reject (only H1/H4 allowed)."""
        bot = SwingBot()
        signal = _sig(
            timeframe=Timeframe.FIVE_MIN,
            estimated_risk_reward=3.0,
        )
        assert bot.should_take_signal(signal) is False

    def test_swing_skips_low_rr(self):
        """H4 + R:R 1.5 (below 2.5 minimum) → reject."""
        bot = SwingBot()
        signal = _sig(
            timeframe=Timeframe.FOUR_HOUR,
            estimated_risk_reward=1.5,
        )
        assert bot.should_take_signal(signal) is False

    def test_swing_target_exit_is_tp2(self):
        """SwingBot always targets tp2."""
        bot = SwingBot()
        assert bot._target_exit() == "tp2"
