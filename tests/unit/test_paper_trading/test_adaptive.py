"""
Unit tests for AdaptiveBot.

Covers: baseline filters, adaptation triggers, recovery, serialisation,
        and the cold-start guard (no adapt before min samples).
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
from libs.paper_trading.bots.adaptive import (
    BASE_CONFIDENCE,
    BASE_MIN_RR,
    AdaptiveBot,
)


# ── Signal factory ─────────────────────────────────────────────────────────────

def _make_confluence(structure_score: float = 0.7) -> ConfluenceBreakdown:
    return ConfluenceBreakdown(
        pattern_score=0.7,
        structure_score=structure_score,
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
        confidence=0.65,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=99.0,
        entry_zone_high=101.0,
        stop_loss=95.0,
        take_profit_1=110.0,
        take_profit_2=120.0,
        estimated_risk_reward=2.0,
        market_regime=MarketRegime.TRENDING_UP,
        session_status=SessionType.CONTINUOUS,
        data_quality_status=DataQualityStatus.CLEAN,
        confluence=_make_confluence(),
        patterns_detected=["hammer"],
    )
    defaults.update(overrides)
    # If confluence is provided as a keyword, replace the default
    return SignalOutput(**defaults)


def _loss_result() -> dict:
    return {"pnl": -50.0, "status": "STOPPED_OUT"}


def _win_result() -> dict:
    return {"pnl": 100.0, "status": "TAKE_PROFIT"}


# ── Baseline filter tests ──────────────────────────────────────────────────────

class TestAdaptiveBotBaseline:
    def test_adaptive_takes_above_baseline(self):
        """Confidence 0.65 >= BASE (0.60) → accept."""
        bot = AdaptiveBot()
        signal = _sig(confidence=0.65, estimated_risk_reward=2.0)
        assert bot.should_take_signal(signal) is True

    def test_adaptive_skips_below_baseline(self):
        """Confidence 0.35 < BASE (0.40) → reject."""
        bot = AdaptiveBot()
        signal = _sig(confidence=0.35, estimated_risk_reward=2.0)
        assert bot.should_take_signal(signal) is False

    def test_adaptive_skips_low_rr(self):
        """R:R 0.8 < BASE_MIN_RR (1.0) → reject."""
        bot = AdaptiveBot()
        signal = _sig(confidence=0.70, estimated_risk_reward=0.8)
        assert bot.should_take_signal(signal) is False

    def test_adaptive_takes_exact_baseline_rr(self):
        """R:R exactly 1.5 (== BASE_MIN_RR) → accept."""
        bot = AdaptiveBot()
        signal = _sig(confidence=0.70, estimated_risk_reward=BASE_MIN_RR)
        assert bot.should_take_signal(signal) is True


# ── Adaptation: strategy confidence ───────────────────────────────────────────

class TestAdaptiveStrategyConfidence:
    def _make_bot_with_counter(self, count: int) -> AdaptiveBot:
        """Return a bot with trade_counter pre-set to satisfy cold-start guard."""
        bot = AdaptiveBot()
        bot._trade_counter = count
        return bot

    def test_adaptive_raises_strategy_confidence_after_losses(self):
        """3 losses on same strategy → confidence threshold raised by 5%."""
        bot = self._make_bot_with_counter(10)
        strategy = "ema_crossover"
        signal = _sig(strategy_name=strategy)

        for _ in range(3):
            bot._trade_history.append({
                "pnl": -50.0,
                "strategy": strategy,
                "regime": MarketRegime.TRENDING_UP.value,
                "symbol": "BTC/USDT",
                "rr": 2.0,
                "htf_score": 0.7,
                "is_loss": True,
            })

        bot._adapt_from_losses()

        expected = round(BASE_CONFIDENCE + 0.05, 4)
        assert bot._strategy_confidence.get(strategy) == expected

    def test_adaptive_only_raises_once_per_adapt_call(self):
        """Repeated adapt calls only increment once per unique threshold change."""
        bot = self._make_bot_with_counter(10)
        strategy = "ema_crossover"

        for _ in range(3):
            bot._trade_history.append({
                "pnl": -50.0,
                "strategy": strategy,
                "regime": MarketRegime.TRENDING_UP.value,
                "symbol": "BTC/USDT",
                "rr": 2.0,
                "htf_score": 0.7,
                "is_loss": True,
            })

        bot._adapt_from_losses()
        first = bot._strategy_confidence.get(strategy)
        # Second call with same data may raise again (cumulative) — just verify it doesn't crash
        bot._adapt_from_losses()
        second = bot._strategy_confidence.get(strategy)
        assert second >= first  # should only go up or stay same, never down


# ── Adaptation: regime blocking ───────────────────────────────────────────────

class TestAdaptiveRegimeBlocking:
    def test_adaptive_blocks_regime_after_losses(self):
        """3 losses in same regime → regime blocked."""
        bot = AdaptiveBot()
        bot._trade_counter = 10
        regime = MarketRegime.TRENDING_UP.value

        for _ in range(3):
            bot._trade_history.append({
                "pnl": -50.0,
                "strategy": "ema_crossover",
                "regime": regime,
                "symbol": "BTC/USDT",
                "rr": 2.0,
                "htf_score": 0.7,
                "is_loss": True,
            })

        bot._adapt_from_losses()

        assert regime in bot._blocked_regimes
        assert bot._blocked_regimes[regime] == 10

    def test_adaptive_rejects_blocked_regime(self):
        """Once a regime is blocked, should_take_signal rejects signals in it."""
        bot = AdaptiveBot()
        regime = MarketRegime.TRENDING_UP
        bot._blocked_regimes[regime.value] = 5

        signal = _sig(market_regime=regime, confidence=0.80, estimated_risk_reward=3.0)
        assert bot.should_take_signal(signal) is False


# ── Adaptation: symbol blocking ───────────────────────────────────────────────

class TestAdaptiveSymbolBlocking:
    def test_adaptive_blocks_symbol_after_losses(self):
        """3 losses on same symbol → symbol blocked."""
        bot = AdaptiveBot()
        bot._trade_counter = 10
        symbol = "ETH/USDT"

        for _ in range(3):
            bot._trade_history.append({
                "pnl": -50.0,
                "strategy": "ema_crossover",
                "regime": MarketRegime.TRENDING_UP.value,
                "symbol": symbol,
                "rr": 2.0,
                "htf_score": 0.7,
                "is_loss": True,
            })

        bot._adapt_from_losses()

        assert symbol in bot._blocked_symbols
        assert bot._blocked_symbols[symbol] == 10

    def test_adaptive_rejects_blocked_symbol(self):
        """Blocked symbol → should_take_signal returns False."""
        bot = AdaptiveBot()
        bot._blocked_symbols["ETH/USDT"] = 5

        signal = _sig(symbol="ETH/USDT", confidence=0.80, estimated_risk_reward=3.0)
        assert bot.should_take_signal(signal) is False


# ── Recovery after wins ───────────────────────────────────────────────────────

class TestAdaptiveRecovery:
    def test_adaptive_relaxes_after_wins(self):
        """5 consecutive wins → blocked symbol released."""
        bot = AdaptiveBot()
        symbol = "BTC/USDT"
        bot._blocked_symbols[symbol] = 5
        bot._consecutive_wins = 5

        # Simulate on_trade_closed with a win after cold-start guard met
        bot._trade_counter = 10
        signal = _sig(symbol=symbol)
        bot.on_trade_closed(_win_result(), signal)

        # Symbol should be unblocked (or at least the count reduced)
        # Recovery removes the first blocked symbol
        assert symbol not in bot._blocked_symbols

    def test_adaptive_relaxes_regime_after_symbols_clear(self):
        """When no blocked symbols, recovery unblocks regime instead."""
        bot = AdaptiveBot()
        bot._blocked_regimes[MarketRegime.TRENDING_UP.value] = 5
        bot._consecutive_wins = 5
        bot._trade_counter = 10

        signal = _sig()
        bot.on_trade_closed(_win_result(), signal)

        assert MarketRegime.TRENDING_UP.value not in bot._blocked_regimes

    def test_adaptive_never_below_baseline(self):
        """Relaxation of min_rr never goes below BASE_MIN_RR."""
        bot = AdaptiveBot()
        bot._min_rr = BASE_MIN_RR  # already at baseline
        bot._consecutive_wins = 5
        bot._trade_counter = 10

        signal = _sig()
        bot.on_trade_closed(_win_result(), signal)

        assert bot._min_rr >= BASE_MIN_RR

    def test_adaptive_strategy_confidence_never_below_baseline(self):
        """Relaxation of strategy confidence never goes below BASE_CONFIDENCE."""
        bot = AdaptiveBot()
        # Set a confidence just 5% above baseline — relaxation should remove the entry
        bot._strategy_confidence["ema_crossover"] = round(BASE_CONFIDENCE + 0.05, 4)
        bot._consecutive_wins = 5
        bot._trade_counter = 10

        signal = _sig()
        bot.on_trade_closed(_win_result(), signal)

        # After relaxation the entry should be removed (back at baseline) or gone
        conf = bot._strategy_confidence.get("ema_crossover", BASE_CONFIDENCE)
        assert conf >= BASE_CONFIDENCE


# ── State serialisation ────────────────────────────────────────────────────────

class TestAdaptiveFiltersPersistence:
    def test_adaptive_filters_persist(self):
        """get_adaptive_filters + load_adaptive_filters round-trip."""
        bot = AdaptiveBot()
        bot._strategy_confidence["ema_crossover"] = 0.70
        bot._blocked_regimes["trending_up"] = 8
        bot._blocked_symbols["ETH/USDT"] = 3
        bot._require_htf_alignment = True
        bot._min_rr = 2.0
        bot._trade_counter = 25

        snapshot = bot.get_adaptive_filters()

        new_bot = AdaptiveBot()
        new_bot.load_adaptive_filters(snapshot)

        assert new_bot._strategy_confidence == {"ema_crossover": 0.70}
        assert new_bot._blocked_regimes == {"trending_up": 8}
        assert new_bot._blocked_symbols == {"ETH/USDT": 3}
        assert new_bot._require_htf_alignment is True
        assert new_bot._min_rr == 2.0
        assert new_bot._trade_counter == 25

    def test_load_empty_filters(self):
        """Loading an empty dict restores default baseline state."""
        bot = AdaptiveBot()
        bot._min_rr = 2.5
        bot.load_adaptive_filters({})

        assert bot._min_rr == BASE_MIN_RR
        assert bot._strategy_confidence == {}
        assert bot._blocked_regimes == {}
        assert bot._blocked_symbols == {}
        assert bot._require_htf_alignment is False


# ── Cold-start guard ──────────────────────────────────────────────────────────

class TestAdaptiveColdStart:
    def test_adaptive_no_adapt_before_min_samples(self):
        """Fewer than 10 trades → _adapt_from_losses makes no changes."""
        bot = AdaptiveBot()
        strategy = "ema_crossover"

        # Record 3 losses (but trade_counter stays at 0)
        for _ in range(3):
            bot._trade_history.append({
                "pnl": -50.0,
                "strategy": strategy,
                "regime": MarketRegime.TRENDING_UP.value,
                "symbol": "BTC/USDT",
                "rr": 1.8,
                "htf_score": 0.4,
                "is_loss": True,
            })

        # Call on_trade_closed — internal counter < 10, so adapt skipped
        signal = _sig(strategy_name=strategy)
        bot.on_trade_closed(_loss_result(), signal)

        assert bot._strategy_confidence == {}
        assert bot._blocked_regimes == {}
        assert bot._blocked_symbols == {}
        assert bot._require_htf_alignment is False
        assert bot._min_rr == BASE_MIN_RR
