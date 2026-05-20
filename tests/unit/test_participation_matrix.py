"""Tests for MarketParticipationMatrix."""
import pytest
from libs.analysis.participation.matrix import (
    MarketState,
    classify_market_state,
    get_participation,
)


class TestClassifyMarketState:
    def test_news_lockdown_overrides_all(self):
        state = classify_market_state("bullish", 0.5, 0.1, 0.3, 30, is_news_lockdown=True)
        assert state == MarketState.NEWS_LOCKDOWN

    def test_high_volatility_overrides(self):
        state = classify_market_state("bullish", 0.5, 0.1, 0.3, 30, is_high_volatility=True)
        assert state == MarketState.HIGH_VOLATILITY_EVENT

    def test_high_conflict_with_spread_is_weak_bull(self):
        # spread=0.05 > 0.01 → WEAK_BULL even at high conflict
        state = classify_market_state("bullish", 0.3, 0.25, 0.95, 20)
        assert state == MarketState.WEAK_BULL

    def test_high_conflict_no_spread_is_neutral(self):
        # spread < 0.02, conflict > 0.90 → truly NEUTRAL
        state = classify_market_state("neutral", 0.20, 0.19, 0.95, 20)
        assert state == MarketState.NEUTRAL

    def test_strong_bull(self):
        state = classify_market_state("bullish", 0.5, 0.1, 0.4, 30)
        assert state == MarketState.STRONG_BULL

    def test_strong_bear(self):
        state = classify_market_state("bearish", 0.1, 0.5, 0.4, 30)
        assert state == MarketState.STRONG_BEAR

    def test_weak_bull(self):
        state = classify_market_state("bullish", 0.25, 0.15, 0.6, 12)
        assert state == MarketState.WEAK_BULL

    def test_weak_bear(self):
        state = classify_market_state("bearish", 0.15, 0.25, 0.6, 12)
        assert state == MarketState.WEAK_BEAR

    def test_neutral_low_scores(self):
        state = classify_market_state("neutral", 0.1, 0.1, 0.85, 10)
        assert state == MarketState.NEUTRAL


class TestGetParticipation:
    def test_strong_bull_all_long(self):
        p = get_participation("bullish", 0.5, 0.1, 0.3, 30)
        assert p.spot_allowed is True
        assert p.futures_long_allowed is True
        assert p.futures_short_allowed is False

    def test_strong_bear_short_only(self):
        p = get_participation("bearish", 0.1, 0.5, 0.3, 30)
        assert p.spot_allowed is False
        assert p.futures_long_allowed is False
        assert p.futures_short_allowed is True

    def test_neutral_blocks_all(self):
        p = get_participation("neutral", 0.2, 0.2, 0.95, 15)
        assert p.spot_allowed is False
        assert p.futures_long_allowed is False
        assert p.futures_short_allowed is False

    def test_news_lockdown_blocks_all(self):
        p = get_participation("bullish", 0.5, 0.1, 0.3, 30, is_news_lockdown=True)
        assert p.spot_allowed is False
        assert p.futures_long_allowed is False
        assert p.futures_short_allowed is False
        assert p.confidence_multiplier == 0.0

    def test_weak_bear_futures_short_reduced_size(self):
        p = get_participation("bearish", 0.15, 0.25, 0.6, 12)
        assert p.futures_short_allowed is True
        assert p.size_multiplier < 1.0

    def test_high_conflict_reduces_multipliers(self):
        p = get_participation("bullish", 0.35, 0.15, 0.80, 30)
        # STRONG_BULL but high conflict → multipliers reduced
        assert p.confidence_multiplier < 1.0
        assert p.size_multiplier < 1.0
