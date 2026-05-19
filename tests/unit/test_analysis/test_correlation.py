"""
Unit tests for MarketCorrelationFilter.

All price data is synthetic — no external APIs are called.
Async methods (check_crypto / check_stock) are tested via mocking
the internal _fetch_leader_prices helper so no network I/O occurs.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from libs.analysis.macro.correlation import (
    CorrelationCheck,
    MarketCorrelationFilter,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def run(coro):
    """Run a coroutine synchronously (Python 3.10-compatible helper)."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _rising(n: int = 30, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start + i * step for i in range(n)]


def _falling(n: int = 30, start: float = 130.0, step: float = 1.0) -> list[float]:
    return [start - i * step for i in range(n)]


def _flat(n: int = 30, value: float = 100.0) -> list[float]:
    return [value] * n


# ── _calc_trend ────────────────────────────────────────────────────────────────

class TestCalcTrend:
    """_calc_trend is pure / synchronous — tested directly, no mocks needed."""

    def setup_method(self):
        self.filt = MarketCorrelationFilter()

    def test_calc_trend_bullish(self):
        """Steadily rising prices → ('bullish', positive change)."""
        prices = _rising(30, start=100.0, step=1.0)
        direction, change_pct = self.filt._calc_trend(prices)
        assert direction == "bullish"
        assert change_pct > 0.0

    def test_calc_trend_bearish(self):
        """Steadily falling prices → ('bearish', negative change)."""
        prices = _falling(30, start=130.0, step=1.0)
        direction, change_pct = self.filt._calc_trend(prices)
        assert direction == "bearish"
        assert change_pct < 0.0

    def test_calc_trend_flat(self):
        """Flat prices → ('neutral', ~0 change)."""
        prices = _flat(30, value=100.0)
        direction, change_pct = self.filt._calc_trend(prices)
        assert direction == "neutral"
        assert abs(change_pct) < 0.01


# ── check_crypto (async, mocked) ───────────────────────────────────────────────

class TestCheckCrypto:
    """check_crypto logic tested by patching _fetch_leader_prices."""

    def setup_method(self):
        self.filt = MarketCorrelationFilter()

    def test_btc_dumping_blocks_altcoin_buy(self):
        """BTC bearish + BUY ETHUSDT → conflicts_with_trade=True."""
        bearish_prices = _falling(30)

        with patch.object(
            self.filt, "_fetch_crypto_prices", new=AsyncMock(return_value=bearish_prices)
        ):
            result: CorrelationCheck = run(self.filt.check_crypto("ETHUSDT", "BUY"))

        assert result.conflicts_with_trade is True
        assert result.leader_symbol == "BTCUSDT"
        assert result.leader_trend == "bearish"

    def test_btc_bullish_allows_altcoin_buy(self):
        """BTC bullish + BUY ETHUSDT → conflicts_with_trade=False."""
        bullish_prices = _rising(30)

        with patch.object(
            self.filt, "_fetch_crypto_prices", new=AsyncMock(return_value=bullish_prices)
        ):
            result: CorrelationCheck = run(self.filt.check_crypto("ETHUSDT", "BUY"))

        assert result.conflicts_with_trade is False
        assert result.leader_trend == "bullish"

    def test_btc_trend_doesnt_block_btc_itself(self):
        """BUY BTCUSDT + BTC bearish → conflicts_with_trade=False (BTC trades its own signal)."""
        bearish_prices = _falling(30)

        with patch.object(
            self.filt, "_fetch_crypto_prices", new=AsyncMock(return_value=bearish_prices)
        ):
            result: CorrelationCheck = run(self.filt.check_crypto("BTCUSDT", "BUY"))

        assert result.conflicts_with_trade is False


# ── CorrelationCheck dataclass ─────────────────────────────────────────────────

class TestCorrelationCheckDataclass:
    """CorrelationCheck must be frozen and carry expected fields."""

    def test_is_frozen(self):
        """CorrelationCheck should be immutable."""
        check = CorrelationCheck(
            leader_symbol="BTCUSDT",
            leader_trend="bullish",
            leader_change_pct=2.5,
            correlation_strength="strong",
            conflicts_with_trade=False,
            explanation="BTC is bullish; trade aligned",
        )
        with pytest.raises((AttributeError, TypeError)):
            check.leader_trend = "bearish"  # type: ignore[misc]

    def test_fields_populated(self):
        """All six fields must be accessible."""
        check = CorrelationCheck(
            leader_symbol="SPY",
            leader_trend="bearish",
            leader_change_pct=-1.2,
            correlation_strength="moderate",
            conflicts_with_trade=True,
            explanation="SPY is falling; blocking tech long",
        )
        assert check.leader_symbol == "SPY"
        assert check.leader_trend == "bearish"
        assert check.leader_change_pct == pytest.approx(-1.2)
        assert check.correlation_strength == "moderate"
        assert check.conflicts_with_trade is True
        assert "SPY" in check.explanation
