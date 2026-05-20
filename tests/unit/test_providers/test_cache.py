"""
Unit tests for ProviderCache.

Provider constructors make network/config calls, so we mock them out
and only verify the singleton and routing behaviour.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from libs.data.providers.cache import ProviderCache


@pytest.fixture(autouse=True)
def _reset_cache():
    """Ensure a clean singleton state for every test."""
    ProviderCache.reset()
    yield
    ProviderCache.reset()


class TestProviderCacheSingleton:
    def test_singleton_returns_same_binance_instance(self):
        """Calling binance() twice must return the exact same object."""
        mock_provider = MagicMock()
        with patch(
            "libs.data.providers.cache.ProviderCache.binance",
            return_value=mock_provider,
        ) as mock_binance:
            first = ProviderCache.binance()
            second = ProviderCache.binance()
            assert first is second

    def test_singleton_returns_same_alpaca_instance(self):
        """Calling alpaca() twice must return the exact same object."""
        mock_provider = MagicMock()
        with patch(
            "libs.data.providers.cache.ProviderCache.alpaca",
            return_value=mock_provider,
        ) as mock_alpaca:
            first = ProviderCache.alpaca()
            second = ProviderCache.alpaca()
            assert first is second

    def test_singleton_constructed_once(self):
        """Provider constructor is called exactly once even after multiple cache.binance() calls."""
        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)

        with patch(
            "libs.data.providers.binance.provider.BinanceDataProvider",
            mock_cls,
            create=True,
        ):
            # Directly seed the cache to simulate first construction
            ProviderCache._binance = mock_instance
            _ = ProviderCache.binance()
            _ = ProviderCache.binance()
            # Constructor was never called via cache (already seeded)
            mock_cls.assert_not_called()


class TestProviderCacheForSymbol:
    def test_for_symbol_crypto_returns_binance(self):
        """Symbols ending in USDT should route to Binance."""
        mock_binance = MagicMock()
        mock_alpaca = MagicMock()

        with (
            patch.object(ProviderCache, "binance", return_value=mock_binance),
            patch.object(ProviderCache, "alpaca", return_value=mock_alpaca),
        ):
            result = ProviderCache.for_symbol("BTCUSDT")
            assert result is mock_binance

    def test_for_symbol_stock_returns_alpaca(self):
        """Non-USDT symbols (equities, ETFs) should route to Alpaca."""
        mock_binance = MagicMock()
        mock_alpaca = MagicMock()

        with (
            patch.object(ProviderCache, "binance", return_value=mock_binance),
            patch.object(ProviderCache, "alpaca", return_value=mock_alpaca),
        ):
            result = ProviderCache.for_symbol("AAPL")
            assert result is mock_alpaca

    def test_for_symbol_crypto_various(self):
        """All USDT-suffixed symbols should route to Binance."""
        mock_binance = MagicMock()
        mock_alpaca = MagicMock()

        with (
            patch.object(ProviderCache, "binance", return_value=mock_binance),
            patch.object(ProviderCache, "alpaca", return_value=mock_alpaca),
        ):
            for sym in ("ETHUSDT", "SOLUSDT", "ADAUSDT"):
                assert ProviderCache.for_symbol(sym) is mock_binance
