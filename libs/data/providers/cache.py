"""
ProviderCache — singleton cache for data providers.

Prevents creating new provider instances per request, eliminating
repeated connection setup and config parsing overhead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libs.data.providers.binance.provider import BinanceDataProvider
    from libs.data.providers.alpaca.provider import AlpacaDataProvider


class ProviderCache:
    """Singleton cache for data providers — prevents creating new instances per request."""

    _binance: BinanceDataProvider | None = None
    _alpaca: AlpacaDataProvider | None = None

    @classmethod
    def binance(cls) -> BinanceDataProvider:
        if cls._binance is None:
            from libs.data.providers.binance.provider import BinanceDataProvider
            cls._binance = BinanceDataProvider()
        return cls._binance

    @classmethod
    def alpaca(cls) -> AlpacaDataProvider:
        if cls._alpaca is None:
            from libs.data.providers.alpaca.provider import AlpacaDataProvider
            cls._alpaca = AlpacaDataProvider()
        return cls._alpaca

    @classmethod
    def for_symbol(cls, symbol: str) -> BinanceDataProvider | AlpacaDataProvider:
        """Return appropriate provider for a symbol.

        Crypto pairs end in USDT (e.g. BTCUSDT) → Binance.
        Everything else (equities, ETFs) → Alpaca.
        """
        return cls.binance() if symbol.endswith("USDT") else cls.alpaca()

    @classmethod
    def reset(cls) -> None:
        """Clear all cached instances (useful for testing)."""
        cls._binance = None
        cls._alpaca = None
