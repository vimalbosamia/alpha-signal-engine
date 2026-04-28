"""
Coinbase Advanced Trade data provider adapter — Crypto (stub).

This adapter is a READ-ONLY extension point. It currently raises
ProviderError on all live data methods. Implement when Coinbase
coverage is needed as a backup to Binance.

IMPORTANT: No order execution logic. Read-only data adapter only.
"""
from __future__ import annotations
from datetime import datetime
from typing import AsyncIterator
import pandas as pd
from libs.core.models.domain import AssetClass, Candle, SymbolMetadata, Timeframe
from libs.core.exceptions.exceptions import ProviderError
from libs.data.providers.base import BaseDataProvider


class CoinbaseDataProvider(BaseDataProvider):
    """Coinbase Advanced Trade data adapter (stub — not yet implemented)."""

    @property
    def name(self) -> str:
        return "coinbase"

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.CRYPTO

    async def get_candles(self, symbol, timeframe, start, end, limit=None) -> pd.DataFrame:
        raise ProviderError("coinbase", "get_candles not yet implemented")

    async def stream_candles(self, symbols, timeframe) -> AsyncIterator[Candle]:
        raise ProviderError("coinbase", "stream_candles not yet implemented")
        # unreachable, but needed to satisfy AsyncIterator type
        if False:
            yield  # type: ignore[misc]

    async def get_metadata(self, symbol) -> SymbolMetadata:
        return SymbolMetadata(symbol=symbol, asset_class=AssetClass.CRYPTO, exchange="coinbase")

    async def ping(self) -> bool:
        return False
