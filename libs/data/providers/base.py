"""
Abstract base for all market-data provider adapters.

Design principle: the rest of the system NEVER imports a concrete
provider. It only depends on this interface. Swapping Alpaca for
Polygon, or Binance for Coinbase, is a configuration change only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator

import pandas as pd

from libs.core.models.domain import AssetClass, Candle, SymbolMetadata, Timeframe


class BaseDataProvider(ABC):
    """
    Contract all market-data providers must satisfy.

    Methods:
      get_candles()     — historical OHLCV fetch
      stream_candles()  — real-time candle stream (async generator)
      get_metadata()    — symbol details (tick size, lot size, etc.)
      ping()            — health check
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier, e.g. 'alpaca', 'binance'."""
        ...

    @property
    @abstractmethod
    def asset_class(self) -> AssetClass:
        """Which asset class this provider serves."""
        ...

    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical candles.

        Returns a DataFrame with DatetimeIndex (UTC) and columns:
            open, high, low, close, volume[, vwap, trade_count]
        Raises ProviderError on failure, NoDataError if empty.
        """
        ...

    @abstractmethod
    async def stream_candles(
        self,
        symbols: list[str],
        timeframe: Timeframe,
    ) -> AsyncIterator[Candle]:
        """
        Stream real-time candles as they close.
        Yields Candle objects. Runs indefinitely until cancelled.
        """
        ...

    @abstractmethod
    async def get_metadata(self, symbol: str) -> SymbolMetadata:
        """Return symbol-specific metadata."""
        ...

    @abstractmethod
    async def ping(self) -> bool:
        """Return True if provider is reachable and healthy."""
        ...

    # ── Optional but encouraged overrides ────────────────────────────────────

    async def get_latest_price(self, symbol: str) -> float | None:
        """Return the latest trade/mark price. None if unavailable."""
        return None

    async def get_supported_symbols(self) -> list[str]:
        """Return list of all tradable symbols on this provider."""
        return []
