"""
HistoricalDataProvider — serves cached parquet data via BaseDataProvider interface.

Used for backtesting and historical training. Reads data downloaded by
BinanceHistoricalDownloader and serves it through the same interface as
live providers, so the full pipeline works unchanged.
"""
from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator

import pandas as pd

from libs.core.models.domain import AssetClass, Candle, SymbolMetadata, Timeframe
from libs.data.providers.base import BaseDataProvider
from libs.data.providers.historical.downloader import BinanceHistoricalDownloader


class HistoricalDataProvider(BaseDataProvider):
    """Serves historical OHLCV data from local parquet files."""

    def __init__(self, data_dir: str = "data/historical") -> None:
        self._downloader = BinanceHistoricalDownloader(data_dir=data_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def name(self) -> str:
        return "historical"

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.CRYPTO

    def preload(self, symbol: str, timeframe: str = "15m") -> bool:
        """Load symbol data into memory cache. Returns True if data available."""
        if symbol in self._cache:
            return True
        df = self._downloader.load_symbol(symbol, timeframe)
        if df is None:
            return False
        self._cache[symbol] = df
        return True

    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV data from cache, filtered to [start, end]."""
        tf_str = timeframe.value if hasattr(timeframe, "value") else str(timeframe)

        if symbol not in self._cache:
            df = self._downloader.load_symbol(symbol, tf_str)
            if df is None:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            self._cache[symbol] = df

        df = self._cache[symbol]

        # Make start/end timezone-aware if needed
        if start.tzinfo is None:
            from datetime import timezone
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            from datetime import timezone
            end = end.replace(tzinfo=timezone.utc)

        mask = (df.index >= start) & (df.index <= end)
        result = df.loc[mask].copy()

        if limit and len(result) > limit:
            result = result.tail(limit)

        result.attrs["symbol"] = symbol
        return result

    async def stream_candles(
        self,
        symbols: list[str],
        timeframe: Timeframe,
    ) -> AsyncIterator[Candle]:
        """Not supported for historical data."""
        raise NotImplementedError("Historical provider does not support streaming")
        yield  # make it a generator

    async def get_metadata(self, symbol: str) -> SymbolMetadata:
        """Return basic metadata for a symbol."""
        return SymbolMetadata(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            tick_size=0.01,
            lot_size=0.001,
            min_notional=10.0,
        )

    async def ping(self) -> bool:
        """Always healthy — data is local."""
        return True

    async def get_latest_price(self, symbol: str) -> float | None:
        """Return last close price from cached data."""
        if symbol in self._cache and len(self._cache[symbol]) > 0:
            return float(self._cache[symbol]["close"].iloc[-1])
        return None

    async def get_supported_symbols(self) -> list[str]:
        """Return symbols that have cached data."""
        return list(self._cache.keys())
