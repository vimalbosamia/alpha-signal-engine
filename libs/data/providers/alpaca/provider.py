"""
Alpaca market-data provider adapter — US Equities.

Supports:
  - Historical bar fetching (REST)
  - Real-time bar streaming (WebSocket)
  - Basic plan / IEX feed coverage assumptions
  - Split + dividend adjusted data (adjustment="all")

This adapter is the ONLY place in the codebase that imports alpaca-py.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

import pandas as pd

from libs.core.config import get_settings
from libs.core.exceptions.exceptions import NoDataError, ProviderError
from libs.core.logging.logger import get_logger
from libs.core.models.domain import AssetClass, Candle, SymbolMetadata, Timeframe
from libs.data.providers.base import BaseDataProvider

log = get_logger(__name__)

# Alpaca timeframe string mapping
_TF_MAP: dict[Timeframe, str] = {
    Timeframe.ONE_MIN: "1Min",
    Timeframe.THREE_MIN: "3Min",
    Timeframe.FIVE_MIN: "5Min",
    Timeframe.FIFTEEN_MIN: "15Min",
    Timeframe.THIRTY_MIN: "30Min",
    Timeframe.ONE_HOUR: "1Hour",
    Timeframe.FOUR_HOUR: "4Hour",
    Timeframe.ONE_DAY: "1Day",
    Timeframe.ONE_WEEK: "1Week",
}


class AlpacaDataProvider(BaseDataProvider):
    """
    Alpaca Markets data provider for US equities.

    Uses alpaca-py SDK; lazy-initialised so tests can import without
    valid credentials.
    """

    def __init__(self) -> None:
        self._hist_client = None
        self._stream_client = None
        self._settings = get_settings().alpaca

    @property
    def name(self) -> str:
        return "alpaca"

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.STOCK

    def _get_hist_client(self):
        """Lazy-init historical data client."""
        if self._hist_client is None:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                self._hist_client = StockHistoricalDataClient(
                    api_key=self._settings.api_key.get_secret_value(),
                    secret_key=self._settings.secret_key.get_secret_value(),
                )
            except ImportError as e:
                raise ProviderError("alpaca", f"alpaca-py not installed: {e}") from e
        return self._hist_client

    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch historical bars from Alpaca REST API."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_str = _TF_MAP.get(timeframe)
        if not tf_str:
            raise ProviderError("alpaca", f"Unsupported timeframe: {timeframe}")

        # Build Alpaca TimeFrame object
        _alpaca_tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "3Min": TimeFrame(3, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "30Min": TimeFrame(30, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
            "1Week": TimeFrame(1, TimeFrameUnit.Week),
        }

        log.info(
            "alpaca_fetch",
            symbol=symbol,
            timeframe=timeframe.value,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        try:
            client = self._get_hist_client()
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=_alpaca_tf_map[tf_str],
                start=start,
                end=end,
                adjustment="all",
                feed=self._settings.data_feed,
                limit=limit,
            )
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, client.get_stock_bars, req)
            df = resp.df

            if df.empty:
                raise NoDataError(f"Alpaca returned no data for {symbol}")

            # Flatten multi-index (symbol, timestamp) → DatetimeIndex
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level="symbol")

            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "timestamp"
            df.columns = [c.lower() for c in df.columns]

            required = ["open", "high", "low", "close", "volume"]
            return df[[c for c in required + ["vwap", "trade_count"] if c in df.columns]].copy()

        except (NoDataError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderError("alpaca", f"get_candles failed for {symbol}: {exc}") from exc

    async def stream_candles(
        self,
        symbols: list[str],
        timeframe: Timeframe,
    ) -> AsyncIterator[Candle]:
        """
        Stream real-time minute bars via Alpaca WebSocket.
        Higher timeframes are aggregated in the CandleBuilder layer.
        """
        if timeframe != Timeframe.ONE_MIN:
            raise ProviderError(
                "alpaca",
                "Real-time streaming only supports 1m bars; "
                "higher timeframes are built by the CandleBuilder."
            )

        try:
            from alpaca.data.live import StockDataStream
        except ImportError as e:
            raise ProviderError("alpaca", f"alpaca-py not installed: {e}") from e

        queue: asyncio.Queue[Candle] = asyncio.Queue()

        async def _on_bar(bar) -> None:  # type: ignore[type-arg]
            candle = Candle(
                symbol=bar.symbol,
                asset_class=AssetClass.STOCK,
                timeframe=Timeframe.ONE_MIN,
                timestamp=bar.timestamp.replace(tzinfo=timezone.utc),
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                vwap=float(bar.vwap) if bar.vwap else None,
                trade_count=int(bar.trade_count) if bar.trade_count else None,
                is_confirmed=True,
            )
            await queue.put(candle)

        stream = StockDataStream(
            api_key=self._settings.api_key.get_secret_value(),
            secret_key=self._settings.secret_key.get_secret_value(),
            feed=self._settings.data_feed,
        )
        stream.subscribe_bars(_on_bar, *symbols)

        stream_task = asyncio.create_task(stream.run())
        log.info("alpaca_stream_started", symbols=symbols)

        try:
            while True:
                candle = await queue.get()
                yield candle
        finally:
            stream_task.cancel()
            log.info("alpaca_stream_stopped")

    async def get_metadata(self, symbol: str) -> SymbolMetadata:
        """Fetch symbol asset details from Alpaca."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import GetAssetsRequest

            settings = get_settings()
            tc = TradingClient(
                api_key=settings.alpaca.api_key.get_secret_value(),
                secret_key=settings.alpaca.secret_key.get_secret_value(),
                paper=True,
            )
            loop = asyncio.get_running_loop()
            asset = await loop.run_in_executor(None, tc.get_asset, symbol)

            return SymbolMetadata(
                symbol=symbol,
                asset_class=AssetClass.STOCK,
                tick_size=0.01,
                lot_size=1.0,
                is_tradable=bool(asset.tradable),
                exchange=str(asset.exchange),
            )
        except Exception as exc:
            log.warning("alpaca_metadata_failed", symbol=symbol, error=str(exc))
            return SymbolMetadata(symbol=symbol, asset_class=AssetClass.STOCK)

    async def ping(self) -> bool:
        try:
            client = self._get_hist_client()
            return client is not None
        except Exception:
            return False

    async def get_latest_price(self, symbol: str) -> float | None:
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            client = self._get_hist_client()
            loop = asyncio.get_running_loop()
            req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            resp = await loop.run_in_executor(None, client.get_stock_latest_trade, req)
            trade = resp.get(symbol)
            return float(trade.price) if trade else None
        except Exception:
            return None
