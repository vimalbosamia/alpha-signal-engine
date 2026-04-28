"""
Binance Spot market-data provider adapter — Crypto.

Supports:
  - Historical klines via REST (free, no auth required for public data)
  - Real-time kline/trade streams via WebSocket
  - Symbol metadata (tick size, lot size, filters)
  - 24/7 market — no session close logic

IMPORTANT: This adapter is READ-ONLY.
  - It fetches market data for analysis only.
  - It does NOT place orders.
  - It does NOT sign authenticated trade requests.
  - No order management code exists here.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncIterator

import pandas as pd

from libs.core.config import get_settings
from libs.core.exceptions.exceptions import NoDataError, ProviderError, ProviderDisconnectError
from libs.core.logging.logger import get_logger
from libs.core.models.domain import AssetClass, Candle, SymbolMetadata, Timeframe
from libs.data.providers.base import BaseDataProvider

log = get_logger(__name__)

# Binance kline interval strings
_TF_MAP: dict[Timeframe, str] = {
    Timeframe.ONE_MIN: "1m",
    Timeframe.THREE_MIN: "3m",
    Timeframe.FIVE_MIN: "5m",
    Timeframe.FIFTEEN_MIN: "15m",
    Timeframe.THIRTY_MIN: "30m",
    Timeframe.ONE_HOUR: "1h",
    Timeframe.FOUR_HOUR: "4h",
    Timeframe.ONE_DAY: "1d",
    Timeframe.ONE_WEEK: "1w",
}

_BINANCE_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "trade_count",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]


class BinanceDataProvider(BaseDataProvider):
    """
    Binance Spot data provider for crypto assets.

    Public REST endpoints require no authentication.
    WebSocket streams require no authentication either.

    Authentication keys are stored in config but are NEVER used
    in this signal-only implementation — they exist only for potential
    future rate-limit increases via authenticated endpoints.
    """

    def __init__(self) -> None:
        self._settings = get_settings().binance
        self._http_client = None

    @property
    def name(self) -> str:
        return "binance"

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.CRYPTO

    def _base_url(self) -> str:
        return self._settings.base_url

    async def _get_http(self):
        """Lazy-init async HTTP client."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(
                base_url=self._base_url(),
                timeout=30.0,
            )
        return self._http_client

    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical klines from Binance REST.
        No authentication required — public endpoint.
        """
        interval = _TF_MAP.get(timeframe)
        if not interval:
            raise ProviderError("binance", f"Unsupported timeframe: {timeframe}")

        # Binance uses millisecond timestamps
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        log.info(
            "binance_fetch",
            symbol=symbol,
            interval=interval,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        all_rows: list[list] = []
        fetch_start = start_ms

        try:
            client = await self._get_http()

            while True:
                params: dict = {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": fetch_start,
                    "endTime": end_ms,
                    "limit": min(limit or 1000, 1000),
                }

                resp = await client.get("/api/v3/klines", params=params)
                resp.raise_for_status()
                batch = resp.json()

                if not batch:
                    break

                all_rows.extend(batch)

                if len(batch) < 1000 or (limit and len(all_rows) >= limit):
                    break

                # Advance past last candle's close time
                fetch_start = int(batch[-1][6]) + 1

                if fetch_start >= end_ms:
                    break

        except Exception as exc:
            raise ProviderError("binance", f"get_candles failed for {symbol}: {exc}") from exc

        if not all_rows:
            raise NoDataError(f"Binance returned no klines for {symbol}")

        df = pd.DataFrame(all_rows, columns=_BINANCE_KLINE_COLS)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").astype("Int64")

        return df[["open", "high", "low", "close", "volume", "trade_count"]].copy()

    async def stream_candles(
        self,
        symbols: list[str],
        timeframe: Timeframe,
    ) -> AsyncIterator[Candle]:
        """
        Stream real-time kline updates via Binance WebSocket.
        Yields only CONFIRMED (closed) candles.
        """
        import websockets

        interval = _TF_MAP.get(timeframe)
        if not interval:
            raise ProviderError("binance", f"Unsupported timeframe: {timeframe}")

        # Build combined stream URL for multiple symbols
        streams = "/".join(f"{s.lower()}@kline_{interval}" for s in symbols)
        ws_base = "wss://testnet.binance.vision" if self._settings.env == "testnet" else "wss://stream.binance.com:9443"
        url = f"{ws_base}/stream?streams={streams}"

        reconnect_attempts = self._settings.ws_reconnect_attempts
        attempt = 0

        while attempt < reconnect_attempts:
            try:
                log.info("binance_ws_connecting", url=url, symbols=symbols)
                async with websockets.connect(
                    url,
                    ping_interval=self._settings.ws_ping_interval,
                ) as ws:
                    attempt = 0  # reset on successful connection
                    log.info("binance_ws_connected", symbols=symbols)

                    async for raw_msg in ws:
                        try:
                            msg = json.loads(raw_msg)
                            data = msg.get("data", msg)
                            kline = data.get("k", {})

                            # Only yield confirmed (closed) candles
                            if not kline.get("x", False):
                                continue

                            ts = datetime.fromtimestamp(
                                int(kline["t"]) / 1000, tz=timezone.utc
                            )
                            symbol = kline["s"]

                            candle = Candle(
                                symbol=symbol,
                                asset_class=AssetClass.CRYPTO,
                                timeframe=timeframe,
                                timestamp=ts,
                                open=float(kline["o"]),
                                high=float(kline["h"]),
                                low=float(kline["l"]),
                                close=float(kline["c"]),
                                volume=float(kline["v"]),
                                trade_count=int(kline.get("n", 0)),
                                is_confirmed=True,
                            )
                            yield candle

                        except (KeyError, ValueError, json.JSONDecodeError) as exc:
                            log.warning("binance_ws_parse_error", error=str(exc))

            except Exception as exc:
                attempt += 1
                log.warning(
                    "binance_ws_disconnected",
                    attempt=attempt,
                    max_attempts=reconnect_attempts,
                    error=str(exc),
                )
                if attempt >= reconnect_attempts:
                    raise ProviderDisconnectError(
                        "binance",
                        f"WebSocket failed after {attempt} reconnection attempts"
                    ) from exc
                await asyncio.sleep(min(2 ** attempt, 30))  # exponential backoff

    async def get_metadata(self, symbol: str) -> SymbolMetadata:
        """Fetch symbol filters from Binance Exchange Info."""
        try:
            client = await self._get_http()
            resp = await client.get("/api/v3/exchangeInfo", params={"symbol": symbol.upper()})
            resp.raise_for_status()
            data = resp.json()

            sym_info = next(
                (s for s in data.get("symbols", []) if s["symbol"] == symbol.upper()),
                None,
            )
            if not sym_info:
                return SymbolMetadata(symbol=symbol, asset_class=AssetClass.CRYPTO)

            tick_size = 0.01
            lot_size = 0.001

            for f in sym_info.get("filters", []):
                if f["filterType"] == "PRICE_FILTER":
                    tick_size = float(f["tickSize"])
                elif f["filterType"] == "LOT_SIZE":
                    lot_size = float(f["stepSize"])

            return SymbolMetadata(
                symbol=symbol,
                asset_class=AssetClass.CRYPTO,
                base_currency=sym_info.get("baseAsset", ""),
                quote_currency=sym_info.get("quoteAsset", "USDT"),
                tick_size=tick_size,
                lot_size=lot_size,
                is_tradable=sym_info.get("status") == "TRADING",
                exchange="binance",
            )

        except Exception as exc:
            log.warning("binance_metadata_failed", symbol=symbol, error=str(exc))
            return SymbolMetadata(symbol=symbol, asset_class=AssetClass.CRYPTO)

    async def ping(self) -> bool:
        try:
            client = await self._get_http()
            resp = await client.get("/api/v3/ping")
            return resp.status_code == 200
        except Exception:
            return False

    async def get_latest_price(self, symbol: str) -> float | None:
        try:
            client = await self._get_http()
            resp = await client.get(
                "/api/v3/ticker/price",
                params={"symbol": symbol.upper()},
            )
            resp.raise_for_status()
            return float(resp.json()["price"])
        except Exception:
            return None

    async def get_supported_symbols(self) -> list[str]:
        try:
            client = await self._get_http()
            resp = await client.get("/api/v3/exchangeInfo")
            resp.raise_for_status()
            return [
                s["symbol"]
                for s in resp.json().get("symbols", [])
                if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
            ]
        except Exception:
            return []
