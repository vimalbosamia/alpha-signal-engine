"""
Binance USDT-M Perpetual Futures data provider — READ-ONLY.

Differences from spot (BinanceDataProvider):
  REST base : https://fapi.binance.com
  Klines    : GET /fapi/v1/klines
  Price     : GET /fapi/v1/ticker/price
  WS base   : wss://fstream.binance.com
  Streams   : <symbol>@miniTicker  (same field names as spot)

Funding-rate and open-interest endpoints are not used here — this
provider only fetches OHLCV candles and latest mark/last prices for
signal generation and position-watcher price checks.

IMPORTANT: READ-ONLY. No order placement. No signing.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncIterator

import pandas as pd

from libs.core.exceptions.exceptions import NoDataError, ProviderError, ProviderDisconnectError
from libs.core.logging.logger import get_logger
from libs.core.models.domain import AssetClass, Candle, SymbolMetadata, Timeframe
from libs.data.providers.base import BaseDataProvider

log = get_logger(__name__)

_BASE_REST = "https://fapi.binance.com"
_BASE_WS   = "wss://fstream.binance.com"

_TF_MAP: dict[Timeframe, str] = {
    Timeframe.ONE_MIN:     "1m",
    Timeframe.THREE_MIN:   "3m",
    Timeframe.FIVE_MIN:    "5m",
    Timeframe.FIFTEEN_MIN: "15m",
    Timeframe.THIRTY_MIN:  "30m",
    Timeframe.ONE_HOUR:    "1h",
    Timeframe.FOUR_HOUR:   "4h",
    Timeframe.ONE_DAY:     "1d",
    Timeframe.ONE_WEEK:    "1w",
}

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "trade_count",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]


class BinanceFuturesProvider(BaseDataProvider):
    """
    Binance USDT-M Perpetual Futures data provider.

    Provides OHLCV candles and latest prices for futures contracts.
    Signals produced from this provider represent perpetual futures setups,
    NOT spot market positions.
    """

    def __init__(self) -> None:
        self._http_client = None

    @property
    def name(self) -> str:
        return "binance_futures"

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.CRYPTO

    async def _get_http(self):
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(
                base_url=_BASE_REST,
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
        interval = _TF_MAP.get(timeframe)
        if not interval:
            raise ProviderError("binance_futures", f"Unsupported timeframe: {timeframe}")

        start_ms = int(start.timestamp() * 1000)
        end_ms   = int(end.timestamp()   * 1000)

        log.info(
            "binance_futures_fetch",
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
                params = {
                    "symbol":    symbol.upper(),
                    "interval":  interval,
                    "startTime": fetch_start,
                    "endTime":   end_ms,
                    "limit":     min(limit or 1000, 1500),  # futures allows up to 1500
                }
                resp = await client.get("/fapi/v1/klines", params=params)
                resp.raise_for_status()
                batch = resp.json()

                if not batch:
                    break

                all_rows.extend(batch)

                if len(batch) < 1000 or (limit and len(all_rows) >= limit):
                    break

                fetch_start = int(batch[-1][6]) + 1
                if fetch_start >= end_ms:
                    break

        except Exception as exc:
            raise ProviderError(
                "binance_futures", f"get_candles failed for {symbol}: {exc}"
            ) from exc

        if not all_rows:
            raise NoDataError(f"Binance Futures returned no klines for {symbol}")

        df = pd.DataFrame(all_rows, columns=_KLINE_COLS)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["trade_count"] = pd.to_numeric(
            df["trade_count"], errors="coerce"
        ).astype("Int64")

        return df[["open", "high", "low", "close", "volume", "trade_count"]].copy()

    async def stream_candles(
        self,
        symbols: list[str],
        timeframe: Timeframe,
    ) -> AsyncIterator[Candle]:
        import websockets

        interval = _TF_MAP.get(timeframe)
        if not interval:
            raise ProviderError("binance_futures", f"Unsupported timeframe: {timeframe}")

        streams = "/".join(f"{s.lower()}@kline_{interval}" for s in symbols)
        url = f"{_BASE_WS}/stream?streams={streams}"

        attempt = 0
        max_attempts = 5

        while attempt < max_attempts:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    attempt = 0
                    async for raw_msg in ws:
                        try:
                            msg   = json.loads(raw_msg)
                            data  = msg.get("data", msg)
                            kline = data.get("k", {})

                            if not kline.get("x", False):
                                continue

                            ts = datetime.fromtimestamp(
                                int(kline["t"]) / 1000, tz=timezone.utc
                            )
                            yield Candle(
                                symbol=kline["s"],
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
                        except (KeyError, ValueError, json.JSONDecodeError) as exc:
                            log.warning("binance_futures_ws_parse_error", error=str(exc))

            except Exception as exc:
                attempt += 1
                log.warning(
                    "binance_futures_ws_disconnected",
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt >= max_attempts:
                    raise ProviderDisconnectError(
                        "binance_futures",
                        f"WS failed after {attempt} attempts",
                    ) from exc
                await asyncio.sleep(min(2 ** attempt, 30))

    async def get_latest_price(self, symbol: str) -> float | None:
        """Fetch latest futures mark price (most accurate for futures P&L)."""
        try:
            client = await self._get_http()
            # Use mark price for more accurate futures pricing
            resp = await client.get(
                "/fapi/v1/ticker/price",
                params={"symbol": symbol.upper()},
            )
            resp.raise_for_status()
            return float(resp.json()["price"])
        except Exception:
            return None

    async def get_metadata(self, symbol: str) -> SymbolMetadata:
        try:
            client = await self._get_http()
            resp = await client.get(
                "/fapi/v1/exchangeInfo",
            )
            resp.raise_for_status()
            sym_info = next(
                (s for s in resp.json().get("symbols", [])
                 if s["symbol"] == symbol.upper()),
                None,
            )
            if not sym_info:
                return SymbolMetadata(symbol=symbol, asset_class=AssetClass.CRYPTO)

            tick_size = 0.01
            lot_size  = 0.001
            for f in sym_info.get("filters", []):
                if f["filterType"] == "PRICE_FILTER":
                    tick_size = float(f["tickSize"])
                elif f["filterType"] == "LOT_SIZE":
                    lot_size = float(f.get("stepSize", lot_size))

            return SymbolMetadata(
                symbol=symbol,
                asset_class=AssetClass.CRYPTO,
                base_currency=sym_info.get("baseAsset", ""),
                quote_currency=sym_info.get("quoteAsset", "USDT"),
                tick_size=tick_size,
                lot_size=lot_size,
                is_tradable=sym_info.get("status") == "TRADING",
                exchange="binance_futures",
            )
        except Exception as exc:
            log.warning("binance_futures_metadata_failed", symbol=symbol, error=str(exc))
            return SymbolMetadata(symbol=symbol, asset_class=AssetClass.CRYPTO)

    async def ping(self) -> bool:
        try:
            client = await self._get_http()
            resp = await client.get("/fapi/v1/ping")
            return resp.status_code == 200
        except Exception:
            return False

    async def get_supported_symbols(self) -> list[str]:
        try:
            client = await self._get_http()
            resp = await client.get("/fapi/v1/exchangeInfo")
            resp.raise_for_status()
            return [
                s["symbol"]
                for s in resp.json().get("symbols", [])
                if s.get("status") == "TRADING"
                and s.get("quoteAsset") == "USDT"
                and s.get("contractType") == "PERPETUAL"
            ]
        except Exception:
            return []
