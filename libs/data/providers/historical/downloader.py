"""
Binance Historical Data Downloader — fetches OHLCV klines from data.binance.vision.

Downloads monthly CSV files for spot klines and concatenates into per-symbol
parquet files for fast loading during training.

Usage:
    from libs.data.providers.historical.downloader import BinanceHistoricalDownloader

    dl = BinanceHistoricalDownloader()
    await dl.download_symbol("BTCUSDT", timeframe="15m", months=12)
    await dl.download_all(symbols=["BTCUSDT", "ETHUSDT"], timeframe="15m", months=12)

Data source: https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/
"""
from __future__ import annotations

import asyncio
import io
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd

from libs.core.logging.logger import get_logger

log = get_logger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
DATA_DIR = "data/historical"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


class BinanceHistoricalDownloader:
    """Downloads historical klines from Binance public data."""

    def __init__(self, data_dir: str = DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _output_path(self, symbol: str, timeframe: str) -> Path:
        return self._data_dir / f"{symbol}_{timeframe}.parquet"

    def _csv_cache_dir(self, symbol: str, timeframe: str) -> Path:
        d = self._data_dir / "csv_cache" / symbol / timeframe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def has_data(self, symbol: str, timeframe: str) -> bool:
        """Check if parquet already exists for this symbol+tf."""
        return self._output_path(symbol, timeframe).exists()

    def _generate_months(self, months: int) -> list[tuple[int, int]]:
        """Generate (year, month) tuples going back N months from now."""
        now = datetime.now(timezone.utc)
        result = []
        y, m = now.year, now.month
        for _ in range(months):
            m -= 1
            if m == 0:
                m = 12
                y -= 1
            result.append((y, m))
        return list(reversed(result))

    async def _download_month(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        timeframe: str,
        year: int,
        month: int,
    ) -> pd.DataFrame | None:
        """Download one month of klines. Returns DataFrame or None if not available."""
        cache_dir = self._csv_cache_dir(symbol, timeframe)
        cache_file = cache_dir / f"{year}-{month:02d}.csv"

        # Use cache if exists
        if cache_file.exists():
            try:
                return pd.read_csv(cache_file, header=None, names=KLINE_COLUMNS)
            except Exception:
                cache_file.unlink(missing_ok=True)

        filename = f"{symbol}-{timeframe}-{year}-{month:02d}.zip"
        url = f"{BASE_URL}/{symbol}/{timeframe}/{filename}"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    log.debug("download_skip", symbol=symbol, month=f"{year}-{month:02d}",
                              status=resp.status)
                    return None
                data = await resp.read()
        except Exception as exc:
            log.warning("download_error", symbol=symbol, month=f"{year}-{month:02d}",
                        error=str(exc))
            return None

        # Extract CSV from zip
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                csv_name = zf.namelist()[0]
                csv_bytes = zf.read(csv_name)

            # Cache the extracted CSV
            cache_file.write_bytes(csv_bytes)

            df = pd.read_csv(io.BytesIO(csv_bytes), header=None, names=KLINE_COLUMNS)
            log.info("download_ok", symbol=symbol, month=f"{year}-{month:02d}",
                     rows=len(df))
            return df
        except Exception as exc:
            log.warning("extract_error", symbol=symbol, month=f"{year}-{month:02d}",
                        error=str(exc))
            return None

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format a raw klines DataFrame."""
        df = df.copy()

        # Detect timestamp unit: microseconds (16 digits) vs milliseconds (13 digits)
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
        df = df.dropna(subset=["open_time"])
        if len(df) == 0:
            return df

        sample_ts = df["open_time"].iloc[0]
        if sample_ts > 1e15:
            # Microseconds — convert to milliseconds
            df["open_time"] = df["open_time"] // 1000

        # Filter out obviously bad timestamps (> year 2100 or < year 2000)
        valid_ts = (df["open_time"] > 946684800000) & (df["open_time"] < 4102444800000)
        df = df[valid_ts]

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.set_index("open_time")
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df

    async def download_symbol(
        self,
        symbol: str,
        timeframe: str = "15m",
        months: int = 12,
        force: bool = False,
    ) -> Path | None:
        """Download and save historical data for one symbol.

        Returns path to parquet file, or None on failure.
        """
        output = self._output_path(symbol, timeframe)
        if output.exists() and not force:
            log.info("already_cached", symbol=symbol, path=str(output))
            return output

        month_list = self._generate_months(months)
        frames: list[pd.DataFrame] = []

        async with aiohttp.ClientSession() as session:
            # Download in batches of 4 for politeness
            for i in range(0, len(month_list), 4):
                batch = month_list[i:i + 4]
                tasks = [
                    self._download_month(session, symbol, timeframe, y, m)
                    for y, m in batch
                ]
                results = await asyncio.gather(*tasks)
                for df in results:
                    if df is not None:
                        frames.append(df)

        if not frames:
            log.warning("no_data", symbol=symbol)
            return None

        combined = pd.concat(frames, ignore_index=True)
        combined = self._process_df(combined)

        combined.to_parquet(output)
        log.info("saved_parquet", symbol=symbol, rows=len(combined),
                 path=str(output), start=str(combined.index[0]),
                 end=str(combined.index[-1]))
        return output

    async def download_all(
        self,
        symbols: list[str],
        timeframe: str = "15m",
        months: int = 12,
        force: bool = False,
        max_concurrent: int = 3,
    ) -> dict[str, Path]:
        """Download data for multiple symbols with concurrency limit.

        Returns dict of symbol → parquet path for successful downloads.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: dict[str, Path] = {}

        async def _dl(sym: str) -> None:
            async with semaphore:
                path = await self.download_symbol(sym, timeframe, months, force)
                if path:
                    results[sym] = path

        await asyncio.gather(*[_dl(s) for s in symbols])
        log.info("download_all_complete", total=len(symbols),
                 success=len(results), failed=len(symbols) - len(results))
        return results

    def load_symbol(self, symbol: str, timeframe: str = "15m") -> pd.DataFrame | None:
        """Load cached parquet data for a symbol."""
        path = self._output_path(symbol, timeframe)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df.attrs["symbol"] = symbol
        return df
