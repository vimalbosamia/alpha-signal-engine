"""
Candle builder and multi-timeframe aggregator.

Responsibilities:
  1. Aggregate 1m base candles to any higher timeframe
  2. Compute derived candle metrics (body%, wicks, relative volume)
  3. Compute VWAP, anchored VWAP, and opening range
  4. Maintain a rolling candle cache per symbol/timeframe

Works identically for stocks and crypto — asset differences are
only in session filtering (handled upstream by the session manager).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import NamedTuple

import numpy as np
import pandas as pd

from libs.core.logging.logger import get_logger
from libs.core.models.domain import AssetClass, Candle, Timeframe

log = get_logger(__name__)

_OFFSET_MAP: dict[Timeframe, str] = {
    Timeframe.ONE_MIN: "1min",
    Timeframe.THREE_MIN: "3min",
    Timeframe.FIVE_MIN: "5min",
    Timeframe.FIFTEEN_MIN: "15min",
    Timeframe.THIRTY_MIN: "30min",
    Timeframe.ONE_HOUR: "1h",
    Timeframe.FOUR_HOUR: "4h",
    Timeframe.ONE_DAY: "1D",
    Timeframe.ONE_WEEK: "1W",
}


class CandleFeatures(NamedTuple):
    """Pre-computed structural features for a single candle row."""
    body_size: float
    total_range: float
    body_pct: float
    upper_wick: float
    lower_wick: float
    is_bullish: bool
    is_bearish: bool
    is_doji: bool
    relative_volume: float


class CandleBuilder:
    """
    Aggregates and enriches candle DataFrames.
    """

    def __init__(self, avg_volume_period: int = 20, max_cache_bars: int = 1000) -> None:
        self._avg_vol_period = avg_volume_period
        self._max_cache = max_cache_bars
        # Rolling cache: {(symbol, timeframe): DataFrame}
        self._cache: dict[tuple[str, str], pd.DataFrame] = defaultdict(pd.DataFrame)

    def aggregate(
        self,
        df_base: pd.DataFrame,
        target_tf: Timeframe,
        base_tf: Timeframe = Timeframe.ONE_MIN,
    ) -> pd.DataFrame:
        """
        Aggregate a lower-timeframe DataFrame to a higher timeframe.

        Returns enriched DataFrame with derived columns.
        """
        if target_tf == base_tf:
            return self.enrich(df_base)

        offset = _OFFSET_MAP.get(target_tf)
        if not offset:
            raise ValueError(f"Unsupported target timeframe: {target_tf}")

        df = df_base.resample(offset, label="left", closed="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(how="all")

        if "vwap" in df_base.columns:
            vol_price = (df_base["close"] * df_base["volume"]).resample(
                offset, label="left", closed="left"
            ).sum()
            vol_sum = df_base["volume"].resample(offset, label="left", closed="left").sum()
            df["vwap"] = vol_price / vol_sum.replace(0, np.nan)

        return self.enrich(df)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all derived candle columns to an OHLCV DataFrame.
        Returns a copy — does not mutate the input.
        """
        if df.empty:
            return df.copy()

        df = df.copy()
        df = df.dropna(subset=["open", "high", "low", "close"])

        o = df["open"]
        h = df["high"]
        l = df["low"]
        c = df["close"]

        body = (c - o).abs()
        rng = (h - l).replace(0, np.nan)

        df["body_size"] = body
        df["total_range"] = h - l
        df["body_pct"] = body / rng
        df["upper_wick"] = h - pd.concat([o, c], axis=1).max(axis=1)
        df["lower_wick"] = pd.concat([o, c], axis=1).min(axis=1) - l
        df["is_bullish"] = c > o
        df["is_bearish"] = c < o
        df["is_doji"] = df["body_pct"].fillna(0) < 0.10

        if "volume" in df.columns:
            avg_vol = df["volume"].rolling(self._avg_vol_period, min_periods=1).mean()
            df["relative_volume"] = df["volume"] / avg_vol.replace(0, np.nan)
        else:
            df["relative_volume"] = 1.0

        # VWAP (session-level if intraday; simple rolling if not tagged)
        if "vwap" not in df.columns and "volume" in df.columns:
            typical = (h + l + c) / 3
            cum_tp_vol = (typical * df["volume"]).cumsum()
            cum_vol = df["volume"].cumsum()
            df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

        return df

    def get_features(self, df: pd.DataFrame, idx: int = -1) -> CandleFeatures:
        """Return CandleFeatures for bar at position idx."""
        row = df.iloc[idx]
        return CandleFeatures(
            body_size=float(row.get("body_size", abs(row["close"] - row["open"]))),
            total_range=float(row.get("total_range", row["high"] - row["low"])),
            body_pct=float(row.get("body_pct", 0)),
            upper_wick=float(row.get("upper_wick", 0)),
            lower_wick=float(row.get("lower_wick", 0)),
            is_bullish=bool(row.get("is_bullish", row["close"] > row["open"])),
            is_bearish=bool(row.get("is_bearish", row["close"] < row["open"])),
            is_doji=bool(row.get("is_doji", False)),
            relative_volume=float(row.get("relative_volume", 1.0)),
        )

    def compute_opening_range(
        self,
        df: pd.DataFrame,
        minutes: int = 30,
    ) -> tuple[float, float] | None:
        """
        Compute the opening range high/low for the first N minutes.
        Returns (or_high, or_low) or None if insufficient data.
        Used for stocks where the first 30m sets the daily OR.
        """
        if df.empty or len(df) < 1:
            return None
        # Assume df has DatetimeIndex UTC; use first `minutes` bars
        first_bar = df.index[0]
        cutoff = first_bar + pd.Timedelta(minutes=minutes)
        or_bars = df[df.index < cutoff]
        if or_bars.empty:
            return None
        return float(or_bars["high"].max()), float(or_bars["low"].min())

    def push_to_cache(self, symbol: str, timeframe: Timeframe, df: pd.DataFrame) -> None:
        key = (symbol, timeframe.value)
        existing = self._cache[key]
        if existing.empty:
            self._cache[key] = df.iloc[-self._max_cache:]
        else:
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            self._cache[key] = combined.sort_index().iloc[-self._max_cache:]

    def get_from_cache(self, symbol: str, timeframe: Timeframe) -> pd.DataFrame:
        return self._cache.get((symbol, timeframe.value), pd.DataFrame())
