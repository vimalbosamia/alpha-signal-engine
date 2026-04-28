"""
Integration tests for SignalPipeline.run_once().

Uses a mock provider — no real API calls are made.
"""
from __future__ import annotations

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from libs.core.models.domain import (
    AssetClass, Timeframe, SymbolMetadata, SignalOutput, SignalAction
)
from libs.data.providers.base import BaseDataProvider
from apps.signal_agent.pipeline import SignalPipeline
from libs.strategies.reversal.hammer_reversal import HammerReversalStrategy
from libs.strategies.breakout.resistance_breakout import ResistanceBreakoutStrategy
from libs.strategies.continuation.pullback import PullbackContinuationStrategy


def make_mock_provider(df: pd.DataFrame) -> BaseDataProvider:
    provider = MagicMock(spec=BaseDataProvider)
    provider.name = "mock"
    provider.asset_class = AssetClass.STOCK
    provider.get_candles = AsyncMock(return_value=df)
    return provider


def make_test_df(n: int = 100, trend: str = "up") -> pd.DataFrame:
    if trend == "up":
        closes = np.linspace(100, 120, n)
    else:
        closes = np.array([100 + 2 * np.sin(i * 0.3) for i in range(n)])
    closes = np.array(closes, dtype=float)
    df = pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": np.random.uniform(800, 1200, n),
        },
        index=pd.date_range(
            datetime.now(timezone.utc) - timedelta(hours=n),
            periods=n,
            freq="5min",
            tz="UTC",
        ),
    )
    return df


def make_pipeline(provider: BaseDataProvider) -> SignalPipeline:
    return SignalPipeline(
        provider=provider,
        strategies=[
            HammerReversalStrategy(),
            ResistanceBreakoutStrategy(),
            PullbackContinuationStrategy(),
        ],
    )


@pytest.mark.asyncio
async def test_pipeline_returns_list_no_exception_on_valid_data():
    """Pipeline returns a list (possibly empty) without raising on valid uptrend data."""
    df = make_test_df(n=100, trend="up")
    provider = make_mock_provider(df)
    pipeline = make_pipeline(provider)

    result = await pipeline.run_once("AAPL", AssetClass.STOCK, Timeframe.FIVE_MIN)

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_pipeline_returns_empty_list_when_provider_raises():
    """Pipeline returns empty list (not exception) when provider raises."""
    provider = MagicMock(spec=BaseDataProvider)
    provider.name = "mock"
    provider.asset_class = AssetClass.STOCK
    provider.get_candles = AsyncMock(side_effect=RuntimeError("network error"))

    pipeline = make_pipeline(provider)
    result = await pipeline.run_once("AAPL", AssetClass.STOCK, Timeframe.FIVE_MIN)

    assert result == []


@pytest.mark.asyncio
async def test_all_returned_items_are_signal_outputs():
    """All returned items are SignalOutput instances."""
    df = make_test_df(n=100, trend="up")
    provider = make_mock_provider(df)
    pipeline = make_pipeline(provider)

    result = await pipeline.run_once("AAPL", AssetClass.STOCK, Timeframe.FIVE_MIN)

    for item in result:
        assert isinstance(item, SignalOutput), f"Expected SignalOutput, got {type(item)}"


@pytest.mark.asyncio
async def test_pipeline_returns_empty_list_for_too_short_df():
    """Pipeline returns empty list (not exception) when df has fewer than 30 bars."""
    df = make_test_df(n=10, trend="up")
    provider = make_mock_provider(df)
    pipeline = make_pipeline(provider)

    result = await pipeline.run_once("AAPL", AssetClass.STOCK, Timeframe.FIVE_MIN)

    assert isinstance(result, list)
    # With only 10 bars, no strategy should produce signals (all require more bars)
    assert result == []
