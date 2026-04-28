"""
Unit tests for HTF (higher timeframe) data fetching and bias logic in SignalPipeline.

Tests:
  1. _htf_timeframe mapping
  2. HTF fetch failure is non-fatal
  3. HTF bias overrides primary bias on candidate
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from apps.signal_agent.pipeline import SignalPipeline
from libs.analysis.structure.engine import MarketStructure
from libs.core.models.domain import (
    AssetClass,
    DataQualityReport,
    DataQualityStatus,
    MarketRegime,
    SessionState,
    SessionType,
    SignalAction,
    SignalCandidate,
    Timeframe,
    TrendDirection,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_minimal_df(n: int = 20) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with n rows."""
    return pd.DataFrame({
        "open":   [100.0] * n,
        "high":   [101.0] * n,
        "low":    [99.0]  * n,
        "close":  [100.5] * n,
        "volume": [1000.0] * n,
    })


def _make_candidate(higher_tf_bias: TrendDirection = TrendDirection.RANGING) -> SignalCandidate:
    quality = DataQualityReport(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        status=DataQualityStatus.CLEAN,
        rows_checked=100,
    )
    session = SessionState(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        session_type=SessionType.REGULAR,
        is_tradable=True,
        quality_score=1.0,
    )
    return SignalCandidate(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        strategy_name="test_strategy",
        proposed_action=SignalAction.BUY,
        timeframe=Timeframe.FIVE_MIN,
        higher_tf_bias=higher_tf_bias,
        entry_zone_low=100.0,
        entry_zone_high=100.5,
        stop_loss=98.0,
        take_profit_1=105.0,
        regime=MarketRegime.TRENDING_UP,
        quality=quality,
        session=session,
        raw_features={},
    )


# ── test 1: HTF timeframe mapping ─────────────────────────────────────────────

class TestHtfTimerframeMapping:
    def test_five_min_maps_to_fifteen_min(self):
        result = SignalPipeline._htf_timeframe(Timeframe.FIVE_MIN)
        assert result == Timeframe.FIFTEEN_MIN

    def test_one_min_maps_to_five_min(self):
        assert SignalPipeline._htf_timeframe(Timeframe.ONE_MIN) == Timeframe.FIVE_MIN

    def test_fifteen_min_maps_to_one_hour(self):
        assert SignalPipeline._htf_timeframe(Timeframe.FIFTEEN_MIN) == Timeframe.ONE_HOUR

    def test_thirty_min_maps_to_four_hour(self):
        assert SignalPipeline._htf_timeframe(Timeframe.THIRTY_MIN) == Timeframe.FOUR_HOUR

    def test_one_hour_maps_to_four_hour(self):
        assert SignalPipeline._htf_timeframe(Timeframe.ONE_HOUR) == Timeframe.FOUR_HOUR

    def test_four_hour_maps_to_one_day(self):
        assert SignalPipeline._htf_timeframe(Timeframe.FOUR_HOUR) == Timeframe.ONE_DAY

    def test_one_day_maps_to_one_week(self):
        assert SignalPipeline._htf_timeframe(Timeframe.ONE_DAY) == Timeframe.ONE_WEEK

    def test_one_week_returns_none(self):
        """ONE_WEEK has no higher timeframe — must return None."""
        result = SignalPipeline._htf_timeframe(Timeframe.ONE_WEEK)
        assert result is None


# ── test 2: HTF fetch failure is non-fatal ────────────────────────────────────

class TestHtfFetchFailureIsNonfatal:
    """Pipeline must not crash or return empty when HTF candle fetch raises."""

    @pytest.fixture()
    def pipeline(self):
        provider = MagicMock()
        strategy = MagicMock()
        strategy.is_eligible.return_value = True
        strategy.min_bars_required = 1
        strategy.name = "test_strategy"
        strategy.generate_candidate.return_value = None  # no signal; simplest path

        return SignalPipeline(
            provider=provider,
            strategies=[strategy],
        )

    @pytest.mark.asyncio
    async def test_htf_failure_does_not_crash_pipeline(self, pipeline):
        primary_df = _make_minimal_df(30)

        call_count = {"n": 0}

        async def side_effect(symbol, tf, start, end):
            call_count["n"] += 1
            if tf == Timeframe.FIVE_MIN:
                # primary call succeeds
                return primary_df
            # HTF call raises
            raise RuntimeError("HTF endpoint unavailable")

        pipeline._provider.get_candles = side_effect

        # Patch away engines that need real enriched data
        with (
            patch.object(pipeline._builder, "enrich", side_effect=lambda df: df),
            patch.object(pipeline._validator, "validate", return_value=MagicMock(is_safe=True, errors=[])),
            patch("libs.data.session.session_manager.get_session_manager") as mock_sm,
        ):
            mock_sm.return_value.get_state.return_value = MagicMock(is_tradable=True)

            # Patch all engines to avoid needing real enriched DataFrames
            with (
                patch.object(pipeline._structure, "analyze", return_value=MagicMock(trend=TrendDirection.RANGING, spec=MarketStructure)),
                patch.object(pipeline._levels, "analyze", return_value=MagicMock()),
                patch.object(pipeline._regime, "analyze", return_value=MagicMock()),
                patch.object(pipeline._indicators, "compute", return_value=MagicMock()),
            ):
                outputs = await pipeline.run_once("AAPL", AssetClass.STOCK, Timeframe.FIVE_MIN)

        # Pipeline must complete (not raise) and HTF call must have been attempted
        assert call_count["n"] == 2   # primary + HTF attempt
        assert isinstance(outputs, list)


# ── test 3: HTF bias overrides primary bias ───────────────────────────────────

class TestHtfBiasOverridesPrimaryBias:
    """
    When primary structure is RANGING but HTF structure is UPTREND,
    the candidate's higher_tf_bias must be UPTREND after the pipeline override.
    """

    @pytest.mark.asyncio
    async def test_htf_bias_overrides_primary_bias(self):
        primary_df = _make_minimal_df(30)
        htf_df = _make_minimal_df(20)

        # Candidate initially carries RANGING bias (from primary)
        candidate = _make_candidate(higher_tf_bias=TrendDirection.RANGING)

        provider = MagicMock()

        async def get_candles(symbol, tf, start, end):
            if tf == Timeframe.FIVE_MIN:
                return primary_df
            return htf_df

        provider.get_candles = get_candles

        strategy = MagicMock()
        strategy.is_eligible.return_value = True
        strategy.min_bars_required = 1
        strategy.name = "test_strategy"
        strategy.generate_candidate.return_value = candidate

        pipeline = SignalPipeline(provider=provider, strategies=[strategy])

        primary_structure = MagicMock(spec=MarketStructure)
        primary_structure.trend = TrendDirection.RANGING

        htf_structure = MagicMock(spec=MarketStructure)
        htf_structure.trend = TrendDirection.UPTREND

        emitted_candidates: list[SignalCandidate] = []

        def capture_score(cand, *args, **kwargs):
            emitted_candidates.append(cand)
            return MagicMock(final_score=0.8, action=SignalAction.BUY)

        with (
            patch.object(pipeline._builder, "enrich", side_effect=lambda df: df),
            patch.object(pipeline._validator, "validate", return_value=MagicMock(is_safe=True, errors=[])),
            patch("libs.data.session.session_manager.get_session_manager") as mock_sm,
            patch.object(
                pipeline._structure,
                "analyze",
                side_effect=lambda df: htf_structure if df is htf_df else primary_structure,
            ),
            patch.object(pipeline._levels, "analyze", return_value=MagicMock()),
            patch.object(pipeline._regime, "analyze", return_value=MagicMock()),
            patch.object(pipeline._indicators, "compute", return_value=MagicMock()),
            patch.object(pipeline._volume, "analyze", return_value=MagicMock()),
            patch.object(pipeline._risk, "assess", return_value=MagicMock(passed=True, score=0.8)),
            patch.object(pipeline._confluence, "score", side_effect=capture_score),
            patch.object(pipeline._emitter, "emit", return_value=MagicMock(
                action=SignalAction.NO_TRADE, warnings=[], to_display=lambda: {}
            )),
            patch.object(pipeline._audit, "record_signal", new_callable=AsyncMock),
            patch("libs.data.storage.db.get_session_factory"),
            patch("libs.monitoring.outcome_tracker.is_strategy_muted", return_value=False),
        ):
            mock_sm.return_value.get_state.return_value = MagicMock(is_tradable=True)

            # Patch DEFAULT_DETECTORS to avoid needing real enriched df
            with patch("apps.signal_agent.pipeline.DEFAULT_DETECTORS", []):
                await pipeline.run_once("AAPL", AssetClass.STOCK, Timeframe.FIVE_MIN)

        assert len(emitted_candidates) == 1, "Expected exactly one candidate to reach confluence scoring"
        assert emitted_candidates[0].higher_tf_bias == TrendDirection.UPTREND, (
            f"Expected UPTREND from HTF, got {emitted_candidates[0].higher_tf_bias}"
        )
