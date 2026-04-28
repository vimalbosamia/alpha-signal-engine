"""
Unit tests for strategy framework and three strategy implementations.

Coverage:
  - BaseStrategy.is_eligible (3 cases)
  - HammerReversalStrategy (4 cases)
  - ResistanceBreakoutStrategy (3 cases)
  - PullbackContinuationStrategy (3 cases)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from libs.analysis.levels.engine import KeyLevel
from libs.analysis.regime.engine import RegimeAnalysis
from libs.analysis.structure.engine import MarketStructure, StructurePoint
from libs.analysis.volume.engine import VolumeContext
from libs.core.models.domain import (
    AssetClass,
    DataQualityReport,
    DataQualityStatus,
    MarketRegime,
    PatternBias,
    SessionState,
    SessionType,
    SignalAction,
    TrendDirection,
)
from libs.strategies.breakout.resistance_breakout import ResistanceBreakoutStrategy
from libs.strategies.continuation.pullback import PullbackContinuationStrategy
from libs.strategies.reversal.hammer_reversal import HammerReversalStrategy


# ── DataFrame factory helpers ─────────────────────────────────────────────────


def make_df(
    closes: list[float],
    volumes: list[float] | None = None,
    with_vwap: bool = False,
) -> pd.DataFrame:
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    vols = np.array(volumes or [1000.0] * n, dtype=float)

    df = pd.DataFrame(
        {
            "open": closes_arr * 0.999,
            "high": closes_arr * 1.005,
            "low": closes_arr * 0.995,
            "close": closes_arr,
            "volume": vols,
        },
        index=pd.date_range("2024-01-02", periods=n, freq="5min", tz="UTC"),
    )

    df["body_size"] = (df["close"] - df["open"]).abs()
    df["total_range"] = df["high"] - df["low"]
    df["body_pct"] = df["body_size"] / df["total_range"].replace(0, np.nan)

    avg_vol = df["volume"].rolling(20, min_periods=1).mean()
    df["relative_volume"] = df["volume"] / avg_vol.replace(0, np.nan)

    if with_vwap:
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


def make_uptrend_df(n: int = 60) -> pd.DataFrame:
    closes = np.linspace(100.0, 120.0, n).tolist()
    return make_df(closes, with_vwap=True)


def make_downtrend_df(n: int = 60) -> pd.DataFrame:
    closes = np.linspace(120.0, 100.0, n).tolist()
    return make_df(closes)


def make_ranging_df(n: int = 60) -> pd.DataFrame:
    closes = [100.0 + 2.0 * math.sin(i * 0.5) for i in range(n)]
    return make_df(closes)


# ── Domain object helpers ─────────────────────────────────────────────────────


def make_uptrend() -> MarketStructure:
    return MarketStructure(
        trend=TrendDirection.UPTREND,
        points=[],
        trend_strength=0.8,
        break_of_structure=False,
        swing_high=120.0,
        swing_low=100.0,
    )


def make_downtrend() -> MarketStructure:
    return MarketStructure(
        trend=TrendDirection.DOWNTREND,
        points=[],
        trend_strength=0.7,
        break_of_structure=False,
        swing_high=110.0,
        swing_low=90.0,
    )


def make_support(price: float) -> KeyLevel:
    return KeyLevel(
        price=price,
        level_type="support",
        strength=0.8,
        asset_class=AssetClass.STOCK,
    )


def make_resistance(price: float) -> KeyLevel:
    return KeyLevel(
        price=price,
        level_type="resistance",
        strength=0.8,
        asset_class=AssetClass.STOCK,
    )


def make_session(tradable: bool = True) -> SessionState:
    return SessionState(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        session_type=SessionType.REGULAR,
        is_tradable=tradable,
        quality_score=1.0,
    )


def make_quality(status: DataQualityStatus = DataQualityStatus.CLEAN) -> DataQualityReport:
    return DataQualityReport(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        status=status,
        rows_checked=100,
    )


def make_high_volume_context(rel_vol: float = 2.0) -> VolumeContext:
    return VolumeContext(
        relative_volume=rel_vol,
        volume_trend="rising",
        breakout_confirmed=True,
        reversal_confirmed=False,
        divergence=False,
        effort_vs_result="normal",
        above_vwap=True,
        score=0.8,
    )


# ── BaseStrategy tests ────────────────────────────────────────────────────────


class TestBaseStrategyEligibility:
    """Tests for BaseStrategy.is_eligible via a concrete implementation."""

    def _strategy(self) -> HammerReversalStrategy:
        return HammerReversalStrategy()

    # Test 1
    def test_not_eligible_when_asset_class_not_supported(self) -> None:
        """is_eligible returns False when asset class is not in supported list."""
        # Arrange
        strategy = self._strategy()

        # HammerReversalStrategy supports STOCK and CRYPTO only.
        # Create a fake unsupported class by monkey-patching supported list.
        # Use FOREX (not defined in AssetClass enum) — simulate by passing an
        # unsupported string value. Since AssetClass is a StrEnum we must use
        # a value not in the strategy's list; CRYPTO vs STOCK covers this if
        # we override the property.
        class _StockOnlyStrategy(HammerReversalStrategy):
            @property
            def supported_asset_classes(self) -> list[AssetClass]:
                return [AssetClass.STOCK]

        stock_only = _StockOnlyStrategy()

        # Act
        result = stock_only.is_eligible(
            asset_class=AssetClass.CRYPTO,
            session=make_session(tradable=True),
            quality=make_quality(DataQualityStatus.CLEAN),
        )

        # Assert
        assert result is False

    # Test 2
    def test_not_eligible_when_session_not_tradable(self) -> None:
        """is_eligible returns False when session.is_tradable is False."""
        # Arrange
        strategy = self._strategy()
        session = make_session(tradable=False)

        # Act
        result = strategy.is_eligible(
            asset_class=AssetClass.STOCK,
            session=session,
            quality=make_quality(),
        )

        # Assert
        assert result is False

    # Test 3
    def test_not_eligible_when_quality_is_blocked(self) -> None:
        """is_eligible returns False when DataQualityStatus is BLOCKED."""
        # Arrange
        strategy = self._strategy()
        quality = make_quality(DataQualityStatus.BLOCKED)

        # Act
        result = strategy.is_eligible(
            asset_class=AssetClass.STOCK,
            session=make_session(tradable=True),
            quality=quality,
        )

        # Assert
        assert result is False


# ── HammerReversalStrategy tests ──────────────────────────────────────────────


class TestHammerReversalStrategy:
    """Unit tests for HammerReversalStrategy."""

    def _strategy(self) -> HammerReversalStrategy:
        return HammerReversalStrategy()

    def _common_kwargs(self, df: pd.DataFrame, **overrides) -> dict:
        return {
            "symbol": "AAPL",
            "asset_class": AssetClass.STOCK,
            "df": df,
            "df_htf": None,
            "session": make_session(tradable=True),
            "quality": make_quality(DataQualityStatus.CLEAN),
            "structure": make_downtrend(),
            "levels": [make_support(df["close"].iloc[-1])],
            "volume": None,
            "regime": None,
            **overrides,
        }

    # Test 4
    def test_returns_none_when_df_too_short(self) -> None:
        """Returns None when df has fewer than min_bars_required rows."""
        # Arrange
        strategy = self._strategy()
        df = make_df([100.0] * 10)  # only 10 bars, need 50

        # Act
        result = strategy.generate_candidate(**self._common_kwargs(df))

        # Assert
        assert result is None

    # Test 5
    def test_no_exception_on_valid_downtrend_df_with_support(self) -> None:
        """Does not raise on valid downtrend df; returns candidate or None."""
        # Arrange
        strategy = self._strategy()
        df = make_downtrend_df(n=60)

        # Last bar: force hammer shape — long lower wick
        last_idx = df.index[-1]
        close_val = float(df["close"].iloc[-1])
        df.loc[last_idx, "open"] = close_val * 1.001   # bearish body above
        df.loc[last_idx, "high"] = close_val * 1.002
        df.loc[last_idx, "low"] = close_val * 0.970    # long lower wick
        df.loc[last_idx, "close"] = close_val

        support_price = close_val  # support at last close

        # Act — must not raise
        try:
            result = strategy.generate_candidate(
                **self._common_kwargs(df, levels=[make_support(support_price)])
            )
        except Exception as exc:
            pytest.fail(f"generate_candidate raised unexpectedly: {exc}")

        # Assert — result is either a candidate or None; both are valid
        assert result is None or result.proposed_action == SignalAction.BUY

    # Test 6
    def test_candidate_stop_loss_below_entry_when_returned(self) -> None:
        """When a SignalCandidate is returned, stop_loss < entry_zone_low."""
        # Arrange
        strategy = self._strategy()
        df = make_downtrend_df(n=60)

        last_idx = df.index[-1]
        close_val = float(df["close"].iloc[-1])
        # Force a strong hammer: very long lower wick, tiny body, tiny upper wick
        df.loc[last_idx, "open"] = close_val + 0.05
        df.loc[last_idx, "close"] = close_val
        df.loc[last_idx, "high"] = close_val + 0.06
        df.loc[last_idx, "low"] = close_val - 1.50   # long lower wick

        support_price = close_val

        # Act
        result = strategy.generate_candidate(
            **self._common_kwargs(df, levels=[make_support(support_price)])
        )

        # Assert
        if result is not None:
            assert result.stop_loss < result.entry_zone_low, (
                f"stop_loss={result.stop_loss} should be below entry_zone_low={result.entry_zone_low}"
            )

    # Test 7
    def test_returns_none_when_session_not_tradable(self) -> None:
        """Returns None when session.is_tradable is False."""
        # Arrange
        strategy = self._strategy()
        df = make_downtrend_df(n=60)
        non_tradable_session = make_session(tradable=False)

        # Act
        result = strategy.generate_candidate(
            **self._common_kwargs(df, session=non_tradable_session)
        )

        # Assert
        assert result is None


# ── ResistanceBreakoutStrategy tests ─────────────────────────────────────────


class TestResistanceBreakoutStrategy:
    """Unit tests for ResistanceBreakoutStrategy."""

    def _strategy(self) -> ResistanceBreakoutStrategy:
        return ResistanceBreakoutStrategy()

    def _make_breakout_df(self, resistance: float, n: int = 40) -> pd.DataFrame:
        """DataFrame whose last close is clearly above resistance."""
        closes = np.linspace(resistance * 0.95, resistance * 1.005, n).tolist()
        vols = [1000.0] * n
        # Last bar: high volume breakout above resistance
        vols[-1] = 2000.0
        return make_df(closes, volumes=vols)

    # Test 8
    def test_returns_none_when_no_resistance_above_close(self) -> None:
        """Returns None when no resistance level is positioned near close."""
        # Arrange
        strategy = self._strategy()
        df = make_uptrend_df(n=40)
        last_close = float(df["close"].iloc[-1])

        # Place resistance far above close — beyond the search band
        far_resistance = make_resistance(last_close * 1.10)

        # Act
        result = strategy.generate_candidate(
            symbol="AAPL",
            asset_class=AssetClass.STOCK,
            df=df,
            df_htf=None,
            session=make_session(),
            quality=make_quality(),
            structure=make_uptrend(),
            levels=[far_resistance],
            volume=make_high_volume_context(rel_vol=2.0),
            regime=None,
        )

        # Assert
        assert result is None

    # Test 9
    def test_returns_candidate_on_clear_breakout_with_high_volume(self) -> None:
        """Returns a SignalCandidate when price breaks resistance with sufficient volume."""
        # Arrange
        strategy = self._strategy()
        resistance_price = 110.0

        # Build df where last close is above resistance * (1 + BREAKOUT_THRESHOLD_PCT)
        n = 40
        closes = np.linspace(100.0, resistance_price * 1.005, n).tolist()
        vols = [1000.0] * n
        vols[-1] = 2000.0
        df = make_df(closes, volumes=vols)

        last_close = float(df["close"].iloc[-1])
        assert last_close > resistance_price * (1 + strategy.BREAKOUT_THRESHOLD_PCT)

        # Resistance just below current close (was just broken)
        resistance_lvl = make_resistance(resistance_price)
        high_vol = make_high_volume_context(rel_vol=2.0)

        # Act
        result = strategy.generate_candidate(
            symbol="AAPL",
            asset_class=AssetClass.STOCK,
            df=df,
            df_htf=None,
            session=make_session(),
            quality=make_quality(),
            structure=make_uptrend(),
            levels=[resistance_lvl],
            volume=high_vol,
            regime=None,
        )

        # Assert
        assert result is not None
        assert result.proposed_action == SignalAction.BUY

    # Test 10
    def test_candidate_proposed_action_is_buy(self) -> None:
        """Confirmed breakout candidate has proposed_action == BUY."""
        # Arrange
        strategy = self._strategy()
        resistance_price = 110.0

        n = 40
        closes = np.linspace(100.0, resistance_price * 1.005, n).tolist()
        vols = [2000.0] * n
        df = make_df(closes, volumes=vols)

        last_close = float(df["close"].iloc[-1])
        # Verify breakout condition
        if last_close <= resistance_price * (1 + strategy.BREAKOUT_THRESHOLD_PCT):
            pytest.skip("Data setup did not produce a breakout close — skip.")

        resistance_lvl = make_resistance(resistance_price)
        high_vol = make_high_volume_context(rel_vol=2.0)

        # Act
        result = strategy.generate_candidate(
            symbol="AAPL",
            asset_class=AssetClass.STOCK,
            df=df,
            df_htf=None,
            session=make_session(),
            quality=make_quality(),
            structure=make_uptrend(),
            levels=[resistance_lvl],
            volume=high_vol,
            regime=None,
        )

        # Assert
        if result is not None:
            assert result.proposed_action == SignalAction.BUY


# ── PullbackContinuationStrategy tests ───────────────────────────────────────


class TestPullbackContinuationStrategy:
    """Unit tests for PullbackContinuationStrategy."""

    def _strategy(self) -> PullbackContinuationStrategy:
        return PullbackContinuationStrategy()

    def _common_kwargs(self, df: pd.DataFrame, **overrides) -> dict:
        return {
            "symbol": "AAPL",
            "asset_class": AssetClass.STOCK,
            "df": df,
            "df_htf": None,
            "session": make_session(tradable=True),
            "quality": make_quality(DataQualityStatus.CLEAN),
            "structure": make_uptrend(),
            "levels": None,
            "volume": None,
            "regime": None,
            **overrides,
        }

    # Test 11
    def test_returns_none_when_structure_is_downtrend(self) -> None:
        """Returns None when market structure indicates downtrend."""
        # Arrange
        strategy = self._strategy()
        df = make_uptrend_df(n=50)

        # Act
        result = strategy.generate_candidate(
            **self._common_kwargs(df, structure=make_downtrend())
        )

        # Assert
        assert result is None

    # Test 12
    def test_returns_none_when_no_vwap_column(self) -> None:
        """Returns None when df has no 'vwap' column."""
        # Arrange
        strategy = self._strategy()
        closes = np.linspace(100.0, 120.0, 50).tolist()
        df = make_df(closes, with_vwap=False)   # no vwap column

        # Act
        result = strategy.generate_candidate(
            **self._common_kwargs(df, structure=make_uptrend())
        )

        # Assert
        assert result is None

    # Test 13
    def test_returns_candidate_when_close_near_vwap_in_uptrend(self) -> None:
        """Returns a SignalCandidate when close is near VWAP in an uptrend."""
        # Arrange
        strategy = self._strategy()
        n = 50
        closes = np.linspace(100.0, 110.0, n).tolist()
        df = make_df(closes, with_vwap=True)

        # Force the last bar to be near VWAP and bullish
        last_idx = df.index[-1]
        vwap_val = float(df["vwap"].iloc[-2])  # use penultimate bar's VWAP as target
        # Set last bar so close is at vwap_val (within 0.5%) and close > open
        df.loc[last_idx, "open"] = vwap_val * 0.9995
        df.loc[last_idx, "close"] = vwap_val * 1.0001   # bullish, within 0.05% of vwap
        df.loc[last_idx, "high"] = vwap_val * 1.002
        df.loc[last_idx, "low"] = vwap_val * 0.998
        # Recalculate vwap for last bar (approximate)
        df.loc[last_idx, "vwap"] = vwap_val

        # Act
        result = strategy.generate_candidate(
            **self._common_kwargs(df, structure=make_uptrend())
        )

        # Assert
        assert result is not None
        assert result.proposed_action == SignalAction.BUY
        assert result.stop_loss < result.entry_zone_low
