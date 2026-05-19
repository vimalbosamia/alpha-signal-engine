"""
Failed Breakout Reversal Strategy.

Logic:
  Bull trap (SELL): Price broke above the 20-bar high 1-5 bars ago but the
  current close is back below that level. High volume on the failed breakout
  bar (> 1.5x average). Stop above the failed high; TP = ATR*3.

  Bear trap (BUY): Price broke below the 20-bar low 1-5 bars ago but the
  current close is back above that level. Same volume requirement.
  Stop below the failed low; TP = ATR*3.

Design rules:
  - Never raises in generate_candidate — returns None on any data issue.
  - No execution logic — produces signal candidates only.
  - All thresholds are named constants.
"""
from __future__ import annotations

import pandas as pd

from libs.core.models.domain import (
    AssetClass,
    DataQualityReport,
    MarketRegime,
    SessionState,
    SignalAction,
    SignalCandidate,
    Timeframe,
    TrendDirection,
)
from libs.analysis.indicators.engine import IndicatorSnapshot
from libs.analysis.levels.engine import KeyLevel
from libs.analysis.regime.engine import RegimeAnalysis
from libs.analysis.structure.engine import MarketStructure
from libs.analysis.volume.engine import VolumeContext
from libs.strategies.base.strategy import BaseStrategy


class FailedBreakoutReversalStrategy(BaseStrategy):
    """Reversal trade against a failed breakout (bull or bear trap)."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "failed_breakout_reversal"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 25

    # ── Thresholds ─────────────────────────────────────────────────────────────

    LEVEL_LOOKBACK: int = 20            # bars used to define swing high/low
    FAILURE_MIN_BARS_AGO: int = 1       # failed bar happened at least this many bars ago
    FAILURE_MAX_BARS_AGO: int = 5       # but no more than this many bars ago
    MIN_FAILED_REL_VOL: float = 1.5    # failed bar must have been high volume
    TP_ATR_MULTIPLIER: float = 3.0     # take-profit distance from entry
    BASE_CONFIDENCE: float = 0.58

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_candidate(
        self,
        symbol: str,
        asset_class: AssetClass,
        df: pd.DataFrame,
        df_htf: pd.DataFrame | None,
        session: SessionState | None,
        quality: DataQualityReport | None,
        structure: MarketStructure | None,
        levels: list[KeyLevel] | None,
        volume: VolumeContext | None,
        regime: RegimeAnalysis | None,
        indicators: IndicatorSnapshot | None = None,
    ) -> SignalCandidate | None:
        try:
            # 1. Pre-flight eligibility
            if not self.is_eligible(asset_class, session, quality):
                return None

            # 2. Require minimum bars
            if df is None or len(df) < self.min_bars_required:
                return None

            atr = self._get_atr(df, indicators)
            if atr is None or atr <= 0.0:
                return None

            last_close = float(df.iloc[-1]["close"])
            higher_tf_bias = structure.trend if structure is not None else TrendDirection.UNKNOWN

            # 3a. Bull trap → SELL
            sell_result = self._check_bull_trap(df, last_close, atr)
            if sell_result is not None:
                failed_high, entry_low, entry_high, stop_loss, take_profit_1 = sell_result
                return SignalCandidate(
                    symbol=symbol,
                    asset_class=asset_class,
                    strategy_name=self.name,
                    proposed_action=SignalAction.SELL,
                    timeframe=Timeframe.FIFTEEN_MIN,
                    higher_tf_bias=higher_tf_bias,
                    entry_zone_low=entry_low,
                    entry_zone_high=entry_high,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                    session=session,
                    quality=quality,
                    raw_features={
                        "failed_level": failed_high,
                        "last_close": last_close,
                        "atr": atr,
                        "confidence": self.BASE_CONFIDENCE,
                        "setup": "bull_trap",
                    },
                )

            # 3b. Bear trap → BUY
            buy_result = self._check_bear_trap(df, last_close, atr)
            if buy_result is not None:
                failed_low, entry_low, entry_high, stop_loss, take_profit_1 = buy_result
                return SignalCandidate(
                    symbol=symbol,
                    asset_class=asset_class,
                    strategy_name=self.name,
                    proposed_action=SignalAction.BUY,
                    timeframe=Timeframe.FIFTEEN_MIN,
                    higher_tf_bias=higher_tf_bias,
                    entry_zone_low=entry_low,
                    entry_zone_high=entry_high,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                    session=session,
                    quality=quality,
                    raw_features={
                        "failed_level": failed_low,
                        "last_close": last_close,
                        "atr": atr,
                        "confidence": self.BASE_CONFIDENCE,
                        "setup": "bear_trap",
                    },
                )

            return None

        except Exception:  # noqa: BLE001 — never propagate
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _check_bull_trap(
        self,
        df: pd.DataFrame,
        last_close: float,
        atr: float,
    ) -> tuple[float, float, float, float, float] | None:
        """
        Detect a bull trap: broke above 20-bar high recently but now back below.

        Returns (failed_high, entry_low, entry_high, stop_loss, take_profit_1)
        or None.
        """
        n = len(df)

        for bars_ago in range(self.FAILURE_MIN_BARS_AGO, self.FAILURE_MAX_BARS_AGO + 1):
            failed_idx = n - 1 - bars_ago
            if failed_idx < self.LEVEL_LOOKBACK:
                break

            # 20-bar high just before the failed breakout bar
            pre_failed_df = df.iloc[failed_idx - self.LEVEL_LOOKBACK:failed_idx]
            swing_high = float(pre_failed_df["high"].max())

            failed_bar = df.iloc[failed_idx]
            failed_bar_high = float(failed_bar["high"])
            failed_bar_close = float(failed_bar["close"])

            # Failed bar must have spiked above and then closed below the swing high,
            # or closed above but current bar is already back below
            spiked_above = failed_bar_high > swing_high
            if not spiked_above:
                continue

            # Current close must be back below the swing high (trap confirmed)
            if last_close >= swing_high:
                continue

            # Volume check on the failed bar
            if not self._bar_has_high_volume(df, failed_idx):
                continue

            # Build trade parameters (short/sell)
            entry_low = last_close * 0.999
            entry_high = last_close * 1.001
            # Stop above the failed bar's high
            stop_loss = failed_bar_high + atr * 0.5
            if stop_loss <= last_close:
                continue
            take_profit_1 = last_close - atr * self.TP_ATR_MULTIPLIER
            if take_profit_1 <= 0.0:
                continue

            return (swing_high, entry_low, entry_high, stop_loss, take_profit_1)

        return None

    def _check_bear_trap(
        self,
        df: pd.DataFrame,
        last_close: float,
        atr: float,
    ) -> tuple[float, float, float, float, float] | None:
        """
        Detect a bear trap: broke below 20-bar low recently but now back above.

        Returns (failed_low, entry_low, entry_high, stop_loss, take_profit_1)
        or None.
        """
        n = len(df)

        for bars_ago in range(self.FAILURE_MIN_BARS_AGO, self.FAILURE_MAX_BARS_AGO + 1):
            failed_idx = n - 1 - bars_ago
            if failed_idx < self.LEVEL_LOOKBACK:
                break

            # 20-bar low just before the failed breakdown bar
            pre_failed_df = df.iloc[failed_idx - self.LEVEL_LOOKBACK:failed_idx]
            swing_low = float(pre_failed_df["low"].min())

            failed_bar = df.iloc[failed_idx]
            failed_bar_low = float(failed_bar["low"])

            # Failed bar must have spiked below the swing low
            spiked_below = failed_bar_low < swing_low
            if not spiked_below:
                continue

            # Current close must be back above the swing low (trap confirmed)
            if last_close <= swing_low:
                continue

            # Volume check on the failed bar
            if not self._bar_has_high_volume(df, failed_idx):
                continue

            # Build trade parameters (long/buy)
            entry_low = last_close * 0.9995
            entry_high = last_close * 1.0005
            # Stop below the failed bar's low
            stop_loss = failed_bar_low - atr * 0.5
            if stop_loss <= 0.0 or stop_loss >= last_close:
                continue
            take_profit_1 = last_close + atr * self.TP_ATR_MULTIPLIER

            return (swing_low, entry_low, entry_high, stop_loss, take_profit_1)

        return None

    def _bar_has_high_volume(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Return True if the bar at `idx` has relative_volume >= MIN_FAILED_REL_VOL,
        or if relative_volume is not available (permissive fallback).
        """
        if "relative_volume" not in df.columns:
            return True  # can't confirm, allow through
        rel_vol = df.iloc[idx]["relative_volume"]
        if pd.isna(rel_vol):
            return True
        return float(rel_vol) >= self.MIN_FAILED_REL_VOL

    def _get_atr(
        self,
        df: pd.DataFrame,
        indicators: IndicatorSnapshot | None,
    ) -> float | None:
        """Return ATR from snapshot if available, else compute from df."""
        if indicators is not None and indicators.atr is not None:
            return indicators.atr
        if len(df) < 14:
            return None
        tr = (df["high"] - df["low"]).abs().rolling(14).mean().iloc[-1]
        import math
        if tr is not None and not math.isnan(float(tr)):
            return float(tr)
        return None
