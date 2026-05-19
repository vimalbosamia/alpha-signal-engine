"""
SMA Crossover Strategy.

Logic:
  - BUY  when SMA20 crosses above SMA50 (golden cross).
  - SELL when SMA20 crosses below SMA50 (death cross).

Uses ATR-based stop (1.5x) and 3:1 reward-to-risk target.

Design rules:
  - Never raises in generate_candidate — returns None on any data issue.
  - No execution logic — produces signal candidates only.
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


class SMACrossoverStrategy(BaseStrategy):
    """SMA(20) / SMA(50) crossover trend-following strategy."""

    FAST_PERIOD: int = 20
    SLOW_PERIOD: int = 50
    ATR_STOP_MULTIPLIER: float = 1.5
    DEFAULT_RR: float = 3.0
    CONFIDENCE: float = 0.60

    @property
    def name(self) -> str:
        return "sma_crossover"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 55

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
            if not self.is_eligible(asset_class, session, quality):
                return None
            if df is None or len(df) < self.min_bars_required:
                return None

            close = df["close"]

            # Compute full SMA series for crossover detection (need prev bar).
            sma_fast_series = close.rolling(self.FAST_PERIOD).mean()
            sma_slow_series = close.rolling(self.SLOW_PERIOD).mean()

            if len(sma_fast_series) < 2:
                return None
            if sma_fast_series.isna().iloc[-1] or sma_slow_series.isna().iloc[-1]:
                return None
            if sma_fast_series.isna().iloc[-2] or sma_slow_series.isna().iloc[-2]:
                return None

            # Prefer snapshot for current-bar values (consistent with pipeline).
            fast_now = (
                indicators.sma_20
                if (indicators and indicators.sma_20 is not None)
                else float(sma_fast_series.iloc[-1])
            )
            slow_now = (
                indicators.sma_50
                if (indicators and indicators.sma_50 is not None)
                else float(sma_slow_series.iloc[-1])
            )
            fast_prev = float(sma_fast_series.iloc[-2])
            slow_prev = float(sma_slow_series.iloc[-2])

            last_close = float(df["close"].iloc[-1])
            atr_dist = self._atr_stop(df, self.ATR_STOP_MULTIPLIER)
            if atr_dist <= 0.0:
                return None

            # ── BUY: SMA20 crosses above SMA50 ──────────────────────────────────
            golden_cross = fast_prev <= slow_prev and fast_now > slow_now
            if golden_cross:
                stop_loss = last_close - atr_dist
                if stop_loss <= 0.0:
                    return None
                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0
                if stop_loss >= entry_mid:
                    return None
                stop_distance = entry_mid - stop_loss
                take_profit_1 = entry_mid + self.DEFAULT_RR * stop_distance
                take_profit_2 = entry_mid + self.DEFAULT_RR * 1.5 * stop_distance

                return SignalCandidate(
                    symbol=symbol,
                    asset_class=asset_class,
                    strategy_name=self.name,
                    proposed_action=SignalAction.BUY,
                    timeframe=Timeframe.FIFTEEN_MIN,
                    higher_tf_bias=self._bias(structure),
                    entry_zone_low=entry_low,
                    entry_zone_high=entry_high,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                    regime=regime.regime if regime else MarketRegime.UNKNOWN,
                    session=session,
                    quality=quality,
                    raw_features={
                        "sma_fast": fast_now,
                        "sma_slow": slow_now,
                        "cross": "golden",
                        "confidence": self.CONFIDENCE,
                    },
                )

            # ── SELL: SMA20 crosses below SMA50 ─────────────────────────────────
            death_cross = fast_prev >= slow_prev and fast_now < slow_now
            if death_cross:
                stop_loss = last_close + atr_dist
                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0
                if stop_loss <= entry_mid:
                    return None
                stop_distance = stop_loss - entry_mid
                take_profit_1 = entry_mid - self.DEFAULT_RR * stop_distance
                if take_profit_1 <= 0.0:
                    return None
                take_profit_2 = entry_mid - self.DEFAULT_RR * 1.5 * stop_distance
                if take_profit_2 <= 0.0:
                    take_profit_2 = None

                return SignalCandidate(
                    symbol=symbol,
                    asset_class=asset_class,
                    strategy_name=self.name,
                    proposed_action=SignalAction.SELL,
                    timeframe=Timeframe.FIFTEEN_MIN,
                    higher_tf_bias=self._bias(structure),
                    entry_zone_low=entry_low,
                    entry_zone_high=entry_high,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                    regime=regime.regime if regime else MarketRegime.UNKNOWN,
                    session=session,
                    quality=quality,
                    raw_features={
                        "sma_fast": fast_now,
                        "sma_slow": slow_now,
                        "cross": "death",
                        "confidence": self.CONFIDENCE,
                    },
                )

        except Exception:
            return None

        return None

    @staticmethod
    def _bias(structure: MarketStructure | None) -> TrendDirection:
        return structure.trend if structure else TrendDirection.UNKNOWN
