"""
VWAP Reclaim Strategy.

Logic:
  - BUY  when price crosses above VWAP from below with volume > 1.2x average.
  - SELL when price crosses below VWAP from above with volume > 1.2x average.

Cross detection: close > vwap AND prev_close <= prev_vwap (BUY),
                 close < vwap AND prev_close >= prev_vwap (SELL).

Stop: ATR*1.5 below VWAP (BUY) / above VWAP (SELL).
TP  : ATR*3 above/below entry.

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


class VWAPReclaimStrategy(BaseStrategy):
    """VWAP reclaim momentum strategy: buy above VWAP, sell below VWAP."""

    VOLUME_THRESHOLD: float = 1.2   # relative volume minimum
    ATR_STOP_MULTIPLIER: float = 1.5
    ATR_TP_MULTIPLIER: float = 3.0
    CONFIDENCE: float = 0.58

    @property
    def name(self) -> str:
        return "vwap_reclaim"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 20

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

            # VWAP must be present in df.
            if "vwap" not in df.columns:
                return None

            close_series = df["close"]
            vwap_series = df["vwap"]

            if len(close_series) < 2:
                return None

            close_now = float(close_series.iloc[-1])
            close_prev = float(close_series.iloc[-2])
            vwap_now = float(vwap_series.iloc[-1])
            vwap_prev = float(vwap_series.iloc[-2])

            if pd.isna(vwap_now) or pd.isna(vwap_prev):
                return None

            # Volume confirmation: prefer VolumeContext, fall back to df column.
            rel_vol = self._relative_volume(df, volume)

            atr_dist_stop = self._atr_stop(df, self.ATR_STOP_MULTIPLIER)
            atr_dist_tp = self._atr_stop(df, self.ATR_TP_MULTIPLIER)
            if atr_dist_stop <= 0.0 or atr_dist_tp <= 0.0:
                return None

            # ── BUY: price crosses above VWAP from below ────────────────────────
            reclaim_up = close_prev <= vwap_prev and close_now > vwap_now
            if reclaim_up and rel_vol >= self.VOLUME_THRESHOLD:
                stop_loss = vwap_now - atr_dist_stop
                if stop_loss <= 0.0:
                    return None
                entry_low = close_now * 0.9995
                entry_high = close_now * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0
                if stop_loss >= entry_mid:
                    return None
                take_profit_1 = entry_mid + atr_dist_tp

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
                    regime=regime.regime if regime else MarketRegime.UNKNOWN,
                    session=session,
                    quality=quality,
                    raw_features={
                        "vwap": vwap_now,
                        "close": close_now,
                        "relative_volume": rel_vol,
                        "cross": "above_vwap",
                        "confidence": self.CONFIDENCE,
                    },
                )

            # ── SELL: price crosses below VWAP from above ───────────────────────
            break_down = close_prev >= vwap_prev and close_now < vwap_now
            if break_down and rel_vol >= self.VOLUME_THRESHOLD:
                stop_loss = vwap_now + atr_dist_stop
                entry_low = close_now * 0.9995
                entry_high = close_now * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0
                if stop_loss <= entry_mid:
                    return None
                take_profit_1 = entry_mid - atr_dist_tp
                if take_profit_1 <= 0.0:
                    return None

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
                    regime=regime.regime if regime else MarketRegime.UNKNOWN,
                    session=session,
                    quality=quality,
                    raw_features={
                        "vwap": vwap_now,
                        "close": close_now,
                        "relative_volume": rel_vol,
                        "cross": "below_vwap",
                        "confidence": self.CONFIDENCE,
                    },
                )

        except Exception:
            return None

        return None

    @staticmethod
    def _relative_volume(df: pd.DataFrame, volume: VolumeContext | None) -> float:
        """Return relative volume: prefer VolumeContext, fall back to df column."""
        if volume is not None and volume.relative_volume is not None:
            return float(volume.relative_volume)
        if "relative_volume" in df.columns:
            val = df["relative_volume"].iloc[-1]
            if not pd.isna(val):
                return float(val)
        # No volume info — use neutral 1.0 (below threshold, won't fire)
        return 1.0

    @staticmethod
    def _bias(structure: MarketStructure | None) -> TrendDirection:
        return structure.trend if structure else TrendDirection.UNKNOWN
