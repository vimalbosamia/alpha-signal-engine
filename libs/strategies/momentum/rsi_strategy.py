"""
RSI Oversold/Overbought Strategy.

Logic:
  - BUY  when RSI(14) crosses up through oversold threshold (< 35) and turns up.
  - SELL when RSI(14) crosses down through overbought threshold (> 65) and turns down.

Design rules:
  - Never raises in generate_candidate — returns None on any data issue.
  - No execution logic — produces signal candidates only.
"""
from __future__ import annotations

import numpy as np
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


class RSIStrategy(BaseStrategy):
    """RSI mean-reversion: buy oversold, sell overbought."""

    RSI_PERIOD: int = 14
    OVERSOLD: float = 35.0
    OVERBOUGHT: float = 65.0
    ATR_STOP_MULTIPLIER: float = 2.0
    DEFAULT_RR: float = 2.5

    @property
    def name(self) -> str:
        return "rsi_mean_reversion"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 30

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

            # Prefer pre-computed indicator snapshot; fall back to local computation
            if indicators is not None and indicators.rsi is not None and indicators.rsi_prev is not None:
                rsi_now = indicators.rsi
                rsi_prev = indicators.rsi_prev
            else:
                rsi = self._compute_rsi(df["close"], self.RSI_PERIOD)
                if rsi.isna().iloc[-1]:
                    return None
                rsi_now = float(rsi.iloc[-1])
                rsi_prev = float(rsi.iloc[-2]) if len(rsi) >= 2 else rsi_now
            last_close = float(df["close"].iloc[-1])
            atr_dist = self._atr_stop(df, self.ATR_STOP_MULTIPLIER)
            if atr_dist <= 0.0:
                return None

            # ── BUY: RSI was oversold and is now turning up ────────────────────
            if rsi_prev < self.OVERSOLD and rsi_now > rsi_prev:
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
                    raw_features={"rsi_now": rsi_now, "rsi_prev": rsi_prev},
                )

            # ── SELL: RSI was overbought and is now turning down ───────────────
            if rsi_prev > self.OVERBOUGHT and rsi_now < rsi_prev:
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
                    raw_features={"rsi_now": rsi_now, "rsi_prev": rsi_prev},
                )

        except Exception:
            return None

        return None

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _bias(structure: MarketStructure | None) -> TrendDirection:
        return structure.trend if structure else TrendDirection.UNKNOWN
