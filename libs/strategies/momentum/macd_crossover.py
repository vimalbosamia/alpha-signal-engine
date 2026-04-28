"""
MACD Crossover Strategy.

Logic:
  - BUY  when MACD line crosses above signal line (histogram: -→+).
  - SELL when MACD line crosses below signal line (histogram: +→-).

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


class MACDCrossoverStrategy(BaseStrategy):
    """MACD(12,26,9) line-cross signal generator."""

    FAST: int = 12
    SLOW: int = 26
    SIGNAL: int = 9
    ATR_STOP_MULTIPLIER: float = 2.0
    DEFAULT_RR: float = 2.5

    @property
    def name(self) -> str:
        return "macd_crossover"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 40

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
            if (
                indicators is not None
                and indicators.macd_histogram is not None
                and indicators.macd_histogram_prev is not None
                and indicators.macd_line is not None
                and indicators.macd_signal is not None
            ):
                hist_now = indicators.macd_histogram
                hist_prev = indicators.macd_histogram_prev
                macd_line_val = indicators.macd_line
                signal_line_val = indicators.macd_signal
            else:
                macd_line_series, signal_line_series, histogram = self._compute_macd(df["close"])
                if len(histogram) < 2:
                    return None
                hist_now = float(histogram.iloc[-1])
                hist_prev = float(histogram.iloc[-2])
                if pd.isna(hist_now) or pd.isna(hist_prev):
                    return None
                macd_line_val = float(macd_line_series.iloc[-1])
                signal_line_val = float(signal_line_series.iloc[-1])

            last_close = float(df["close"].iloc[-1])
            atr_dist = self._atr_stop(df, self.ATR_STOP_MULTIPLIER)
            if atr_dist <= 0.0:
                return None

            # ── BUY: histogram flips negative → positive ───────────────────────
            if hist_prev < 0.0 and hist_now > 0.0:
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
                        "macd": macd_line_val,
                        "signal": signal_line_val,
                        "histogram": hist_now,
                    },
                )

            # ── SELL: histogram flips positive → negative ──────────────────────
            if hist_prev > 0.0 and hist_now < 0.0:
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
                        "macd": macd_line_val,
                        "signal": signal_line_val,
                        "histogram": hist_now,
                    },
                )

        except Exception:
            return None

        return None

    def _compute_macd(
        self, close: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=self.FAST, adjust=False).mean()
        ema_slow = close.ewm(span=self.SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.SIGNAL, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _bias(structure: MarketStructure | None) -> TrendDirection:
        return structure.trend if structure else TrendDirection.UNKNOWN
