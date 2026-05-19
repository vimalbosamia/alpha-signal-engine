"""
Bollinger Band Mean Reversion Strategy.

Logic:
  - BUY  when price touches/crosses below the lower band (bb_pct_b < 0).
  - SELL when price touches/crosses above the upper band (bb_pct_b > 1).

Target is the Bollinger middle band (mean reversion).
Stop is ATR*2 beyond the touched band.

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


class BollingerReversionStrategy(BaseStrategy):
    """Bollinger Band mean-reversion: buy at lower band, sell at upper band."""

    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    ATR_STOP_MULTIPLIER: float = 2.0
    CONFIDENCE: float = 0.55

    @property
    def name(self) -> str:
        return "bollinger_mean_reversion"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 25

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

            last_close = float(df["close"].iloc[-1])

            # Prefer pre-computed snapshot; fall back to local computation.
            if (
                indicators is not None
                and indicators.bb_pct_b is not None
                and indicators.bb_upper is not None
                and indicators.bb_lower is not None
                and indicators.bb_middle is not None
            ):
                bb_pct_b = indicators.bb_pct_b
                bb_upper = indicators.bb_upper
                bb_lower = indicators.bb_lower
                bb_middle = indicators.bb_middle
            else:
                bb_upper, bb_middle, bb_lower, bb_pct_b = self._compute_bb(df)
                if bb_upper is None:
                    return None

            atr_dist = self._atr_stop(df, self.ATR_STOP_MULTIPLIER)
            if atr_dist <= 0.0:
                return None

            # ── BUY: price at/below lower band (bb_pct_b < 0) ──────────────────
            if bb_pct_b < 0.0:
                stop_loss = bb_lower - atr_dist
                if stop_loss <= 0.0:
                    return None
                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0
                if stop_loss >= entry_mid:
                    return None
                take_profit_1 = bb_middle  # mean reversion target
                if take_profit_1 <= entry_mid:
                    return None

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
                        "bb_pct_b": bb_pct_b,
                        "bb_lower": bb_lower,
                        "bb_middle": bb_middle,
                        "bb_upper": bb_upper,
                        "touch": "lower_band",
                        "confidence": self.CONFIDENCE,
                    },
                )

            # ── SELL: price at/above upper band (bb_pct_b > 1) ─────────────────
            if bb_pct_b > 1.0:
                stop_loss = bb_upper + atr_dist
                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0
                if stop_loss <= entry_mid:
                    return None
                take_profit_1 = bb_middle  # mean reversion target
                if take_profit_1 >= entry_mid:
                    return None
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
                        "bb_pct_b": bb_pct_b,
                        "bb_lower": bb_lower,
                        "bb_middle": bb_middle,
                        "bb_upper": bb_upper,
                        "touch": "upper_band",
                        "confidence": self.CONFIDENCE,
                    },
                )

        except Exception:
            return None

        return None

    def _compute_bb(
        self, df: pd.DataFrame
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Return (upper, middle, lower, pct_b) from df, or (None, None, None, None)."""
        close = df["close"]
        middle_series = close.rolling(self.BB_PERIOD).mean()
        std_series = close.rolling(self.BB_PERIOD).std()

        if middle_series.isna().iloc[-1] or std_series.isna().iloc[-1]:
            return None, None, None, None

        middle = float(middle_series.iloc[-1])
        std = float(std_series.iloc[-1])
        upper = middle + self.BB_STD * std
        lower = middle - self.BB_STD * std
        band_range = upper - lower

        if band_range == 0.0:
            return None, None, None, None

        last_close = float(close.iloc[-1])
        pct_b = (last_close - lower) / band_range
        return upper, middle, lower, pct_b

    @staticmethod
    def _bias(structure: MarketStructure | None) -> TrendDirection:
        return structure.trend if structure else TrendDirection.UNKNOWN
