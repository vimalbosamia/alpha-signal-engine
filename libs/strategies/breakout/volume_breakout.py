"""
Volume Breakout Strategy.

Logic:
  - BUY  when close breaks above the 20-bar rolling high AND volume > 2x average.
  - SELL when close breaks below the 20-bar rolling low AND volume > 2x average.

Stop: ATR * 1.5 below/above entry.
TP  : ATR * 4 above/below entry.
Confidence: 0.62

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


class VolumeBreakoutStrategy(BaseStrategy):
    """Breakout above/below 20-bar high/low confirmed by a volume spike."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "volume_breakout"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 25

    # ── Thresholds ─────────────────────────────────────────────────────────────

    ROLLING_WINDOW: int = 20
    VOLUME_SPIKE_MULTIPLIER: float = 2.0   # volume must exceed this × rolling mean
    ATR_STOP_MULTIPLIER: float = 1.5
    ATR_TP_MULTIPLIER: float = 4.0
    CONFIDENCE: float = 0.62

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

            last_close = float(df["close"].iloc[-1])

            # 3. Rolling 20-bar resistance and support levels
            rolling_high = df["high"].rolling(self.ROLLING_WINDOW).max()
            rolling_low = df["low"].rolling(self.ROLLING_WINDOW).min()

            # Use the bar *before* the last bar so the current close is compared
            # against the prior 20-bar extremes (avoids look-ahead).
            if len(rolling_high) < 2 or pd.isna(rolling_high.iloc[-2]):
                return None
            if len(rolling_low) < 2 or pd.isna(rolling_low.iloc[-2]):
                return None

            resistance = float(rolling_high.iloc[-2])
            support = float(rolling_low.iloc[-2])

            # 4. Volume spike — current bar volume vs 20-bar average
            vol_mean = float(df["volume"].rolling(self.ROLLING_WINDOW).mean().iloc[-1])
            last_vol = float(df["volume"].iloc[-1])
            has_volume_spike = vol_mean > 0 and last_vol >= self.VOLUME_SPIKE_MULTIPLIER * vol_mean

            if not has_volume_spike:
                return None

            # 5. ATR for stop and TP distances
            atr = self._get_atr(df, indicators)
            if atr is None or atr <= 0.0:
                return None

            stop_dist = atr * self.ATR_STOP_MULTIPLIER
            tp_dist = atr * self.ATR_TP_MULTIPLIER

            market_regime = regime.regime if regime is not None else MarketRegime.UNKNOWN

            # ── BUY: close breaks above 20-bar high ───────────────────────────
            if last_close > resistance:
                entry_low = last_close
                entry_high = last_close * 1.001
                entry_mid = (entry_low + entry_high) / 2.0

                stop_loss = entry_mid - stop_dist
                if stop_loss <= 0.0 or stop_loss >= entry_mid:
                    return None

                take_profit_1 = entry_mid + tp_dist

                return SignalCandidate(
                    symbol=symbol,
                    asset_class=asset_class,
                    strategy_name=self.name,
                    proposed_action=SignalAction.BUY,
                    timeframe=Timeframe.FIFTEEN_MIN,
                    higher_tf_bias=self._htf_bias(structure),
                    entry_zone_low=entry_low,
                    entry_zone_high=entry_high,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    regime=market_regime,
                    session=session,
                    quality=quality,
                    raw_features={
                        "last_close": last_close,
                        "resistance": resistance,
                        "last_volume": last_vol,
                        "volume_mean": vol_mean,
                        "atr": atr,
                        "confidence": self.CONFIDENCE,
                    },
                )

            # ── SELL: close breaks below 20-bar low ───────────────────────────
            if last_close < support:
                entry_low = last_close * 0.999
                entry_high = last_close
                entry_mid = (entry_low + entry_high) / 2.0

                stop_loss = entry_mid + stop_dist
                if entry_mid <= 0.0 or stop_loss <= entry_mid:
                    return None

                take_profit_1 = entry_mid - tp_dist
                if take_profit_1 <= 0.0:
                    return None

                return SignalCandidate(
                    symbol=symbol,
                    asset_class=asset_class,
                    strategy_name=self.name,
                    proposed_action=SignalAction.SELL,
                    timeframe=Timeframe.FIFTEEN_MIN,
                    higher_tf_bias=self._htf_bias(structure),
                    entry_zone_low=entry_low,
                    entry_zone_high=entry_high,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    regime=market_regime,
                    session=session,
                    quality=quality,
                    raw_features={
                        "last_close": last_close,
                        "support": support,
                        "last_volume": last_vol,
                        "volume_mean": vol_mean,
                        "atr": atr,
                        "confidence": self.CONFIDENCE,
                    },
                )

        except Exception:  # noqa: BLE001 — never propagate
            return None

        return None

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _get_atr(df: pd.DataFrame, indicators: IndicatorSnapshot | None) -> float | None:
        """Return ATR from snapshot when available; fall back to rolling TR mean."""
        if indicators is not None and indicators.atr is not None:
            return indicators.atr
        tr = (df["high"] - df["low"]).abs().rolling(14).mean().iloc[-1]
        if pd.isna(tr) or tr <= 0.0:
            return None
        return float(tr)

    @staticmethod
    def _htf_bias(structure: MarketStructure | None) -> TrendDirection:
        return structure.trend if structure is not None else TrendDirection.UNKNOWN
