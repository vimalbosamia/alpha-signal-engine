"""
ATR Breakout Strategy.

Logic:
  - BUY  when close > prev_close + ATR * 1.5 (large up-move) AND ADX > 20.
  - SELL when close < prev_close - ATR * 1.5 (large down-move) AND ADX > 20.

Stop: ATR * 2 from entry.
TP  : ATR * 4 from entry.
Confidence: 0.60

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


class ATRBreakoutStrategy(BaseStrategy):
    """Momentum breakout confirmed by an ADX-gated ATR move."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "atr_breakout"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 20

    # ── Thresholds ─────────────────────────────────────────────────────────────

    ATR_TRIGGER_MULTIPLIER: float = 1.5   # move size required to trigger signal
    ATR_STOP_MULTIPLIER: float = 2.0
    ATR_TP_MULTIPLIER: float = 4.0
    MIN_ADX: float = 20.0                 # market must be trending
    CONFIDENCE: float = 0.60

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

            # 3. ADX filter — trending market only
            adx = self._get_adx(df, indicators)
            if adx is None or adx < self.MIN_ADX:
                return None

            # 4. ATR value
            atr = self._get_atr(df, indicators)
            if atr is None or atr <= 0.0:
                return None

            last_close = float(df["close"].iloc[-1])
            prev_close = float(df["close"].iloc[-2])
            trigger_dist = atr * self.ATR_TRIGGER_MULTIPLIER
            stop_dist = atr * self.ATR_STOP_MULTIPLIER
            tp_dist = atr * self.ATR_TP_MULTIPLIER

            market_regime = regime.regime if regime is not None else MarketRegime.UNKNOWN

            # ── BUY: large up-move ────────────────────────────────────────────
            if last_close > prev_close + trigger_dist:
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
                        "prev_close": prev_close,
                        "atr": atr,
                        "adx": adx,
                        "move": last_close - prev_close,
                        "confidence": self.CONFIDENCE,
                    },
                )

            # ── SELL: large down-move ─────────────────────────────────────────
            if last_close < prev_close - trigger_dist:
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
                        "prev_close": prev_close,
                        "atr": atr,
                        "adx": adx,
                        "move": last_close - prev_close,
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
    def _get_adx(df: pd.DataFrame, indicators: IndicatorSnapshot | None) -> float | None:
        """Return ADX from snapshot when available; fall back to a simple proxy."""
        if indicators is not None and indicators.adx is not None:
            return indicators.adx
        # Without pandas-ta here, return None — caller must provide indicators
        # or have sufficient bars for the snapshot to populate ADX.
        return None

    @staticmethod
    def _htf_bias(structure: MarketStructure | None) -> TrendDirection:
        return structure.trend if structure is not None else TrendDirection.UNKNOWN
