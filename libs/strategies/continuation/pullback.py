"""
Pullback Continuation Strategy.

Logic: Pullback to VWAP in an uptrend, then continuation long.

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
from libs.analysis.levels.engine import KeyLevel
from libs.analysis.regime.engine import RegimeAnalysis
from libs.analysis.structure.engine import MarketStructure
from libs.analysis.volume.engine import VolumeContext
from libs.strategies.base.strategy import BaseStrategy


class PullbackContinuationStrategy(BaseStrategy):
    """Pullback to VWAP continuation long — requires uptrend context."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "pullback_continuation"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 40

    # ── Thresholds ─────────────────────────────────────────────────────────────

    PULLBACK_VWAP_PCT: float = 0.005     # close within 0.5% of VWAP
    ATR_STOP_MULTIPLIER: float = 1.5
    DEFAULT_RR: float = 2.0

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
        indicators: "IndicatorSnapshot | None" = None,
    ) -> SignalCandidate | None:
        try:
            # 1. Pre-flight eligibility
            if not self.is_eligible(asset_class, session, quality):
                return None

            # 2. Require minimum bars
            if df is None or len(df) < self.min_bars_required:
                return None

            # 3. Require uptrend
            if not self._is_uptrend(structure, df_htf):
                return None

            # 4. Require VWAP column
            if "vwap" not in df.columns:
                return None

            vwap_val = df["vwap"].iloc[-1]
            if pd.isna(vwap_val):
                return None
            vwap = float(vwap_val)

            # 5. Check pullback: close within PULLBACK_VWAP_PCT of vwap
            last_close = float(df["close"].iloc[-1])
            proximity = abs(last_close - vwap) / vwap if vwap > 0.0 else float("inf")
            if proximity > self.PULLBACK_VWAP_PCT:
                return None

            # 6. Bullish bar: close > open — bid coming in at VWAP
            last_open = float(df["open"].iloc[-1])
            if last_close <= last_open:
                return None

            # 7. Stop below VWAP
            atr_dist = self._atr_stop(df, self.ATR_STOP_MULTIPLIER)
            stop_loss = vwap - atr_dist
            if stop_loss <= 0.0:
                return None

            # 8. Entry zone
            entry_low = last_close * 0.9998
            entry_high = last_close * 1.0002
            entry_mid = (entry_low + entry_high) / 2.0

            # Ensure stop is below entry
            if stop_loss >= entry_mid:
                return None

            # 9. Take profit
            stop_distance = entry_mid - stop_loss
            take_profit_1 = entry_mid + self.DEFAULT_RR * stop_distance

            # Determine HTF bias
            higher_tf_bias = self._htf_bias(structure)

            return SignalCandidate(
                symbol=symbol,
                asset_class=asset_class,
                strategy_name=self.name,
                proposed_action=SignalAction.BUY,
                timeframe=Timeframe.FIVE_MIN,
                higher_tf_bias=higher_tf_bias,
                entry_zone_low=entry_low,
                entry_zone_high=entry_high,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                session=session,
                quality=quality,
                raw_features={
                    "vwap": vwap,
                    "last_close": last_close,
                    "proximity_pct": proximity,
                    "atr_dist": atr_dist,
                },
            )

        except Exception:  # noqa: BLE001 — never propagate
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _is_uptrend(
        self,
        structure: MarketStructure | None,
        df_htf: pd.DataFrame | None,
    ) -> bool:
        """Return True only when primary structure signals an uptrend."""
        if structure is None:
            return False
        return structure.trend == TrendDirection.UPTREND

    def _htf_bias(self, structure: MarketStructure | None) -> TrendDirection:
        if structure is not None:
            return structure.trend
        return TrendDirection.UNKNOWN
