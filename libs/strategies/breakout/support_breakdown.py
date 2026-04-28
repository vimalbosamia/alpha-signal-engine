"""
Support Breakdown Strategy.

Logic: Price breaks below support with high volume.

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


class SupportBreakdownStrategy(BaseStrategy):
    """Breakdown below support with volume confirmation."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "support_breakdown"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 30

    # ── Thresholds ─────────────────────────────────────────────────────────────

    BREAKDOWN_THRESHOLD_PCT: float = 0.002   # close < support * (1 - threshold)
    ATR_STOP_MULTIPLIER: float = 2.0
    DEFAULT_RR: float = 3.0
    MIN_REL_VOL: float = 1.3                 # breakdown needs above-average volume

    # How far above close we look for the recently broken support (as fraction of close)
    _MAX_SUPPORT_SEARCH_PCT: float = 0.02

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

            # 3. Get last close
            last_close = float(df["close"].iloc[-1])

            # 4. Find nearest support just broken (support now slightly above close)
            support_level = self._nearest_support_broken(last_close, levels or [])
            if support_level is None:
                return None

            support_price = support_level.price

            # 5. Check confirmed breakdown: close below support by threshold
            breakdown_confirmed = last_close < support_price * (1.0 - self.BREAKDOWN_THRESHOLD_PCT)
            if not breakdown_confirmed:
                return None

            # 6. Volume confirmation
            if volume is not None and volume.relative_volume < self.MIN_REL_VOL:
                return None

            # 7. Compute stop above broken support (support becomes new resistance)
            atr_dist = self._atr_stop(df, self.ATR_STOP_MULTIPLIER)
            stop_loss = support_price + atr_dist
            if stop_loss <= 0.0:
                return None

            # 8. Entry zone
            entry_high = last_close
            entry_low = last_close * 0.999
            entry_mid = (entry_low + entry_high) / 2.0

            # Ensure stop is above entry (SELL geometry)
            if stop_loss <= entry_mid:
                return None

            # 9. Take profit
            stop_distance = stop_loss - entry_mid
            take_profit_1 = entry_mid - self.DEFAULT_RR * stop_distance
            if take_profit_1 <= 0.0:
                return None

            # Determine HTF bias
            higher_tf_bias = self._htf_bias(structure)

            return SignalCandidate(
                symbol=symbol,
                asset_class=asset_class,
                strategy_name=self.name,
                proposed_action=SignalAction.SELL,
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
                    "support_price": support_price,
                    "last_close": last_close,
                    "relative_volume": volume.relative_volume if volume is not None else None,
                    "atr_dist": atr_dist,
                },
            )

        except Exception:  # noqa: BLE001 — never propagate
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _nearest_support_broken(
        self,
        close: float,
        levels: list[KeyLevel],
    ) -> KeyLevel | None:
        """
        Return the nearest support level that is currently just above the close
        (i.e., the support the price just broke below), within
        _MAX_SUPPORT_SEARCH_PCT above the close.
        """
        # Broken support should be slightly above current close but not too far
        search_high = close * (1.0 + self._MAX_SUPPORT_SEARCH_PCT)

        best: KeyLevel | None = None
        best_dist = float("inf")

        for lvl in levels:
            if lvl.level_type not in (
                "support", "pdl", "or_low", "weekly_low", "gap_zone_bottom"
            ):
                continue
            # Broken support should be above current close but within search band
            if close <= lvl.price <= search_high:
                dist = abs(lvl.price - close)
                if dist < best_dist:
                    best_dist = dist
                    best = lvl

        return best

    def _htf_bias(self, structure: MarketStructure | None) -> TrendDirection:
        if structure is not None:
            return structure.trend
        return TrendDirection.UNKNOWN
