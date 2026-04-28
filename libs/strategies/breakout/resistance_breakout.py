"""
Resistance Breakout Strategy.

Logic: Price breaks above resistance with high volume.

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


class ResistanceBreakoutStrategy(BaseStrategy):
    """Breakout above resistance with volume confirmation."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "resistance_breakout"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 30

    # ── Thresholds ─────────────────────────────────────────────────────────────

    BREAKOUT_THRESHOLD_PCT: float = 0.002   # close > resistance * (1 + threshold)
    ATR_STOP_MULTIPLIER: float = 2.0
    DEFAULT_RR: float = 3.0
    MIN_REL_VOL: float = 1.3               # breakout needs above-average volume

    # How far above close we look for a relevant resistance (as fraction of close)
    _MAX_RESISTANCE_SEARCH_PCT: float = 0.02

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

            # 4. Find nearest resistance within search band above current price
            resistance_level = self._nearest_resistance(last_close, levels or [])
            if resistance_level is None:
                return None

            resistance_price = resistance_level.price

            # 5. Check confirmed breakout: close above resistance by threshold
            breakout_confirmed = last_close > resistance_price * (1.0 + self.BREAKOUT_THRESHOLD_PCT)
            if not breakout_confirmed:
                return None

            # 6. Volume confirmation
            if volume is not None and volume.relative_volume < self.MIN_REL_VOL:
                return None

            # 7. Compute stop below resistance (resistance becomes new support)
            atr_dist = self._atr_stop(df, self.ATR_STOP_MULTIPLIER)
            stop_loss = resistance_price - atr_dist
            if stop_loss <= 0.0:
                return None

            # 8. Entry zone
            entry_low = last_close
            entry_high = last_close * 1.001
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
                    "resistance_price": resistance_price,
                    "last_close": last_close,
                    "relative_volume": volume.relative_volume if volume is not None else None,
                    "atr_dist": atr_dist,
                },
            )

        except Exception:  # noqa: BLE001 — never propagate
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _nearest_resistance(
        self,
        close: float,
        levels: list[KeyLevel],
    ) -> KeyLevel | None:
        """
        Return the nearest resistance level that is currently at or below the
        breakout close (i.e., the resistance the price just broke through),
        within _MAX_RESISTANCE_SEARCH_PCT above the close before the break.

        Strategy: find a resistance whose price is within the search band
        below the current close (just broken through) and as close to the
        close as possible.
        """
        # The resistance that was just broken should be slightly below close
        # but within our search band (close / (1 + threshold) roughly)
        search_low = close * (1.0 - self._MAX_RESISTANCE_SEARCH_PCT)

        best: KeyLevel | None = None
        best_dist = float("inf")

        for lvl in levels:
            if lvl.level_type not in (
                "resistance", "pdh", "or_high", "weekly_high",
                "gap_zone_top", "gap_zone_bottom",
            ):
                continue
            # The broken resistance should be below current close
            # but not too far below
            if search_low <= lvl.price <= close:
                dist = abs(lvl.price - close)
                if dist < best_dist:
                    best_dist = dist
                    best = lvl

        return best

    def _htf_bias(self, structure: MarketStructure | None) -> TrendDirection:
        if structure is not None:
            return structure.trend
        return TrendDirection.UNKNOWN
