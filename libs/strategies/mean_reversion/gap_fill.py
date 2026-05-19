"""
Gap Fill Strategy.

Logic: Detects a significant gap between the previous bar's close and the
current bar's open.  Gaps tend to fill, so the trade fades the gap direction:
  - Gap up  (open > prev_close * 1.003) → expect price to fall back → SELL.
  - Gap down (open < prev_close * 0.997) → expect price to rise back → BUY.

Design rules:
  - Never raises in generate_candidate — returns None on any data issue.
  - No execution logic — produces signal candidates only.
  - All thresholds are named constants.
"""
from __future__ import annotations

import pandas as pd

from libs.analysis.indicators.engine import IndicatorSnapshot
from libs.analysis.levels.engine import KeyLevel
from libs.analysis.regime.engine import RegimeAnalysis
from libs.analysis.structure.engine import MarketStructure
from libs.analysis.volume.engine import VolumeContext
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
from libs.strategies.base.strategy import BaseStrategy


class GapFillStrategy(BaseStrategy):
    """Mean-reversion strategy that fades intraday price gaps."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "gap_fill"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 5

    # ── Thresholds ─────────────────────────────────────────────────────────────

    GAP_THRESHOLD: float = 0.003      # 0.3 % gap required to trigger
    ATR_STOP_MULTIPLIER: float = 2.0
    CONFIDENCE: float = 0.52          # lower — gaps don't always fill

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

            # 3. Identify the gap: compare current bar open vs previous bar close
            prev_close = float(df["close"].iloc[-2])
            current_open = float(df["open"].iloc[-1])
            last_close = float(df["close"].iloc[-1])

            if prev_close <= 0.0:
                return None

            # 4. ATR for stop placement
            atr_dist = self._atr_stop(df, self.ATR_STOP_MULTIPLIER)
            if atr_dist <= 0.0:
                return None

            gap_up_threshold = prev_close * (1.0 + self.GAP_THRESHOLD)
            gap_down_threshold = prev_close * (1.0 - self.GAP_THRESHOLD)

            # 5. Gap up → fade → SELL back toward prev_close
            if current_open > gap_up_threshold:
                stop_loss = current_open + atr_dist  # stop above gap open
                take_profit_1 = prev_close            # target: fill the gap
                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0

                if stop_loss <= entry_mid:
                    return None
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
                    higher_tf_bias=self._htf_bias(structure),
                    entry_zone_low=entry_low,
                    entry_zone_high=entry_high,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                    session=session,
                    quality=quality,
                    raw_features={
                        "prev_close": prev_close,
                        "current_open": current_open,
                        "gap_direction": "up",
                        "atr_dist": atr_dist,
                        "confidence": self.CONFIDENCE,
                    },
                )

            # 6. Gap down → fade → BUY back toward prev_close
            if current_open < gap_down_threshold:
                stop_loss = current_open - atr_dist  # stop below gap open
                if stop_loss <= 0.0:
                    return None
                take_profit_1 = prev_close            # target: fill the gap
                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0

                if stop_loss >= entry_mid:
                    return None
                if take_profit_1 <= entry_mid:
                    return None

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
                    regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                    session=session,
                    quality=quality,
                    raw_features={
                        "prev_close": prev_close,
                        "current_open": current_open,
                        "gap_direction": "down",
                        "atr_dist": atr_dist,
                        "confidence": self.CONFIDENCE,
                    },
                )

        except Exception:  # noqa: BLE001 — never propagate
            return None

        return None

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _htf_bias(structure: MarketStructure | None) -> TrendDirection:
        return structure.trend if structure is not None else TrendDirection.UNKNOWN
