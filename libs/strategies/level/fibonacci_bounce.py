"""
Fibonacci Retracement Bounce Strategy.

Logic: Identifies 38.2%–61.8% Fibonacci retracement zones over the last 50
bars and triggers when price pulls back into that zone with trend alignment.
  - BUY:  structure.trend == UPTREND and current close is inside the 38.2–61.8%
          retracement zone (price pulled back to support within the bull swing)
  - SELL: structure.trend == DOWNTREND and current close is inside the 38.2–61.8%
          retracement zone (price bounced back to resistance within the bear swing)

Fibonacci levels (from 50-bar swing high H and swing low L):
  fib_382 = H - (H - L) * 0.382
  fib_618 = H - (H - L) * 0.618
  fib_786 = H - (H - L) * 0.786   ← stop reference

Stop:  beyond the 78.6% level (below for BUY, above for SELL).
TP:    back to the swing high (BUY) or swing low (SELL).

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


class FibonacciBounceStrategy(BaseStrategy):
    """
    Fibonacci retracement bounce strategy.

    Waits for price to pull back into the 38.2–61.8% retracement zone of the
    dominant swing, then enters in the direction of the higher-timeframe trend.
    """

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "fibonacci_bounce"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 50

    # ── Thresholds ─────────────────────────────────────────────────────────────

    CONFIDENCE: float = 0.58
    SWING_LOOKBACK: int = 50

    FIB_382: float = 0.382
    FIB_618: float = 0.618
    FIB_786: float = 0.786   # stop reference level

    # Small buffer so price can pierce the zone edge by a fraction without
    # disqualifying the setup (0.1% of price range).
    ZONE_TOLERANCE: float = 0.001

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
            if not self.is_eligible(asset_class, session, quality):
                return None

            if df is None or len(df) < self.min_bars_required:
                return None

            # ── Require an established trend ──────────────────────────────────
            if structure is None:
                return None

            trend = structure.trend
            if trend not in (TrendDirection.UPTREND, TrendDirection.DOWNTREND):
                return None

            # ── Compute swing high / low over SWING_LOOKBACK bars ─────────────
            lookback = df.iloc[-self.SWING_LOOKBACK :]
            swing_high = float(lookback["high"].max())
            swing_low = float(lookback["low"].min())

            swing_range = swing_high - swing_low
            if swing_range <= 0.0:
                return None

            # ── Fibonacci retracement levels ──────────────────────────────────
            fib_382 = swing_high - swing_range * self.FIB_382
            fib_618 = swing_high - swing_range * self.FIB_618
            fib_786 = swing_high - swing_range * self.FIB_786

            # Tolerance band (small fraction of range to avoid false negatives)
            tolerance = swing_range * self.ZONE_TOLERANCE

            last_close = float(df["close"].iloc[-1])

            # ── Check whether price is inside the retracement zone ────────────
            zone_low = fib_618 - tolerance
            zone_high = fib_382 + tolerance
            in_zone = zone_low <= last_close <= zone_high

            if not in_zone:
                return None

            # ── Determine direction and levels ────────────────────────────────
            if trend == TrendDirection.UPTREND:
                action = SignalAction.BUY
                stop_loss = fib_786 - tolerance
                if stop_loss <= 0.0:
                    return None
                take_profit_1 = swing_high
            else:  # DOWNTREND
                action = SignalAction.SELL
                # Mirror: 78.6% from LOW end
                fib_786_sell = swing_low + swing_range * self.FIB_786
                stop_loss = fib_786_sell + tolerance
                take_profit_1 = swing_low

            # ── Entry zone around last close ──────────────────────────────────
            entry_low = last_close * 0.9995
            entry_high = last_close * 1.0005
            entry_mid = (entry_low + entry_high) / 2.0

            # Basic sanity: stop must be on the correct side of entry
            if action == SignalAction.BUY and stop_loss >= entry_mid:
                return None
            if action == SignalAction.SELL and stop_loss <= entry_mid:
                return None
            if take_profit_1 <= 0.0:
                return None

            higher_tf_bias = structure.trend if structure is not None else TrendDirection.UNKNOWN

            return SignalCandidate(
                symbol=symbol,
                asset_class=asset_class,
                strategy_name=self.name,
                proposed_action=action,
                timeframe=Timeframe.FIFTEEN_MIN,
                higher_tf_bias=higher_tf_bias,
                entry_zone_low=entry_low,
                entry_zone_high=entry_high,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                session=session,
                quality=quality,
                raw_features={
                    "swing_high": swing_high,
                    "swing_low": swing_low,
                    "fib_382": fib_382,
                    "fib_618": fib_618,
                    "fib_786": fib_786,
                    "last_close": last_close,
                },
            )

        except Exception:  # noqa: BLE001 — never propagate
            return None
