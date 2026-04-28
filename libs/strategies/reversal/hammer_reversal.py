"""
Hammer Reversal Strategy.

Logic: Bullish reversal at support when in downtrend context.
Detects hammer and bullish-engulfing patterns using raw OHLCV bars.

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
    PatternBias,
    PatternResult,
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


class HammerReversalStrategy(BaseStrategy):
    """Bullish reversal at support — hammer / engulfing pattern required."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "hammer_reversal"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 50

    # ── Thresholds ─────────────────────────────────────────────────────────────

    MIN_PATTERN_CONFIDENCE: float = 0.55
    MIN_LEVEL_PROXIMITY_PCT: float = 0.005   # entry within 0.5% of a support
    ATR_STOP_MULTIPLIER: float = 1.5
    DEFAULT_RR: float = 2.5                  # target = entry + 2.5 * stop_distance

    # Hammer geometry: lower wick must be at least this multiple of the body
    _HAMMER_WICK_BODY_RATIO: float = 2.0
    # Upper wick must be <= this fraction of the total range
    _HAMMER_MAX_UPPER_WICK_PCT: float = 0.25

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

            # 3. Detect bullish patterns from the last 3 bars
            pattern_results: list[PatternResult] = []
            hammer = self._detect_hammer(df)
            engulf = self._detect_bullish_engulf(df)
            if hammer is not None:
                pattern_results.append(hammer)
            if engulf is not None:
                pattern_results.append(engulf)

            # Need at least one confirmed pattern
            confirmed = [p for p in pattern_results if p.detected and p.confidence >= self.MIN_PATTERN_CONFIDENCE]
            if not confirmed:
                return None

            # 4. Confirm downtrend or ranging context
            if structure is not None:
                if structure.trend not in (
                    TrendDirection.DOWNTREND,
                    TrendDirection.RANGING,
                    TrendDirection.UNKNOWN,
                ):
                    return None

            # 5. Find nearest support level within proximity of last close
            last_close = float(df["close"].iloc[-1])
            support_level = self._nearest_support(last_close, levels or [])
            if support_level is None:
                return None

            support_price = support_level.price

            # 6. Compute stop below support
            atr_dist = self._atr_stop(df, self.ATR_STOP_MULTIPLIER)
            stop_loss = support_price - atr_dist
            if stop_loss <= 0.0:
                return None

            # 7. Entry zone around last close
            entry_low = last_close * 0.9995
            entry_high = last_close * 1.0005
            entry_mid = (entry_low + entry_high) / 2.0

            # Ensure stop is below entry
            if stop_loss >= entry_mid:
                return None

            # 8. Take profit
            stop_distance = entry_mid - stop_loss
            take_profit_1 = entry_mid + self.DEFAULT_RR * stop_distance

            # 9. Determine HTF bias
            higher_tf_bias = self._htf_bias(df_htf, structure)

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
                pattern_results=pattern_results,
                regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                session=session,
                quality=quality,
                raw_features={
                    "support_price": support_price,
                    "last_close": last_close,
                    "atr_dist": atr_dist,
                },
            )

        except Exception:  # noqa: BLE001 — never propagate
            return None

    # ── Pattern detectors ──────────────────────────────────────────────────────

    def _detect_hammer(self, df: pd.DataFrame) -> PatternResult | None:
        """
        Hammer: small body, long lower wick (>= 2x body), small upper wick.
        Uses the last bar only.
        """
        if len(df) < 1:
            return None

        bar = df.iloc[-1]
        open_ = float(bar["open"])
        high_ = float(bar["high"])
        low_ = float(bar["low"])
        close_ = float(bar["close"])

        body = abs(close_ - open_)
        total_range = high_ - low_
        if total_range <= 0.0:
            return None

        lower_wick = min(open_, close_) - low_
        upper_wick = high_ - max(open_, close_)

        # Hammer criteria
        has_long_lower_wick = body > 0 and lower_wick >= self._HAMMER_WICK_BODY_RATIO * body
        has_small_upper_wick = upper_wick <= self._HAMMER_MAX_UPPER_WICK_PCT * total_range

        detected = has_long_lower_wick and has_small_upper_wick

        if not detected:
            return None

        # Confidence: ratio of lower wick to total range, scaled
        confidence = min(1.0, lower_wick / total_range + 0.2)

        return PatternResult(
            pattern_name="hammer",
            detected=True,
            confidence=confidence,
            bias=PatternBias.BULLISH,
            candle_span=1,
            details={
                "lower_wick": lower_wick,
                "upper_wick": upper_wick,
                "body": body,
            },
        )

    def _detect_bullish_engulf(self, df: pd.DataFrame) -> PatternResult | None:
        """
        Bullish engulfing: last bar is bullish and its body fully engulfs the
        prior bar's bearish body.
        """
        if len(df) < 2:
            return None

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        prev_open = float(prev["open"])
        prev_close = float(prev["close"])
        curr_open = float(curr["open"])
        curr_close = float(curr["close"])

        prev_is_bearish = prev_close < prev_open
        curr_is_bullish = curr_close > curr_open

        if not (prev_is_bearish and curr_is_bullish):
            return None

        prev_body_top = prev_open      # bearish: open > close
        prev_body_bot = prev_close

        # Engulf: current open <= prior close AND current close >= prior open
        engulfs = curr_open <= prev_body_bot and curr_close >= prev_body_top

        if not engulfs:
            return None

        prev_body_size = abs(prev_open - prev_close)
        curr_body_size = abs(curr_close - curr_open)

        if prev_body_size <= 0.0:
            return None

        # Confidence based on how much bigger the engulfing body is
        ratio = min(2.0, curr_body_size / prev_body_size)
        confidence = min(1.0, 0.50 + (ratio - 1.0) * 0.25)

        return PatternResult(
            pattern_name="bullish_engulfing",
            detected=True,
            confidence=confidence,
            bias=PatternBias.BULLISH,
            candle_span=2,
            details={
                "prev_body": prev_body_size,
                "curr_body": curr_body_size,
                "engulf_ratio": ratio,
            },
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _nearest_support(
        self,
        close: float,
        levels: list[KeyLevel],
    ) -> KeyLevel | None:
        """Return the nearest support level within MIN_LEVEL_PROXIMITY_PCT."""
        threshold = self.MIN_LEVEL_PROXIMITY_PCT * close
        best: KeyLevel | None = None
        best_dist = float("inf")

        for lvl in levels:
            if lvl.level_type not in ("support", "pdl", "or_low", "weekly_low", "vwap"):
                continue
            dist = abs(lvl.price - close)
            if dist <= threshold and dist < best_dist:
                best_dist = dist
                best = lvl

        return best

    def _htf_bias(
        self,
        df_htf: pd.DataFrame | None,
        structure: MarketStructure | None,
    ) -> TrendDirection:
        """Determine higher-timeframe bias; fall back to primary structure."""
        if structure is not None:
            return structure.trend
        return TrendDirection.UNKNOWN
