"""
Indicator Bias Analyzer.

Interprets raw technical indicator values (from IndicatorSnapshot) and
produces per-indicator bias (bullish/bearish/neutral), strength (0-1), and a
human-readable explanation.  Also aggregates all indicators into an
IndicatorBiasReport with normalized scores and an overall net_bias.

Design rules:
  - All result objects are immutable (frozen dataclass)
  - No magic numbers — all thresholds are named class constants
  - Every public method is a pure function (no side effects)
  - analyze_all accepts keyword-only arguments to prevent positional mistakes
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IndicatorBias:
    """Bias reading for a single technical indicator."""

    name: str          # "RSI", "MACD", "Bollinger", "EMA", "ADX", "Volume"
    direction: str     # "bullish", "bearish", "neutral"
    strength: float    # 0.0 to 1.0
    explanation: str   # human-readable summary


@dataclass(frozen=True)
class IndicatorBiasReport:
    """Aggregated bias across all indicators."""

    indicators: list[IndicatorBias]
    bullish_score: float   # 0.0 to 1.0 — normalized sum of bullish strengths
    bearish_score: float   # 0.0 to 1.0 — normalized sum of bearish strengths
    neutral_score: float   # 0.0 to 1.0 — normalized sum of neutral strengths
    net_bias: str          # "bullish", "bearish", "neutral"
    explanation: str       # human-readable overall summary


# ── Analyzer ──────────────────────────────────────────────────────────────────

class IndicatorBiasAnalyzer:
    """
    Interprets technical indicator values and outputs directional bias.

    Each method returns an IndicatorBias.  analyze_all() aggregates all six
    indicators into an IndicatorBiasReport.
    """

    # ── RSI thresholds ────────────────────────────────────────────────────────
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    RSI_EXTREME_OVERSOLD: float = 20.0
    RSI_EXTREME_OVERBOUGHT: float = 80.0

    # ── Bollinger Band thresholds ─────────────────────────────────────────────
    BB_STRONG_LOWER: float = 0.0    # pct_b <= this → strong bullish
    BB_STRONG_UPPER: float = 1.0    # pct_b >= this → strong bearish
    BB_NEAR_LOWER: float = 0.2      # pct_b <= this → mild bullish
    BB_NEAR_UPPER: float = 0.8      # pct_b >= this → mild bearish

    # ── ADX thresholds ────────────────────────────────────────────────────────
    ADX_STRONG: float = 25.0
    ADX_VERY_STRONG: float = 40.0
    ADX_WEAK: float = 20.0

    # ── Volume thresholds ─────────────────────────────────────────────────────
    VOLUME_SPIKE: float = 1.5       # relative_volume >= this → significant
    VOLUME_STRONG_SPIKE: float = 2.0

    # ── Aggregation threshold ─────────────────────────────────────────────────
    NET_BIAS_MARGIN: float = 0.15   # bullish must exceed bearish by this to win

    # ── RSI ───────────────────────────────────────────────────────────────────

    def rsi_bias(self, rsi: float, rsi_prev: float) -> IndicatorBias:
        """
        Classify RSI as bullish (oversold), bearish (overbought), or neutral.

        Strength scales with distance from the extreme threshold.
        When RSI is in the middle zone (30-70), direction is determined by
        whether RSI is rising or falling relative to the previous bar.
        """
        if rsi <= self.RSI_OVERSOLD:
            # Deeper into oversold → stronger bullish signal
            excess = self.RSI_OVERSOLD - rsi
            max_excess = self.RSI_OVERSOLD - self.RSI_EXTREME_OVERSOLD
            base_strength = 0.5 + min(excess / max_excess, 1.0) * 0.5
            return IndicatorBias(
                name="RSI",
                direction="bullish",
                strength=round(min(base_strength, 1.0), 4),
                explanation=f"RSI {rsi:.1f} is in oversold territory (<{self.RSI_OVERSOLD}), "
                            "suggesting a potential bullish reversal.",
            )

        if rsi >= self.RSI_OVERBOUGHT:
            excess = rsi - self.RSI_OVERBOUGHT
            max_excess = self.RSI_EXTREME_OVERBOUGHT - self.RSI_OVERBOUGHT
            base_strength = 0.5 + min(excess / max_excess, 1.0) * 0.5
            return IndicatorBias(
                name="RSI",
                direction="bearish",
                strength=round(min(base_strength, 1.0), 4),
                explanation=f"RSI {rsi:.1f} is in overbought territory (>{self.RSI_OVERBOUGHT}), "
                            "suggesting potential bearish pressure.",
            )

        # Middle zone: direction follows momentum
        mid = 50.0
        distance_from_mid = abs(rsi - mid) / (self.RSI_OVERBOUGHT - mid)
        strength = round(distance_from_mid * 0.4, 4)  # max 0.4 in neutral zone
        if rsi > mid:
            direction = "bullish" if rsi_prev <= rsi else "neutral"
        elif rsi < mid:
            direction = "bearish" if rsi_prev >= rsi else "neutral"
        else:
            direction = "neutral"

        return IndicatorBias(
            name="RSI",
            direction=direction if direction != "neutral" else "neutral",
            strength=strength if direction != "neutral" else round(strength * 0.5, 4),
            explanation=f"RSI {rsi:.1f} is in neutral territory (30-70).",
        )

    # ── MACD ──────────────────────────────────────────────────────────────────

    def macd_bias(
        self,
        macd_line: float,
        macd_signal: float,
        histogram: float,
        histogram_prev: float,
    ) -> IndicatorBias:
        """
        Classify MACD as bullish/bearish/neutral.

        Bullish crossover: histogram flips from negative to positive.
        Bearish crossunder: histogram flips from positive to negative.
        Ongoing positive histogram with rising momentum → bullish.
        Ongoing negative histogram → bearish.
        """
        crossover = histogram > 0.0 and histogram_prev <= 0.0
        crossunder = histogram < 0.0 and histogram_prev >= 0.0

        if crossover:
            return IndicatorBias(
                name="MACD",
                direction="bullish",
                strength=0.85,
                explanation=f"MACD bullish crossover detected — histogram flipped positive "
                            f"({histogram_prev:.4f} → {histogram:.4f}).",
            )

        if crossunder:
            return IndicatorBias(
                name="MACD",
                direction="bearish",
                strength=0.85,
                explanation=f"MACD bearish crossunder detected — histogram flipped negative "
                            f"({histogram_prev:.4f} → {histogram:.4f}).",
            )

        if histogram > 0.0:
            # Positive histogram: bullish; strength grows with line above signal
            line_diff = macd_line - macd_signal
            # Normalize by absolute signal to get a relative measure
            ref = abs(macd_signal) if macd_signal != 0.0 else 1.0
            strength = round(min(0.4 + abs(line_diff) / ref * 0.3, 0.8), 4)
            direction = "bullish"
            explanation = (
                f"MACD histogram positive ({histogram:.4f}); "
                f"line ({macd_line:.4f}) above signal ({macd_signal:.4f})."
            )
        elif histogram < 0.0:
            line_diff = macd_signal - macd_line
            ref = abs(macd_signal) if macd_signal != 0.0 else 1.0
            strength = round(min(0.4 + abs(line_diff) / ref * 0.3, 0.8), 4)
            direction = "bearish"
            explanation = (
                f"MACD histogram negative ({histogram:.4f}); "
                f"line ({macd_line:.4f}) below signal ({macd_signal:.4f})."
            )
        else:
            direction = "neutral"
            strength = 0.1
            explanation = "MACD histogram at zero — no clear directional bias."

        return IndicatorBias(name="MACD", direction=direction, strength=strength, explanation=explanation)

    # ── Bollinger Bands ───────────────────────────────────────────────────────

    def bollinger_bias(
        self,
        close: float,
        bb_upper: float,
        bb_lower: float,
        bb_pct_b: float,
    ) -> IndicatorBias:
        """
        Classify Bollinger Band position as bullish/bearish/neutral.

        Price below lower band → bullish (mean-reversion opportunity).
        Price above upper band → bearish (overextended / reversal risk).
        Inside bands: proximity to either edge influences strength.
        """
        if close <= bb_lower:
            excess = bb_lower - close
            band_range = bb_upper - bb_lower
            extra_strength = min(excess / band_range, 0.5) if band_range > 0.0 else 0.0
            strength = round(min(0.7 + extra_strength, 1.0), 4)
            return IndicatorBias(
                name="Bollinger",
                direction="bullish",
                strength=strength,
                explanation=f"Price ({close:.2f}) is below lower Bollinger Band ({bb_lower:.2f}); "
                            "oversold condition suggests mean-reversion upside.",
            )

        if close >= bb_upper:
            excess = close - bb_upper
            band_range = bb_upper - bb_lower
            extra_strength = min(excess / band_range, 0.5) if band_range > 0.0 else 0.0
            strength = round(min(0.7 + extra_strength, 1.0), 4)
            return IndicatorBias(
                name="Bollinger",
                direction="bearish",
                strength=strength,
                explanation=f"Price ({close:.2f}) is above upper Bollinger Band ({bb_upper:.2f}); "
                            "overbought condition suggests mean-reversion downside.",
            )

        # Inside bands — use pct_b to determine lean
        if bb_pct_b <= self.BB_NEAR_LOWER:
            strength = round(0.3 + (self.BB_NEAR_LOWER - bb_pct_b) / self.BB_NEAR_LOWER * 0.3, 4)
            return IndicatorBias(
                name="Bollinger",
                direction="bullish",
                strength=min(strength, 0.6),
                explanation=f"Price near lower band (pct_b={bb_pct_b:.2f}); mild bullish lean.",
            )

        if bb_pct_b >= self.BB_NEAR_UPPER:
            strength = round(0.3 + (bb_pct_b - self.BB_NEAR_UPPER) / (1.0 - self.BB_NEAR_UPPER) * 0.3, 4)
            return IndicatorBias(
                name="Bollinger",
                direction="bearish",
                strength=min(strength, 0.6),
                explanation=f"Price near upper band (pct_b={bb_pct_b:.2f}); mild bearish lean.",
            )

        return IndicatorBias(
            name="Bollinger",
            direction="neutral",
            strength=0.1,
            explanation=f"Price inside Bollinger Bands (pct_b={bb_pct_b:.2f}); no strong directional signal.",
        )

    # ── EMA ───────────────────────────────────────────────────────────────────

    def ema_bias(
        self,
        ema_9: float,
        ema_20: float,
        ema_50: float,
        close: float,
    ) -> IndicatorBias:
        """
        Classify EMA alignment as bullish/bearish/neutral.

        Stacked bullish: ema_9 > ema_20 > ema_50, and close > ema_9.
        Stacked bearish: ema_9 < ema_20 < ema_50, and close < ema_9.
        Partial alignment → moderate strength.
        """
        stacked_bull = ema_9 > ema_20 > ema_50
        stacked_bear = ema_9 < ema_20 < ema_50
        price_above_fast = close > ema_9
        price_below_fast = close < ema_9

        if stacked_bull and price_above_fast:
            return IndicatorBias(
                name="EMA",
                direction="bullish",
                strength=0.9,
                explanation=f"EMAs stacked bullishly (9>{ema_9:.2f} > 20>{ema_20:.2f} > 50>{ema_50:.2f}) "
                            f"and price ({close:.2f}) above EMA-9.",
            )

        if stacked_bull:
            return IndicatorBias(
                name="EMA",
                direction="bullish",
                strength=0.6,
                explanation=f"EMAs stacked bullishly but price ({close:.2f}) below EMA-9 ({ema_9:.2f}).",
            )

        if stacked_bear and price_below_fast:
            return IndicatorBias(
                name="EMA",
                direction="bearish",
                strength=0.9,
                explanation=f"EMAs stacked bearishly (9<{ema_9:.2f} < 20<{ema_20:.2f} < 50<{ema_50:.2f}) "
                            f"and price ({close:.2f}) below EMA-9.",
            )

        if stacked_bear:
            return IndicatorBias(
                name="EMA",
                direction="bearish",
                strength=0.6,
                explanation=f"EMAs stacked bearishly but price ({close:.2f}) above EMA-9 ({ema_9:.2f}).",
            )

        # Mixed alignment
        bullish_signals = sum([ema_9 > ema_20, ema_20 > ema_50, price_above_fast])
        bearish_signals = sum([ema_9 < ema_20, ema_20 < ema_50, price_below_fast])

        if bullish_signals > bearish_signals:
            return IndicatorBias(
                name="EMA",
                direction="bullish",
                strength=0.35,
                explanation=f"EMAs partially aligned bullishly ({bullish_signals}/3 signals).",
            )
        if bearish_signals > bullish_signals:
            return IndicatorBias(
                name="EMA",
                direction="bearish",
                strength=0.35,
                explanation=f"EMAs partially aligned bearishly ({bearish_signals}/3 signals).",
            )

        return IndicatorBias(
            name="EMA",
            direction="neutral",
            strength=0.1,
            explanation="EMA alignment is mixed — no clear directional bias.",
        )

    # ── ADX ───────────────────────────────────────────────────────────────────

    def adx_bias(self, adx: float) -> IndicatorBias:
        """
        ADX is non-directional — it measures trend strength only.

        > 25  → strong trend (high strength)
        < 20  → weak/ranging market (low strength)
        Direction is always 'neutral' since ADX does not indicate direction.
        """
        if adx >= self.ADX_VERY_STRONG:
            strength = round(min(0.8 + (adx - self.ADX_VERY_STRONG) / 60.0 * 0.2, 1.0), 4)
            explanation = f"ADX {adx:.1f} — very strong trend in progress."
        elif adx >= self.ADX_STRONG:
            excess = adx - self.ADX_STRONG
            range_ = self.ADX_VERY_STRONG - self.ADX_STRONG
            strength = round(0.6 + excess / range_ * 0.2, 4)
            explanation = f"ADX {adx:.1f} — strong trend (>25)."
        elif adx >= self.ADX_WEAK:
            excess = adx - self.ADX_WEAK
            range_ = self.ADX_STRONG - self.ADX_WEAK
            strength = round(0.3 + excess / range_ * 0.3, 4)
            explanation = f"ADX {adx:.1f} — moderate trend strength."
        else:
            # Weak
            strength = round(adx / self.ADX_WEAK * 0.3, 4)
            explanation = f"ADX {adx:.1f} — weak trend or ranging market (<{self.ADX_WEAK})."

        return IndicatorBias(
            name="ADX",
            direction="neutral",
            strength=strength,
            explanation=explanation,
        )

    # ── Volume ────────────────────────────────────────────────────────────────

    def volume_bias(self, relative_volume: float, is_bullish_candle: bool) -> IndicatorBias:
        """
        Volume spike on a bullish candle → bullish conviction.
        Volume spike on a bearish candle → bearish conviction.
        Low or average volume → neutral.
        """
        if relative_volume >= self.VOLUME_SPIKE:
            raw_strength = min((relative_volume - 1.0) / 2.0, 1.0)
            strength = round(0.4 + raw_strength * 0.6, 4)
            if is_bullish_candle:
                return IndicatorBias(
                    name="Volume",
                    direction="bullish",
                    strength=min(strength, 1.0),
                    explanation=f"Volume spike ({relative_volume:.1f}x average) on a bullish candle "
                                "confirms buying conviction.",
                )
            return IndicatorBias(
                name="Volume",
                direction="bearish",
                strength=min(strength, 1.0),
                explanation=f"Volume spike ({relative_volume:.1f}x average) on a bearish candle "
                            "confirms selling conviction.",
            )

        # Below spike threshold — mild lean based on candle direction
        strength = round(relative_volume / self.VOLUME_SPIKE * 0.3, 4)
        direction = "bullish" if is_bullish_candle else "bearish"
        return IndicatorBias(
            name="Volume",
            direction=direction,
            strength=strength,
            explanation=f"Volume ({relative_volume:.1f}x average) below spike threshold; weak directional signal.",
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────

    def analyze_all(
        self,
        *,
        rsi: float,
        rsi_prev: float,
        macd_line: float,
        macd_signal: float,
        histogram: float,
        histogram_prev: float,
        close: float,
        bb_upper: float,
        bb_lower: float,
        bb_pct_b: float,
        ema_9: float,
        ema_20: float,
        ema_50: float,
        adx: float,
        relative_volume: float,
        is_bullish_candle: bool,
    ) -> IndicatorBiasReport:
        """
        Run all six indicator bias methods and aggregate into an
        IndicatorBiasReport.

        Scoring:
          - Sum bullish strengths, sum bearish strengths.
          - Normalize each by total possible (max 6.0) to produce 0-1 scores.
          - net_bias = "bullish" if bullish_score > bearish_score + NET_BIAS_MARGIN
                       "bearish" if bearish_score > bullish_score + NET_BIAS_MARGIN
                       else "neutral"
        """
        indicators = [
            self.rsi_bias(rsi=rsi, rsi_prev=rsi_prev),
            self.macd_bias(
                macd_line=macd_line,
                macd_signal=macd_signal,
                histogram=histogram,
                histogram_prev=histogram_prev,
            ),
            self.bollinger_bias(
                close=close,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_pct_b=bb_pct_b,
            ),
            self.ema_bias(ema_9=ema_9, ema_20=ema_20, ema_50=ema_50, close=close),
            self.adx_bias(adx=adx),
            self.volume_bias(relative_volume=relative_volume, is_bullish_candle=is_bullish_candle),
        ]

        max_possible = float(len(indicators))
        bullish_sum = sum(b.strength for b in indicators if b.direction == "bullish")
        bearish_sum = sum(b.strength for b in indicators if b.direction == "bearish")
        neutral_sum = sum(b.strength for b in indicators if b.direction == "neutral")

        bullish_score = round(bullish_sum / max_possible, 4)
        bearish_score = round(bearish_sum / max_possible, 4)
        neutral_score = round(neutral_sum / max_possible, 4)

        if bullish_score > bearish_score + self.NET_BIAS_MARGIN:
            net_bias = "bullish"
        elif bearish_score > bullish_score + self.NET_BIAS_MARGIN:
            net_bias = "bearish"
        else:
            net_bias = "neutral"

        bullish_names = [b.name for b in indicators if b.direction == "bullish"]
        bearish_names = [b.name for b in indicators if b.direction == "bearish"]
        explanation = (
            f"Net bias: {net_bias}. "
            f"Bullish indicators: {bullish_names or 'none'}. "
            f"Bearish indicators: {bearish_names or 'none'}."
        )

        return IndicatorBiasReport(
            indicators=indicators,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            neutral_score=neutral_score,
            net_bias=net_bias,
            explanation=explanation,
        )
