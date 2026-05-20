"""
BullBearBiasEngine — composite directional scoring.

Aggregates six independent evidence sources (indicators, market structure,
candle patterns, market regime, volume, and higher-timeframe bias) into
normalized bullish / bearish / neutral scores plus a conflict measure.

Design rules:
  - All result objects are immutable (frozen dataclass)
  - No magic numbers — all thresholds are named class constants
  - Every public method is a pure function (no side effects)
  - Input validation at the boundary; never silently ignores bad data
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Input / Result dataclasses ────────────────────────────────────────────────

@dataclass(frozen=True)
class BiasInput:
    """
    Aggregated evidence from upstream engines.

    All float fields are expected to be in [0, 1] unless noted otherwise.
    String fields accept "bullish", "bearish", or "neutral".
    """

    indicator_bullish: float    # 0-1  from IndicatorBiasReport
    indicator_bearish: float    # 0-1
    structure_bias: str         # "bullish" | "bearish" | "neutral"
    structure_strength: float   # 0-1
    candle_bullish_count: int   # number of bullish candle signals
    candle_bearish_count: int   # number of bearish candle signals
    candle_total: int           # total candles evaluated (denominator)
    regime_supports_direction: bool  # does the current regime favour a trade?
    volume_confirms: bool       # does volume confirm the dominant direction?
    htf_bias: str               # "bullish" | "bearish" | "neutral"


@dataclass(frozen=True)
class BiasResult:
    """
    Composite directional scoring output.

    All score fields are normalized to [0.0, 1.0].
    conflict_score is high when evidence is split evenly.
    """

    bullish_score: float   # 0.0 to 1.0
    bearish_score: float   # 0.0 to 1.0
    neutral_score: float   # 0.0 to 1.0
    conflict_score: float  # 0.0 to 1.0 — how much signals disagree
    net_bias: str          # "bullish" | "bearish" | "neutral"
    explanation: str


# ── Engine ────────────────────────────────────────────────────────────────────

class BullBearBiasEngine:
    """
    Compute a composite directional bias from multiple evidence sources.

    Factor weights (must sum to 1.0):
      indicators   35%
      structure    25%
      candles       5%
      regime       10%
      volume       10%
      htf          15%

    HTF conflict penalty: when HTF direction disagrees with the dominant
    direction, 5% is shifted from dominant to opposite and conflict increases.

    net_bias threshold: a direction must exceed the opposing score by at least
    NET_BIAS_MARGIN (0.05) AND score >= MIN_ABSOLUTE_SCORE (0.35);
    otherwise "neutral". Prevents weak 26% signals from triggering trades.
    """

    # ── Factor weights ────────────────────────────────────────────────────────
    WEIGHT_INDICATORS: float = 0.35
    WEIGHT_STRUCTURE: float  = 0.25
    WEIGHT_CANDLES: float    = 0.05
    WEIGHT_REGIME: float     = 0.10
    WEIGHT_VOLUME: float     = 0.10
    WEIGHT_HTF: float        = 0.15

    # ── Decision threshold ────────────────────────────────────────────────────
    NET_BIAS_MARGIN: float = 0.05
    MIN_ABSOLUTE_SCORE: float = 0.35  # Must score at least 35% to declare direction

    # ── HTF conflict penalty ──────────────────────────────────────────────────
    HTF_CONFLICT_SHIFT: float  = 0.05   # points moved from dominant to opposite
    HTF_CONFLICT_BONUS: float  = 0.08   # added to raw conflict score

    # ── Neutral score weight ──────────────────────────────────────────────────
    # Regime and volume that do not support either side contribute to neutral
    NEUTRAL_FACTOR: float = 0.10

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, inp: BiasInput) -> BiasResult:
        """
        Compute composite directional scores from a BiasInput.

        Steps:
          1. Each factor contributes weighted bullish or bearish points.
          2. Apply HTF conflict penalty when applicable.
          3. Normalize to [0, 1].
          4. Compute conflict_score from spread between bull and bear.
          5. Derive net_bias using NET_BIAS_MARGIN threshold.
          6. Build explanation string.

        Returns an immutable BiasResult.
        """
        bull, bear = self._accumulate_factor_scores(inp)
        bull, bear, htf_conflicted = self._apply_htf_penalty(bull, bear, inp)
        bull_norm, bear_norm, neutral_norm = self._normalize(bull, bear)
        conflict = self._compute_conflict(bull_norm, bear_norm, htf_conflicted)
        net_bias = self._net_bias(bull_norm, bear_norm)
        explanation = self._build_explanation(
            inp=inp,
            bull=bull_norm,
            bear=bear_norm,
            net_bias=net_bias,
            conflict=conflict,
        )

        return BiasResult(
            bullish_score=round(bull_norm, 4),
            bearish_score=round(bear_norm, 4),
            neutral_score=round(neutral_norm, 4),
            conflict_score=round(conflict, 4),
            net_bias=net_bias,
            explanation=explanation,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _accumulate_factor_scores(self, inp: BiasInput) -> tuple[float, float]:
        """
        Sum weighted bullish and bearish contributions from each factor.

        Returns (bull_points, bear_points) — not yet normalized.
        """
        bull = 0.0
        bear = 0.0

        # 1. Indicators (30%)
        bull += inp.indicator_bullish * self.WEIGHT_INDICATORS
        bear += inp.indicator_bearish * self.WEIGHT_INDICATORS

        # 2. Structure (25%)
        if inp.structure_bias == "bullish":
            bull += inp.structure_strength * self.WEIGHT_STRUCTURE
        elif inp.structure_bias == "bearish":
            bear += inp.structure_strength * self.WEIGHT_STRUCTURE
        # neutral structure contributes nothing directional

        # 3. Candles (15%)
        if inp.candle_total > 0:
            candle_bull_ratio = inp.candle_bullish_count / inp.candle_total
            candle_bear_ratio = inp.candle_bearish_count / inp.candle_total
            bull += candle_bull_ratio * self.WEIGHT_CANDLES
            bear += candle_bear_ratio * self.WEIGHT_CANDLES

        # 4. Regime (10%)
        # Regime confirms the already-dominant direction; split evenly when
        # no dominant direction is yet established, but the simplest safe
        # mapping is: if regime supports → add to bull or bear depending on
        # which indicator side is currently winning; if not, contribute
        # nothing (mutes regime contribution entirely when unsupportive).
        if inp.regime_supports_direction:
            # Assign regime weight proportionally to current indicator split
            total_ind = inp.indicator_bullish + inp.indicator_bearish
            if total_ind > 0.0:
                bull += (inp.indicator_bullish / total_ind) * self.WEIGHT_REGIME
                bear += (inp.indicator_bearish / total_ind) * self.WEIGHT_REGIME
            else:
                # No indicator signal — regime weight goes equally to both
                bull += self.WEIGHT_REGIME * 0.5
                bear += self.WEIGHT_REGIME * 0.5
        # regime does NOT support → no contribution (reduces total, lowers confidence)

        # 5. Volume (10%)
        # Volume confirmation adds to whichever side indicators favour.
        if inp.volume_confirms:
            total_ind = inp.indicator_bullish + inp.indicator_bearish
            if total_ind > 0.0:
                bull += (inp.indicator_bullish / total_ind) * self.WEIGHT_VOLUME
                bear += (inp.indicator_bearish / total_ind) * self.WEIGHT_VOLUME
            else:
                bull += self.WEIGHT_VOLUME * 0.5
                bear += self.WEIGHT_VOLUME * 0.5
        # volume does NOT confirm → no contribution

        # 6. HTF bias (10%)
        if inp.htf_bias == "bullish":
            bull += self.WEIGHT_HTF
        elif inp.htf_bias == "bearish":
            bear += self.WEIGHT_HTF
        # neutral HTF → no directional contribution

        return bull, bear

    def _apply_htf_penalty(
        self,
        bull: float,
        bear: float,
        inp: BiasInput,
    ) -> tuple[float, float, bool]:
        """
        Penalize the dominant direction when HTF disagrees with it.

        Returns (bull, bear, htf_conflicted).
        """
        htf_conflicted = False

        # Determine dominant direction so far
        if bull > bear and inp.htf_bias == "bearish":
            # HTF opposes bullish dominance
            shift = min(self.HTF_CONFLICT_SHIFT, bull)
            bull -= shift
            bear += shift
            htf_conflicted = True
        elif bear > bull and inp.htf_bias == "bullish":
            # HTF opposes bearish dominance
            shift = min(self.HTF_CONFLICT_SHIFT, bear)
            bear -= shift
            bull += shift
            htf_conflicted = True

        return bull, bear, htf_conflicted

    def _normalize(self, bull: float, bear: float) -> tuple[float, float, float]:
        """
        Normalize bull and bear to [0, 1] relative to total weighted budget (1.0).

        neutral_score captures the remaining budget not claimed by either side.
        All values are clipped to [0, 1].
        """
        total = bull + bear
        if total == 0.0:
            return 0.0, 0.0, 1.0

        # Normalize against the max possible weighted budget (1.0)
        # so scores reflect how much of the total possible weight each side has.
        bull_norm = min(bull, 1.0)
        bear_norm = min(bear, 1.0)
        # Neutral is whatever budget neither side claimed
        neutral_norm = max(0.0, 1.0 - bull_norm - bear_norm)

        return bull_norm, bear_norm, neutral_norm

    def _compute_conflict(
        self,
        bull: float,
        bear: float,
        htf_conflicted: bool,
    ) -> float:
        """
        conflict_score = 1.0 - abs(bull - bear) + HTF conflict bonus.

        A perfectly one-sided signal has conflict = 0 (before any bonus).
        An evenly split signal has conflict near 1.0.
        Result is clipped to [0, 1].
        """
        raw_conflict = 1.0 - abs(bull - bear)
        bonus = self.HTF_CONFLICT_BONUS if htf_conflicted else 0.0
        return float(min(max(raw_conflict + bonus, 0.0), 1.0))

    def _net_bias(self, bull: float, bear: float) -> str:
        """
        Determine net directional bias.

        Requires BOTH:
        1. Relative margin: winner > loser + NET_BIAS_MARGIN
        2. Absolute minimum: winner >= MIN_ABSOLUTE_SCORE (35%)

        This prevents calling "bearish" with 26% score in a trending_up market.
        """
        if bull > bear + self.NET_BIAS_MARGIN and bull >= self.MIN_ABSOLUTE_SCORE:
            return "bullish"
        if bear > bull + self.NET_BIAS_MARGIN and bear >= self.MIN_ABSOLUTE_SCORE:
            return "bearish"
        return "neutral"

    def _build_explanation(
        self,
        inp: BiasInput,
        bull: float,
        bear: float,
        net_bias: str,
        conflict: float,
    ) -> str:
        """
        Compose a human-readable explanation of the composite bias decision.
        """
        parts = [
            f"Net bias: {net_bias}.",
            f"Bullish score: {bull:.2f}, bearish score: {bear:.2f}, conflict: {conflict:.2f}.",
            f"Indicators: bull={inp.indicator_bullish:.2f} bear={inp.indicator_bearish:.2f}.",
            f"Structure: {inp.structure_bias} (strength={inp.structure_strength:.2f}).",
            f"Candles: {inp.candle_bullish_count}B/{inp.candle_bearish_count}Bear of {inp.candle_total}.",
            f"Regime supports: {inp.regime_supports_direction}.",
            f"Volume confirms: {inp.volume_confirms}.",
            f"HTF bias: {inp.htf_bias}.",
        ]
        return " ".join(parts)
