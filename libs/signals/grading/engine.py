"""
Trade Decision Engine.

Grades a trade setup on a 100-point scale and maps the score to a letter
grade (A+/A/B/C/Avoid) and a trade decision (TAKE/WAIT/SKIP/NO_TRADE).

Design rules:
  - All inputs are pure value objects (frozen dataclasses).
  - No magic numbers — all weights and thresholds are named constants.
  - Never raises — returns a GradingResult on any valid GradingInput.
  - Output is a frozen dataclass (GradingResult).
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Score component weights / caps ────────────────────────────────────────────

_CONFIDENCE_MAX_PTS: float = 25.0
_RR_MAX_PTS: float = 20.0
_RR_SCALE: float = 6.67          # pts per unit of R:R up to the cap

_BIAS_ALIGNED_PTS: float = 15.0
_BIAS_NEUTRAL_PTS: float = 5.0
_BIAS_CONFLICT_PTS: float = 0.0

_CONFLICT_HIGH_THRESHOLD: float = 0.6
_CONFLICT_MID_THRESHOLD: float = 0.4
_CONFLICT_HIGH_PENALTY: float = -10.0
_CONFLICT_MID_PENALTY: float = -5.0

_HTF_ALIGNED_PTS: float = 10.0
_HTF_UNALIGNED_PENALTY: float = -5.0

_REGIME_SUPPORTS_PTS: float = 10.0

_STRUCTURE_MAX_PTS: float = 10.0

_VOLUME_CONFIRMS_PTS: float = 5.0

_DATA_QUALITY_CLEAN_PTS: float = 5.0
_DATA_QUALITY_DIRTY_PENALTY: float = -10.0

_LATE_ENTRY_PENALTY: float = -15.0
_OVEREXTENDED_PENALTY: float = -15.0

# ── Grade thresholds ──────────────────────────────────────────────────────────

_GRADE_A_PLUS_MIN: float = 75.0
_GRADE_A_MIN: float = 60.0
_GRADE_B_MIN: float = 45.0
_GRADE_C_MIN: float = 30.0

# ── Grade / decision mappings ─────────────────────────────────────────────────

_GRADE_TO_DECISION: dict[str, str] = {
    "A+": "TAKE",
    "A": "TAKE",
    "B": "WAIT",
    "C": "SKIP",
    "Avoid": "NO_TRADE",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GradingInput:
    """All inputs required to grade a trade setup."""

    confidence: float          # 0-1
    risk_reward: float         # R:R ratio
    bias_net: str              # "bullish" / "bearish" / "neutral"
    bias_conflict: float       # 0-1
    bias_bullish: float        # 0-1
    bias_bearish: float        # 0-1
    htf_aligned: bool
    regime_supports: bool
    structure_strength: float  # 0-1
    volume_confirms: bool
    data_quality_clean: bool
    action: str                # "BUY" or "SELL"
    is_late_entry: bool
    is_overextended: bool


@dataclass(frozen=True)
class GradingResult:
    """Immutable result produced by TradeDecisionEngine."""

    setup_grade: str        # "A+", "A", "B", "C", "Avoid"
    trade_decision: str     # "TAKE", "WAIT", "SKIP", "NO_TRADE"
    quality_score: float    # 0-100
    reasons: list[str]


# ── Engine ────────────────────────────────────────────────────────────────────

class TradeDecisionEngine:
    """
    Grades a trade setup and emits a TAKE/WAIT/SKIP/NO_TRADE decision.

    Usage::

        engine = TradeDecisionEngine()
        result = engine.grade(grading_input)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def grade(self, inp: GradingInput) -> GradingResult:
        """
        Grade a trade setup.

        Parameters
        ----------
        inp:
            Fully populated GradingInput.

        Returns
        -------
        GradingResult
            Frozen result with grade, decision, score, and reasons list.
        """
        score: float = 0.0
        reasons: list[str] = []

        # ── Confidence component ──────────────────────────────────────────────
        confidence_pts = inp.confidence * _CONFIDENCE_MAX_PTS
        score += confidence_pts
        reasons.append(
            f"confidence {inp.confidence:.2f} → +{confidence_pts:.1f} pts"
        )

        # ── R:R component ─────────────────────────────────────────────────────
        rr_pts = min(_RR_MAX_PTS, inp.risk_reward * _RR_SCALE)
        score += rr_pts
        reasons.append(f"R:R {inp.risk_reward:.2f} → +{rr_pts:.1f} pts")

        # ── Bias alignment ────────────────────────────────────────────────────
        bias_pts, bias_note = self._score_bias(inp)
        score += bias_pts
        reasons.append(bias_note)

        # ── Conflict penalty ──────────────────────────────────────────────────
        conflict_pts, conflict_note = self._score_conflict(inp.bias_conflict)
        if conflict_pts != 0.0:
            score += conflict_pts
            reasons.append(conflict_note)

        # ── HTF alignment ─────────────────────────────────────────────────────
        if inp.htf_aligned:
            score += _HTF_ALIGNED_PTS
            reasons.append(f"HTF aligned → +{_HTF_ALIGNED_PTS:.0f} pts")
        else:
            score += _HTF_UNALIGNED_PENALTY
            reasons.append(f"HTF not aligned → {_HTF_UNALIGNED_PENALTY:.0f} pts")

        # ── Regime ───────────────────────────────────────────────────────────
        if inp.regime_supports:
            score += _REGIME_SUPPORTS_PTS
            reasons.append(f"regime supports → +{_REGIME_SUPPORTS_PTS:.0f} pts")

        # ── Structure strength ────────────────────────────────────────────────
        structure_pts = inp.structure_strength * _STRUCTURE_MAX_PTS
        score += structure_pts
        reasons.append(
            f"structure strength {inp.structure_strength:.2f} → +{structure_pts:.1f} pts"
        )

        # ── Volume confirmation ───────────────────────────────────────────────
        if inp.volume_confirms:
            score += _VOLUME_CONFIRMS_PTS
            reasons.append(f"volume confirms → +{_VOLUME_CONFIRMS_PTS:.0f} pts")

        # ── Data quality ──────────────────────────────────────────────────────
        if inp.data_quality_clean:
            score += _DATA_QUALITY_CLEAN_PTS
            reasons.append(f"data quality clean → +{_DATA_QUALITY_CLEAN_PTS:.0f} pts")
        else:
            score += _DATA_QUALITY_DIRTY_PENALTY
            reasons.append(
                f"data quality not clean → {_DATA_QUALITY_DIRTY_PENALTY:.0f} pts"
            )

        # ── Late entry penalty ────────────────────────────────────────────────
        if inp.is_late_entry:
            score += _LATE_ENTRY_PENALTY
            reasons.append(f"late entry → {_LATE_ENTRY_PENALTY:.0f} pts")

        # ── Overextended penalty ──────────────────────────────────────────────
        if inp.is_overextended:
            score += _OVEREXTENDED_PENALTY
            reasons.append(f"overextended → {_OVEREXTENDED_PENALTY:.0f} pts")

        # ── Clamp to [0, 100] ─────────────────────────────────────────────────
        quality_score = float(max(0.0, min(100.0, score)))

        grade = _score_to_grade(quality_score)
        decision = _GRADE_TO_DECISION[grade]

        return GradingResult(
            setup_grade=grade,
            trade_decision=decision,
            quality_score=quality_score,
            reasons=reasons,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _score_bias(inp: GradingInput) -> tuple[float, str]:
        """Return (pts, note) for bias alignment with proposed action."""
        action_is_buy = inp.action.upper() == "BUY"
        bias = inp.bias_net.lower()

        if bias == "neutral":
            return _BIAS_NEUTRAL_PTS, f"bias neutral → +{_BIAS_NEUTRAL_PTS:.0f} pts"

        aligned = (action_is_buy and bias == "bullish") or (
            not action_is_buy and bias == "bearish"
        )
        if aligned:
            return (
                _BIAS_ALIGNED_PTS,
                f"bias {bias} aligns with {inp.action} → +{_BIAS_ALIGNED_PTS:.0f} pts",
            )

        return (
            _BIAS_CONFLICT_PTS,
            f"bias {bias} conflicts with {inp.action} → +{_BIAS_CONFLICT_PTS:.0f} pts",
        )

    @staticmethod
    def _score_conflict(bias_conflict: float) -> tuple[float, str]:
        """Return (pts, note) for the bias-conflict penalty."""
        if bias_conflict > _CONFLICT_HIGH_THRESHOLD:
            return (
                _CONFLICT_HIGH_PENALTY,
                f"bias conflict {bias_conflict:.2f} > {_CONFLICT_HIGH_THRESHOLD} "
                f"→ {_CONFLICT_HIGH_PENALTY:.0f} pts",
            )
        if bias_conflict > _CONFLICT_MID_THRESHOLD:
            return (
                _CONFLICT_MID_PENALTY,
                f"bias conflict {bias_conflict:.2f} > {_CONFLICT_MID_THRESHOLD} "
                f"→ {_CONFLICT_MID_PENALTY:.0f} pts",
            )
        return 0.0, ""


# ── Module-level helper ───────────────────────────────────────────────────────

def _score_to_grade(score: float) -> str:
    """Map a 0-100 quality score to a letter grade."""
    if score >= _GRADE_A_PLUS_MIN:
        return "A+"
    if score >= _GRADE_A_MIN:
        return "A"
    if score >= _GRADE_B_MIN:
        return "B"
    if score >= _GRADE_C_MIN:
        return "C"
    return "Avoid"
