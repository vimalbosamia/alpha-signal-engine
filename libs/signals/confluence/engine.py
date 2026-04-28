"""
Confluence Scoring Engine.

Aggregates multiple analytical signals into a single weighted score that
determines whether a SignalCandidate has sufficient confluence to be acted on.

Design rules:
  - All inputs are optional; missing data returns safe mid-range defaults.
  - No magic numbers — all weights and thresholds are named constants.
  - Never raises — returns a ConfluenceBreakdown on any valid input.
  - Output is a frozen Pydantic model (ConfluenceBreakdown).
"""
from __future__ import annotations

from libs.analysis.levels.engine import KeyLevel
from libs.analysis.regime.engine import RegimeAnalysis
from libs.analysis.structure.engine import MarketStructure
from libs.analysis.volume.engine import VolumeContext
from libs.core.models.domain import (
    ConfluenceBreakdown,
    DataQualityStatus,
    SignalAction,
    SignalCandidate,
    TrendDirection,
)
from libs.risk.engine import RiskAssessment

# ── Default weights ────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "pattern":      0.20,
    "structure":    0.20,
    "level":        0.15,
    "volume":       0.15,
    "regime":       0.10,
    "session":      0.08,
    "data_quality": 0.07,
    "risk":         0.05,
}

# ── Scoring thresholds ────────────────────────────────────────────────────────

_HIGH_CONFIDENCE_THRESHOLD: float = 0.8
_PATTERN_HIGH_CONFIDENCE_BONUS: float = 0.1

_COUNTER_TREND_SCORE: float = 0.2
_TRENDING_ALIGNED_SCORE: float = 1.0
_RANGING_SCORE: float = 0.5
_STRUCTURE_BOS_PENALTY: float = 0.2
_STRUCTURE_STRENGTH_BONUS: float = 0.1
_STRUCTURE_STRENGTH_MIN: float = 0.7
_STRUCTURE_NO_DATA_SCORE: float = 0.5

_LEVEL_PROXIMITY_PCT: float = 0.005   # 0.5 %
_LEVEL_NO_DATA_SCORE: float = 0.4
_LEVEL_NO_NEARBY_SCORE: float = 0.3
_LEVEL_BONUS_TYPES: frozenset[str] = frozenset(
    {"support", "resistance", "pdh", "pdl", "vwap"}
)
_LEVEL_TYPE_BONUS: float = 0.1

_VOLUME_NO_DATA_SCORE: float = 0.5
_REGIME_NO_DATA_SCORE: float = 0.5
_SESSION_NO_DATA_SCORE: float = 0.6
_DATA_QUALITY_NO_DATA_SCORE: float = 0.7
_DATA_QUALITY_WARNING_SCORE: float = 0.6
_RISK_NO_DATA_SCORE: float = 0.5


# ── Engine ─────────────────────────────────────────────────────────────────────

class ConfluenceEngine:
    """
    Aggregates factor scores into a ConfluenceBreakdown.

    Parameters
    ----------
    weights:
        Optional dict overriding DEFAULT_WEIGHTS.  Values are normalised to
        sum to 1.0 so callers may supply un-normalised relative weights.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        raw = weights if weights is not None else DEFAULT_WEIGHTS
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        self._weights: dict[str, float] = {k: v / total for k, v in raw.items()}

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(
        self,
        candidate: SignalCandidate,
        structure: MarketStructure | None = None,
        levels: list[KeyLevel] | None = None,
        volume: VolumeContext | None = None,
        regime: RegimeAnalysis | None = None,
        risk: RiskAssessment | None = None,
    ) -> ConfluenceBreakdown:
        """
        Score a SignalCandidate against all available context.

        Returns a fully populated ConfluenceBreakdown.  Never raises.
        """
        blocked_reasons: list[str] = []
        warning_tags: list[str] = []
        factor_notes: dict[str, str] = {}

        # ── Individual scorers ─────────────────────────────────────────────────
        pattern_score, pattern_note = self._score_patterns(candidate)
        factor_notes["pattern"] = pattern_note

        structure_score, structure_note = self._score_structure(candidate, structure)
        factor_notes["structure"] = structure_note

        level_score, level_note = self._score_levels(candidate, levels)
        factor_notes["level"] = level_note

        volume_score, volume_note = self._score_volume(candidate, volume)
        factor_notes["volume"] = volume_note

        regime_score, regime_note = self._score_regime(regime)
        factor_notes["regime"] = regime_note

        session_score, session_note = self._score_session(candidate)
        factor_notes["session"] = session_note

        dq_score, dq_note, dq_blocked, dq_warnings = self._score_data_quality(candidate)
        factor_notes["data_quality"] = dq_note
        blocked_reasons.extend(dq_blocked)
        warning_tags.extend(dq_warnings)

        risk_score, risk_note, risk_blocked = self._score_risk(risk)
        factor_notes["risk"] = risk_note
        blocked_reasons.extend(risk_blocked)

        # ── Weighted total ─────────────────────────────────────────────────────
        scores: dict[str, float] = {
            "pattern":      pattern_score,
            "structure":    structure_score,
            "level":        level_score,
            "volume":       volume_score,
            "regime":       regime_score,
            "session":      session_score,
            "data_quality": dq_score,
            "risk":         risk_score,
        }

        if blocked_reasons:
            weighted_total = 0.0
        else:
            raw_total = sum(
                scores[k] * self._weights.get(k, 0.0) for k in scores
            )
            weighted_total = float(max(0.0, min(1.0, raw_total)))

        return ConfluenceBreakdown(
            pattern_score=pattern_score,
            structure_score=structure_score,
            level_score=level_score,
            volume_score=volume_score,
            regime_score=regime_score,
            session_score=session_score,
            risk_score=risk_score,
            data_quality_score=dq_score,
            weighted_total=weighted_total,
            weights=dict(self._weights),
            factor_notes=factor_notes,
            blocked_reasons=blocked_reasons,
            warning_tags=warning_tags,
        )

    # ── Individual scorers ──────────────────────────────────────────────────────

    def _score_patterns(
        self, candidate: SignalCandidate
    ) -> tuple[float, str]:
        """Average confidence of detected patterns with high-confidence bonus."""
        detected = [p for p in candidate.pattern_results if p.detected]
        if not detected:
            return 0.0, "no detected patterns"

        avg_confidence = sum(p.confidence for p in detected) / len(detected)
        score = avg_confidence

        has_high_confidence = any(
            p.confidence >= _HIGH_CONFIDENCE_THRESHOLD for p in detected
        )
        if has_high_confidence:
            score += _PATTERN_HIGH_CONFIDENCE_BONUS

        score = float(max(0.0, min(1.0, score)))
        names = ", ".join(p.pattern_name for p in detected)
        return score, f"detected: {names}"

    def _score_structure(
        self,
        candidate: SignalCandidate,
        structure: MarketStructure | None,
    ) -> tuple[float, str]:
        """Score trend alignment between structure and proposed action."""
        if structure is None:
            return _STRUCTURE_NO_DATA_SCORE, "no structure data"

        action = candidate.proposed_action
        trend = structure.trend

        if trend == TrendDirection.UPTREND and action == SignalAction.BUY:
            base = _TRENDING_ALIGNED_SCORE
            note = "uptrend + BUY aligned"
        elif trend == TrendDirection.DOWNTREND and action == SignalAction.SELL:
            base = _TRENDING_ALIGNED_SCORE
            note = "downtrend + SELL aligned"
        elif trend == TrendDirection.UPTREND and action == SignalAction.SELL:
            base = _COUNTER_TREND_SCORE
            note = "counter-trend: SELL in uptrend"
        elif trend == TrendDirection.DOWNTREND and action == SignalAction.BUY:
            base = _COUNTER_TREND_SCORE
            note = "counter-trend: BUY in downtrend"
        else:
            base = _RANGING_SCORE
            note = f"ranging market, {action.value}"

        # Break of structure against trade direction is a negative signal
        bos_penalty = 0.0
        if structure.break_of_structure:
            # BoS in uptrend (support break) hurts BUY; in downtrend hurts SELL
            if (trend == TrendDirection.UPTREND and action == SignalAction.BUY) or (
                trend == TrendDirection.DOWNTREND and action == SignalAction.SELL
            ):
                bos_penalty = _STRUCTURE_BOS_PENALTY
                note += "; BoS against trade"

        strength_bonus = 0.0
        if structure.trend_strength >= _STRUCTURE_STRENGTH_MIN:
            strength_bonus = _STRUCTURE_STRENGTH_BONUS
            note += "; strong trend"

        score = float(max(0.0, min(1.0, base - bos_penalty + strength_bonus)))
        return score, note

    def _score_levels(
        self,
        candidate: SignalCandidate,
        levels: list[KeyLevel] | None,
    ) -> tuple[float, str]:
        """Score proximity to a key price level."""
        if not levels:
            return _LEVEL_NO_DATA_SCORE, "no level data"

        entry_mid = (candidate.entry_zone_low + candidate.entry_zone_high) / 2.0
        threshold = _LEVEL_PROXIMITY_PCT * entry_mid

        closest: KeyLevel | None = None
        best_dist = float("inf")
        for lvl in levels:
            dist = abs(lvl.price - entry_mid)
            if dist <= threshold and dist < best_dist:
                best_dist = dist
                closest = lvl

        if closest is None:
            return _LEVEL_NO_NEARBY_SCORE, "no level near entry"

        base = closest.strength
        bonus = _LEVEL_TYPE_BONUS if closest.level_type in _LEVEL_BONUS_TYPES else 0.0
        score = float(max(0.0, min(1.0, base + bonus)))
        return score, f"near {closest.level_type} @ {closest.price:.4f}"

    def _score_volume(
        self,
        candidate: SignalCandidate,  # noqa: ARG002  (kept for API symmetry)
        volume: VolumeContext | None,
    ) -> tuple[float, str]:
        """Return the pre-computed volume score directly."""
        if volume is None:
            return _VOLUME_NO_DATA_SCORE, "no volume data"
        return float(volume.score), f"volume score {volume.score:.2f}"

    def _score_regime(
        self, regime: RegimeAnalysis | None
    ) -> tuple[float, str]:
        """Return the pre-computed regime vol_score directly."""
        if regime is None:
            return _REGIME_NO_DATA_SCORE, "no regime data"
        return float(regime.vol_score), f"vol_score {regime.vol_score:.2f}"

    def _score_session(
        self, candidate: SignalCandidate
    ) -> tuple[float, str]:
        """Return the session quality_score."""
        if candidate.session is None:
            return _SESSION_NO_DATA_SCORE, "no session data"
        return float(candidate.session.quality_score), (
            f"session {candidate.session.session_type.value} "
            f"q={candidate.session.quality_score:.2f}"
        )

    def _score_data_quality(
        self, candidate: SignalCandidate
    ) -> tuple[float, str, list[str], list[str]]:
        """
        Score data quality.

        Returns (score, note, blocked_reasons, warning_tags).
        """
        if candidate.quality is None:
            return _DATA_QUALITY_NO_DATA_SCORE, "no quality report", [], []

        status = candidate.quality.status

        if status == DataQualityStatus.CLEAN:
            return 1.0, "data quality: CLEAN", [], []

        if status == DataQualityStatus.WARNING:
            return (
                _DATA_QUALITY_WARNING_SCORE,
                "data quality: WARNING",
                [],
                ["data_quality_warning"],
            )

        # BLOCKED
        return (
            0.0,
            "data quality: BLOCKED",
            ["data quality blocked"],
            [],
        )

    def _score_risk(
        self, risk: RiskAssessment | None
    ) -> tuple[float, str, list[str]]:
        """
        Score risk assessment.

        Returns (score, note, blocked_reasons).
        """
        if risk is None:
            return _RISK_NO_DATA_SCORE, "no risk data", []

        if not risk.passed:
            return 0.0, "risk check failed", list(risk.blocked_reasons)

        return float(risk.score), f"risk score {risk.score:.2f}", []
