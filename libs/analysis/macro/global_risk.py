"""
Global Risk Engine.

Assesses macro-level systemic risk from VIX, BTC/SPY drawdowns, geopolitical
events, and banking crises. Returns an immutable risk assessment that can
be used to adjust signal confidence and reduce exposure during crisis periods.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── VIX thresholds ─────────────────────────────────────────────────────────────
_VIX_HIGH = 30
_VIX_VERY_HIGH = 40
_VIX_EXTREME = 50

_VIX_SCORE_HIGH = 20
_VIX_SCORE_VERY_HIGH = 35
_VIX_SCORE_EXTREME = 50

# ── BTC drawdown thresholds (%) ────────────────────────────────────────────────
_BTC_DRAWDOWN_MODERATE = 5.0
_BTC_DRAWDOWN_HIGH = 10.0
_BTC_DRAWDOWN_SEVERE = 20.0

_BTC_SCORE_MODERATE = 15
_BTC_SCORE_HIGH = 25
_BTC_SCORE_SEVERE = 40

# ── SPY drawdown thresholds (%) ────────────────────────────────────────────────
_SPY_DRAWDOWN_MODERATE = 3.0
_SPY_DRAWDOWN_HIGH = 5.0

_SPY_SCORE_MODERATE = 15
_SPY_SCORE_HIGH = 25

# ── Event scores ───────────────────────────────────────────────────────────────
_GEOPOLITICAL_SCORE = 20
_BANKING_CRISIS_SCORE = 30

# ── Risk level thresholds ──────────────────────────────────────────────────────
_ELEVATED_THRESHOLD = 25
_HIGH_THRESHOLD = 50
_EXTREME_THRESHOLD = 75

# ── Confidence adjustments ─────────────────────────────────────────────────────
_CONF_ADJ_LOW = 0.0
_CONF_ADJ_ELEVATED = -0.05
_CONF_ADJ_HIGH = -0.10
_CONF_ADJ_EXTREME = -0.15

# ── Reduce exposure threshold ──────────────────────────────────────────────────
_REDUCE_EXPOSURE_SCORE = 50


@dataclass(frozen=True)
class GlobalRiskAssessment:
    """Immutable snapshot of current global macro risk conditions."""

    risk_score: float
    """Additive risk score, 0–100+ (clamped at 100). 0 = calm, 100 = crisis."""

    risk_level: str
    """Qualitative label: 'low', 'elevated', 'high', or 'extreme'."""

    confidence_adjustment: float
    """Signed confidence delta. Range: -0.15 to 0.0."""

    should_reduce_exposure: bool
    """True when systemic risk warrants reducing position sizes."""

    factors: list[str]
    """Human-readable list of active risk factors that contributed to the score."""


def _score_vix(vix: float) -> tuple[int, list[str]]:
    """Return (score, factors) for VIX level."""
    if vix > _VIX_EXTREME:
        return _VIX_SCORE_EXTREME, [f"VIX={vix:.1f} (extreme panic)"]
    if vix > _VIX_VERY_HIGH:
        return _VIX_SCORE_VERY_HIGH, [f"VIX={vix:.1f} (very high fear)"]
    if vix > _VIX_HIGH:
        return _VIX_SCORE_HIGH, [f"VIX={vix:.1f} (elevated fear)"]
    return 0, []


def _score_btc_drawdown(drawdown_pct: float) -> tuple[int, list[str]]:
    """Return (score, factors) for BTC drawdown magnitude."""
    if drawdown_pct > _BTC_DRAWDOWN_SEVERE:
        return _BTC_SCORE_SEVERE, [f"BTC drawdown {drawdown_pct:.1f}% (severe crash)"]
    if drawdown_pct > _BTC_DRAWDOWN_HIGH:
        return _BTC_SCORE_HIGH, [f"BTC drawdown {drawdown_pct:.1f}% (high)"]
    if drawdown_pct > _BTC_DRAWDOWN_MODERATE:
        return _BTC_SCORE_MODERATE, [f"BTC drawdown {drawdown_pct:.1f}% (moderate)"]
    return 0, []


def _score_spy_drawdown(drawdown_pct: float) -> tuple[int, list[str]]:
    """Return (score, factors) for SPY drawdown magnitude."""
    if drawdown_pct > _SPY_DRAWDOWN_HIGH:
        return _SPY_SCORE_HIGH, [f"SPY drawdown {drawdown_pct:.1f}% (high)"]
    if drawdown_pct > _SPY_DRAWDOWN_MODERATE:
        return _SPY_SCORE_MODERATE, [f"SPY drawdown {drawdown_pct:.1f}% (moderate)"]
    return 0, []


def _classify_risk(score: float) -> tuple[str, float]:
    """Return (risk_level, confidence_adjustment) for a given risk score."""
    if score >= _EXTREME_THRESHOLD:
        return "extreme", _CONF_ADJ_EXTREME
    if score >= _HIGH_THRESHOLD:
        return "high", _CONF_ADJ_HIGH
    if score >= _ELEVATED_THRESHOLD:
        return "elevated", _CONF_ADJ_ELEVATED
    return "low", _CONF_ADJ_LOW


class GlobalRiskEngine:
    """
    Assesses global macro risk from multiple systemic indicators.

    Usage
    -----
    engine = GlobalRiskEngine()
    assessment = engine.assess(vix=35.0, btc_drawdown_pct=8.0)
    if assessment.should_reduce_exposure:
        ...
    """

    def assess(
        self,
        vix: float = 20.0,
        btc_drawdown_pct: float = 0.0,
        spy_drawdown_pct: float = 0.0,
        is_geopolitical_risk: bool = False,
        is_banking_crisis: bool = False,
    ) -> GlobalRiskAssessment:
        """
        Score systemic risk from multiple additive components.

        Score components (additive):
        - VIX > 30: +20, VIX > 40: +35, VIX > 50: +50
        - BTC drawdown > 5%: +15, > 10%: +25, > 20%: +40
        - SPY drawdown > 3%: +15, > 5%: +25
        - Geopolitical risk flag: +20
        - Banking crisis flag: +30

        Levels: <25 low, <50 elevated, <75 high, >=75 extreme
        Confidence: low→0, elevated→-5%, high→-10%, extreme→-15%
        Reduce exposure if score >= 50.

        Parameters
        ----------
        vix:
            CBOE Volatility Index value (default 20.0 = calm market).
        btc_drawdown_pct:
            BTC peak-to-current drawdown in percent (positive = down).
        spy_drawdown_pct:
            SPY peak-to-current drawdown in percent (positive = down).
        is_geopolitical_risk:
            True when a significant geopolitical event is active.
        is_banking_crisis:
            True when a banking/systemic financial crisis is underway.

        Returns
        -------
        GlobalRiskAssessment
            Immutable assessment with risk_score clamped to [0, 100].
        """
        total_score = 0
        factors: list[str] = []

        vix_score, vix_factors = _score_vix(vix)
        total_score += vix_score
        factors.extend(vix_factors)

        btc_score, btc_factors = _score_btc_drawdown(btc_drawdown_pct)
        total_score += btc_score
        factors.extend(btc_factors)

        spy_score, spy_factors = _score_spy_drawdown(spy_drawdown_pct)
        total_score += spy_score
        factors.extend(spy_factors)

        if is_geopolitical_risk:
            total_score += _GEOPOLITICAL_SCORE
            factors.append("geopolitical risk event active")

        if is_banking_crisis:
            total_score += _BANKING_CRISIS_SCORE
            factors.append("banking/systemic crisis active")

        clamped_score = float(min(100, total_score))
        risk_level, confidence_adjustment = _classify_risk(clamped_score)
        should_reduce_exposure = clamped_score >= _REDUCE_EXPOSURE_SCORE

        return GlobalRiskAssessment(
            risk_score=clamped_score,
            risk_level=risk_level,
            confidence_adjustment=confidence_adjustment,
            should_reduce_exposure=should_reduce_exposure,
            factors=factors,
        )
