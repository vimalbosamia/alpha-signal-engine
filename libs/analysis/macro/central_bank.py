"""
Central Bank / Interest Rate Engine.

Assesses the monetary policy environment and its impact on trading confidence.
Rules are based on Fed rate direction, absolute rate levels, DXY trend, and
10-year Treasury yield.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Rate thresholds ────────────────────────────────────────────────────────────
_HIGH_RATE_THRESHOLD = 5.0   # hold + above this → tightening
_LOW_RATE_THRESHOLD = 2.0    # hold + below this → easing
_HIGH_TREASURY_THRESHOLD = 5.0  # 10Y above this → risk_off

# ── Confidence adjustments ────────────────────────────────────────────────────
_ADJ_HIKING = -0.10
_ADJ_CUTTING = +0.05
_ADJ_HOLD_HIGH = -0.05
_ADJ_HOLD_LOW = +0.03
_ADJ_DXY_RISING = -0.05
_ADJ_DXY_FALLING = +0.03
_ADJ_HIGH_TREASURY = -0.05


@dataclass(frozen=True)
class MonetaryEnvironment:
    """Immutable snapshot of the assessed monetary policy environment."""

    regime: str
    """One of: tightening, neutral, easing."""

    risk_appetite: str
    """One of: risk_on, risk_off, neutral."""

    confidence_adjustment: float
    """Signed delta applied to signal confidence. Range: -0.10 to +0.05."""

    explanation: str
    """Human-readable summary of the assessment."""


def _assess_rate_direction(
    fed_rate: float, rate_direction: str
) -> tuple[str, str, float, str]:
    """Return (regime, risk_appetite, adjustment, note) based on rate direction."""
    if rate_direction == "hiking":
        return "tightening", "risk_off", _ADJ_HIKING, "Fed hiking — tightening cycle"
    if rate_direction == "cutting":
        return "easing", "risk_on", _ADJ_CUTTING, "Fed cutting — easing cycle"
    # Hold: classify by absolute rate level
    if fed_rate > _HIGH_RATE_THRESHOLD:
        return "tightening", "risk_off", _ADJ_HOLD_HIGH, f"Hold at high rate ({fed_rate}%) — tightening bias"
    if fed_rate < _LOW_RATE_THRESHOLD:
        return "easing", "risk_on", _ADJ_HOLD_LOW, f"Hold at low rate ({fed_rate}%) — easing bias"
    return "neutral", "neutral", 0.0, f"Hold at neutral rate ({fed_rate}%)"


def _assess_dxy(dxy_trend: str) -> tuple[float, str]:
    """Return (adjustment, note) based on DXY trend."""
    if dxy_trend == "rising":
        return _ADJ_DXY_RISING, "strong dollar (DXY rising) — risk_off for crypto"
    if dxy_trend == "falling":
        return _ADJ_DXY_FALLING, "weak dollar (DXY falling) — risk_on for crypto"
    return 0.0, ""


def _assess_treasury(treasury_10y: float) -> tuple[float, str]:
    """Return (adjustment, note) for 10Y Treasury yield."""
    if treasury_10y > _HIGH_TREASURY_THRESHOLD:
        return _ADJ_HIGH_TREASURY, f"10Y Treasury at {treasury_10y}% — risk_off"
    return 0.0, ""


class CentralBankEngine:
    """Evaluates monetary policy conditions and their effect on trade confidence."""

    def assess(
        self,
        fed_rate: float = 5.25,
        rate_direction: str = "hold",
        treasury_10y: float = 4.5,
        dxy_trend: str = "neutral",
    ) -> MonetaryEnvironment:
        """Assess the monetary environment and return a confidence adjustment.

        Args:
            fed_rate: Current Federal Funds rate in percent (e.g. 5.25).
            rate_direction: One of "hiking", "cutting", "hold".
            treasury_10y: Current 10-year Treasury yield in percent.
            dxy_trend: One of "rising", "falling", "neutral".

        Returns:
            MonetaryEnvironment with regime, risk_appetite, and signed
            confidence_adjustment clamped to [-0.10, +0.05].
        """
        regime, risk_appetite, rate_adj, rate_note = _assess_rate_direction(
            fed_rate, rate_direction
        )
        dxy_adj, dxy_note = _assess_dxy(dxy_trend)
        treasury_adj, treasury_note = _assess_treasury(treasury_10y)

        raw_adjustment = rate_adj + dxy_adj + treasury_adj
        clamped_adjustment = max(-0.10, min(0.05, raw_adjustment))

        notes = [n for n in [rate_note, dxy_note, treasury_note] if n]
        adj_pct = round(clamped_adjustment * 100)
        if adj_pct != 0:
            notes.append(f"confidence {adj_pct:+d}%")
        explanation = "; ".join(notes) if notes else "neutral monetary environment"

        return MonetaryEnvironment(
            regime=regime,
            risk_appetite=risk_appetite,
            confidence_adjustment=clamped_adjustment,
            explanation=explanation,
        )
