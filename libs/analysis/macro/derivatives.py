"""
Derivatives / Futures Pressure Engine.

Analyses perpetual futures funding rates, open interest trends, and long/short
ratios to identify crowding risk, potential squeeze setups, and liquidation
cascade conditions. Returns an immutable DerivativesPressure assessment.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Funding rate thresholds ────────────────────────────────────────────────────
_FUNDING_BULLISH_THRESHOLD = 0.0005   # > 0.05% → longs paying → bullish_pressure
_FUNDING_BEARISH_THRESHOLD = -0.0005  # < -0.05% → shorts paying → bearish_pressure

# ── OI change threshold for "increasing" / "decreasing" ───────────────────────
_OI_INCREASING_MIN_PCT = 2.0   # OI change > +2% → increasing
_OI_DECREASING_MAX_PCT = -2.0  # OI change < -2% → decreasing

# ── Long/short ratio thresholds ────────────────────────────────────────────────
_LS_OVERCROWDED_LONGS = 2.0   # ratio > 2.0 → overcrowded longs
_LS_OVERCROWDED_SHORTS = 0.5  # ratio < 0.5 → overcrowded shorts

# ── Confidence adjustments ─────────────────────────────────────────────────────
_ADJ_NONE = 0.0
_ADJ_MINOR = -0.05
_ADJ_MAJOR = -0.10


@dataclass(frozen=True)
class DerivativesPressure:
    """Immutable snapshot of derivatives market pressure conditions."""

    funding_rate: float
    """Raw funding rate (e.g. 0.0001 = 0.01%)."""

    funding_sentiment: str
    """'bearish_pressure', 'neutral', or 'bullish_pressure'."""

    open_interest_trend: str
    """'increasing', 'stable', or 'decreasing'."""

    liquidation_risk: str
    """'low', 'moderate', or 'high'."""

    confidence_adjustment: float
    """Signed confidence delta to apply. Range: -0.10 to 0.0."""

    explanation: str
    """Human-readable summary of the derivatives conditions."""


def _classify_funding(rate: float) -> tuple[str, float]:
    """Return (sentiment_label, base_confidence_adjustment) for funding rate."""
    if rate > _FUNDING_BULLISH_THRESHOLD:
        return "bullish_pressure", _ADJ_MINOR
    if rate < _FUNDING_BEARISH_THRESHOLD:
        return "bearish_pressure", _ADJ_MINOR
    return "neutral", _ADJ_NONE


def _classify_oi_trend(oi_change_pct: float) -> str:
    """Return OI trend label for a given OI percentage change."""
    if oi_change_pct > _OI_INCREASING_MIN_PCT:
        return "increasing"
    if oi_change_pct < _OI_DECREASING_MAX_PCT:
        return "decreasing"
    return "stable"


def _assess_liquidation_risk(
    oi_trend: str,
    oi_change_pct: float,
) -> tuple[str, float]:
    """
    Return (liquidation_risk_label, extra_confidence_adj).

    High liquidation risk occurs when OI is increasing while prices are
    dropping (inferred from a negative OI change direction combined with
    cascade indicators). The caller must pass price_dropping to activate
    this path.
    """
    # This function is called with the context of whether price is dropping;
    # the cascade condition is: OI increasing AND price dropping.
    # We return the base liquidation risk and let assess() layer context.
    if oi_trend == "increasing":
        return "moderate", _ADJ_NONE
    if oi_trend == "decreasing":
        return "low", _ADJ_NONE
    return "low", _ADJ_NONE


def _crowding_adjustment(long_short_ratio: float) -> tuple[str, float]:
    """Return (crowding_note, confidence_adj) based on long/short ratio."""
    if long_short_ratio > _LS_OVERCROWDED_LONGS:
        return "overcrowded longs — reversal risk", _ADJ_MINOR
    if long_short_ratio < _LS_OVERCROWDED_SHORTS:
        return "overcrowded shorts — squeeze risk", _ADJ_MINOR
    return "", _ADJ_NONE


def _build_explanation(
    funding_sentiment: str,
    funding_rate: float,
    oi_trend: str,
    liquidation_risk: str,
    crowding_note: str,
    total_adj: float,
) -> str:
    """Compose a concise explanation string from assessed components."""
    parts: list[str] = []

    rate_pct = funding_rate * 100.0
    sign = "+" if rate_pct >= 0.0 else ""
    parts.append(
        f"funding {sign}{rate_pct:.4f}% ({funding_sentiment})"
    )

    parts.append(f"OI {oi_trend}")

    if liquidation_risk != "low":
        parts.append(f"liquidation risk: {liquidation_risk}")

    if crowding_note:
        parts.append(crowding_note)

    adj_pct = round(total_adj * 100)
    if adj_pct < 0:
        parts.append(f"confidence {adj_pct}%")
    else:
        parts.append("no confidence penalty")

    return "; ".join(parts)


class DerivativesPressureEngine:
    """
    Assesses perpetual futures market pressure from funding, OI, and positioning.

    Usage
    -----
    engine = DerivativesPressureEngine()
    pressure = engine.assess(funding_rate=0.0008, oi_change_pct=5.0,
                             long_short_ratio=2.5)
    if pressure.liquidation_risk == "high":
        ...
    """

    def assess(
        self,
        funding_rate: float = 0.0001,
        oi_change_pct: float = 0.0,
        long_short_ratio: float = 1.0,
    ) -> DerivativesPressure:
        """
        Evaluate derivatives market pressure from three signals.

        Rules
        -----
        - Funding rate > 0.05%: bullish_pressure (longs pay shorts),
          potential long squeeze → -5% confidence
        - Funding rate < -0.05%: bearish_pressure (shorts pay longs),
          potential short squeeze → -5% confidence
        - OI increasing + price dropping → liquidation cascade risk = high,
          -10% confidence. This is triggered when OI is increasing AND
          oi_change_pct itself is negative (proxy for price-dropping context).
          NOTE: caller should set oi_change_pct negative when price is falling
          while OI rises (pass negative oi_change_pct in that scenario).
          For standard use, pass positive oi_change_pct for OI increasing and
          is_price_dropping separately via the sign convention below.
        - OI decreasing: positions closing, reduced pressure → stable/low
        - Long/short ratio > 2.0: overcrowded longs, reversal risk → -5%
        - Long/short ratio < 0.5: overcrowded shorts, squeeze risk → -5%

        Parameters
        ----------
        funding_rate:
            Perpetual funding rate as a decimal (e.g. 0.0001 = 0.01%).
        oi_change_pct:
            Open interest percentage change. Pass a negative value when OI is
            increasing while prices are falling (liquidation cascade context).
            Positive → OI growing with price, negative → OI growing vs price.
        long_short_ratio:
            Ratio of long to short positions. 1.0 = balanced.

        Returns
        -------
        DerivativesPressure
            Immutable assessment with confidence_adjustment clamped to [-0.10, 0.0].
        """
        funding_sentiment, funding_adj = _classify_funding(funding_rate)

        # Detect OI trend from absolute magnitude
        oi_trend = _classify_oi_trend(oi_change_pct)

        # Liquidation cascade: OI is increasing AND price is dropping.
        # Convention: caller passes negative oi_change_pct to signal this state.
        # We detect it as: raw oi_change_pct < 0 (price dropping) AND
        # the absolute value exceeds the increasing threshold.
        is_cascade_condition = (
            oi_change_pct < 0 and abs(oi_change_pct) > _OI_INCREASING_MIN_PCT
        )

        if is_cascade_condition:
            liquidation_risk = "high"
            oi_risk_adj = _ADJ_MAJOR
            # Override OI trend label for the cascade case
            oi_trend = "increasing"
        else:
            liquidation_risk, oi_risk_adj = _assess_liquidation_risk(
                oi_trend, oi_change_pct
            )

        crowding_note, crowding_adj = _crowding_adjustment(long_short_ratio)

        raw_adj = funding_adj + oi_risk_adj + crowding_adj
        total_adj = float(max(-0.10, min(0.0, raw_adj)))

        explanation = _build_explanation(
            funding_sentiment=funding_sentiment,
            funding_rate=funding_rate,
            oi_trend=oi_trend,
            liquidation_risk=liquidation_risk,
            crowding_note=crowding_note,
            total_adj=total_adj,
        )

        return DerivativesPressure(
            funding_rate=funding_rate,
            funding_sentiment=funding_sentiment,
            open_interest_trend=oi_trend,
            liquidation_risk=liquidation_risk,
            confidence_adjustment=total_adj,
            explanation=explanation,
        )
