"""
Signal Output Emitter.

Converts a scored SignalCandidate + ConfluenceBreakdown into the final
SignalOutput that is written to the audit log and surfaced to operators.

This agent NEVER executes trades — all output is purely informational.

Design rules:
  - No side effects — pure functions that build immutable output.
  - All thresholds come from settings; no magic numbers.
  - Never raises on valid inputs.
"""
from __future__ import annotations

from libs.core.config.settings import get_settings
from libs.core.models.domain import (
    ConfluenceBreakdown,
    DataQualityStatus,
    SessionType,
    SignalAction,
    SignalCandidate,
    SignalOutput,
)

# ── Constant ───────────────────────────────────────────────────────────────────

_NO_RR: float = 0.0
_EPSILON: float = 1e-9


class SignalEmitter:
    """
    Assembles a final SignalOutput from a scored candidate.

    Parameters
    ----------
    agent_mode:
        The running agent mode string (e.g. "paper", "live").  Passed through
        to SignalOutput for traceability.
    data_provider:
        The data provider name (e.g. "alpaca", "binance").  Passed through
        to SignalOutput for traceability.
    """

    def __init__(self, agent_mode: str = "", data_provider: str = "") -> None:
        self._agent_mode = agent_mode
        self._data_provider = data_provider

    # ── Public API ─────────────────────────────────────────────────────────────

    def emit(
        self,
        candidate: SignalCandidate,
        breakdown: ConfluenceBreakdown,
    ) -> SignalOutput:
        """
        Build and return a SignalOutput.  Never raises on valid inputs.
        """
        action = self._determine_action(candidate, breakdown)
        rr = self._calc_rr(candidate) if action != SignalAction.NO_TRADE else _NO_RR
        explanation = self._build_explanation(candidate, breakdown, action)
        patterns = self._detected_pattern_names(candidate)

        session_type = (
            candidate.session.session_type
            if candidate.session is not None
            else SessionType.CLOSED
        )

        dq_status = (
            candidate.quality.status
            if candidate.quality is not None
            else DataQualityStatus.WARNING
        )

        return SignalOutput(
            symbol=candidate.symbol,
            asset_class=candidate.asset_class,
            strategy_name=candidate.strategy_name,
            timeframe=candidate.timeframe,
            action=action,
            confidence=breakdown.weighted_total,
            higher_tf_bias=candidate.higher_tf_bias,
            entry_zone_low=candidate.entry_zone_low,
            entry_zone_high=candidate.entry_zone_high,
            stop_loss=candidate.stop_loss,
            stop_limit_price=self._calc_stop_limit(candidate, action),
            take_profit_1=candidate.take_profit_1,
            take_profit_2=candidate.take_profit_2,
            estimated_risk_reward=rr,
            market_regime=candidate.regime,
            session_status=session_type,
            data_quality_status=dq_status,
            confluence=breakdown,
            patterns_detected=patterns,
            explanation=explanation,
            warnings=list(breakdown.warning_tags),
            blocked_reasons=list(breakdown.blocked_reasons),
            agent_mode=self._agent_mode,
            data_provider=self._data_provider,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _determine_action(
        self,
        candidate: SignalCandidate,
        breakdown: ConfluenceBreakdown,
    ) -> SignalAction:
        """Return final action: NO_TRADE if blocked or below min score."""
        if breakdown.blocked_reasons:
            return SignalAction.NO_TRADE

        min_score = get_settings().signal.min_confluence_score
        if breakdown.weighted_total < min_score:
            return SignalAction.NO_TRADE

        return candidate.proposed_action

    def _calc_rr(self, candidate: SignalCandidate) -> float:
        """Compute reward-to-risk ratio from candidate zones."""
        entry = (candidate.entry_zone_low + candidate.entry_zone_high) / 2.0
        action = candidate.proposed_action
        tp1 = candidate.take_profit_1
        sl = candidate.stop_loss

        if action == SignalAction.BUY:
            return (tp1 - entry) / max(entry - sl, _EPSILON)
        if action == SignalAction.SELL:
            return (entry - tp1) / max(sl - entry, _EPSILON)
        return _NO_RR

    def _build_explanation(
        self,
        candidate: SignalCandidate,
        breakdown: ConfluenceBreakdown,
        action: SignalAction,
    ) -> str:
        """One-sentence human-readable summary of the signal decision."""
        patterns = self._detected_pattern_names(candidate)
        pattern_str = ", ".join(patterns) if patterns else "none"
        rr = self._calc_rr(candidate) if action != SignalAction.NO_TRADE else _NO_RR
        return (
            f"{action.value} on {candidate.symbol}: "
            f"score={breakdown.weighted_total:.2f}, "
            f"pattern={pattern_str}, "
            f"regime={candidate.regime.value}, "
            f"R:R={rr:.1f}"
        )

    def _calc_stop_limit(
        self,
        candidate: SignalCandidate,
        action: SignalAction,
    ) -> float | None:
        """
        Calculate the limit price for a stop-limit order.

        For a BUY position (long):
          - You want to EXIT (SELL) when stop is hit.
          - Stop triggers at stop_loss; limit = stop_loss * (1 - 0.2%)
            so you'll fill at stop or up to 0.2% below — avoids slippage gap.

        For a SELL position (short):
          - You want to EXIT (BUY) when stop is hit.
          - Stop triggers at stop_loss; limit = stop_loss * (1 + 0.2%)
            so you'll fill at stop or up to 0.2% above.

        Returns None for NO_TRADE signals.
        """
        if action == SignalAction.NO_TRADE:
            return None
        sl = candidate.stop_loss
        if action == SignalAction.BUY:
            return round(sl * (1 - 0.002), 8)   # 0.2% below stop
        return round(sl * (1 + 0.002), 8)        # 0.2% above stop

    def _detected_pattern_names(self, candidate: SignalCandidate) -> list[str]:
        """Return names of all detected patterns."""
        return [p.pattern_name for p in candidate.pattern_results if p.detected]
