"""
Futures-specific risk engine.

Calculates liquidation price, liquidation buffer, leverage risk classification,
and rejection logic for futures positions using Binance-style isolated margin.

This module is purely analytical — no trade execution occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Constants ─────────────────────────────────────────────────────────────────

MAINTENANCE_MARGIN_RATE = 0.004  # 0.4% — Binance default for most contracts

_MAX_LEVERAGE = 50.0             # leverage > 50 → reject
_MIN_BUFFER_PCT = 2.0            # buffer < 2% → reject
_MAX_LOSS_PCT = 80.0             # max loss > 80% of margin → reject

# Risk thresholds (checked in priority order: EXTREME → HIGH → MODERATE → LOW)
_EXTREME_LEVERAGE = 50.0
_HIGH_LEVERAGE = 20.0
_MODERATE_LEVERAGE = 10.0

_EXTREME_BUFFER = 2.0
_HIGH_BUFFER = 5.0
_MODERATE_BUFFER = 10.0


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FuturesRiskAssessment:
    """Result of FuturesRiskEngine.assess() for a single futures position."""

    leverage: float
    margin_type: str                    # "isolated" or "cross"
    estimated_liquidation_price: float
    liquidation_buffer_percent: float   # distance from entry to liq as %
    liquidation_risk: str               # "LOW" | "MODERATE" | "HIGH" | "EXTREME"
    max_loss_before_stop: float         # % loss at stop price (amplified by leverage)
    should_reject: bool
    rejection_reason: str               # empty string when should_reject is False


# ── Engine ────────────────────────────────────────────────────────────────────

class FuturesRiskEngine:
    """
    Stateless futures risk assessor.

    Instantiate once; call assess() per position.  All thresholds are
    class-level constants so tests can inspect and override them without
    monkey-patching the module.
    """

    MAX_LEVERAGE: float = _MAX_LEVERAGE
    MIN_BUFFER_PCT: float = _MIN_BUFFER_PCT
    MAX_LOSS_PCT: float = _MAX_LOSS_PCT

    # ── Public API ─────────────────────────────────────────────────────────────

    def assess(
        self,
        entry_price: float,
        leverage: float,
        margin_type: str,
        direction: str,
        stop_loss: float,
        funding_rate: float = 0.0,  # reserved for future use; not yet in formulas
    ) -> FuturesRiskAssessment:
        """
        Evaluate futures-specific risk for a single position.

        Parameters
        ----------
        entry_price:
            Intended entry price for the position.
        leverage:
            Notional leverage multiplier (e.g. 10 for 10x).
        margin_type:
            "isolated" or "cross" — stored on the result, not used in
            the liquidation formula (formula assumes isolated margin).
        direction:
            "long" or "short" (case-insensitive).
        stop_loss:
            Desired stop-loss price.
        funding_rate:
            Periodic funding rate (currently informational only).

        Returns
        -------
        FuturesRiskAssessment
            Immutable assessment dataclass.
        """
        direction_lower = direction.lower()

        liq_price = self._liquidation_price(entry_price, leverage, direction_lower)
        buffer_pct = self._buffer_percent(entry_price, liq_price)
        max_loss = self._max_loss(entry_price, stop_loss, leverage)
        risk_label = self._classify_risk(leverage, buffer_pct)
        should_reject, reason = self._rejection_check(
            entry_price=entry_price,
            leverage=leverage,
            liq_price=liq_price,
            buffer_pct=buffer_pct,
            stop_loss=stop_loss,
            direction=direction_lower,
            max_loss=max_loss,
        )

        return FuturesRiskAssessment(
            leverage=float(leverage),
            margin_type=margin_type,
            estimated_liquidation_price=liq_price,
            liquidation_buffer_percent=buffer_pct,
            liquidation_risk=risk_label,
            max_loss_before_stop=max_loss,
            should_reject=should_reject,
            rejection_reason=reason,
        )

    # ── Internal calculations ──────────────────────────────────────────────────

    @staticmethod
    def _liquidation_price(entry: float, leverage: float, direction: str) -> float:
        """
        Binance-style isolated margin liquidation price.

        LONG:  liq = entry * (1 - 1/leverage + MMR)
        SHORT: liq = entry * (1 + 1/leverage - MMR)
        """
        if direction == "long":
            return entry * (1.0 - 1.0 / leverage + MAINTENANCE_MARGIN_RATE)
        return entry * (1.0 + 1.0 / leverage - MAINTENANCE_MARGIN_RATE)

    @staticmethod
    def _buffer_percent(entry: float, liq_price: float) -> float:
        """Percentage distance from entry to liquidation price."""
        if entry == 0.0:
            return 0.0
        return abs(entry - liq_price) / entry * 100.0

    @staticmethod
    def _max_loss(entry: float, stop_loss: float, leverage: float) -> float:
        """
        Maximum % loss of margin if the stop is hit.

        stop_distance_pct * leverage
        """
        if entry == 0.0:
            return 0.0
        stop_distance_pct = abs(entry - stop_loss) / entry * 100.0
        return stop_distance_pct * leverage

    @staticmethod
    def _classify_risk(leverage: float, buffer_pct: float) -> str:
        """
        Classify liquidation risk.

        EXTREME: leverage >= 50 OR buffer < 2%
        HIGH:    leverage >= 20 OR buffer < 5%
        MODERATE:leverage >= 10 OR buffer < 10%
        LOW:     otherwise
        """
        if leverage >= _EXTREME_LEVERAGE or buffer_pct < _EXTREME_BUFFER:
            return "EXTREME"
        if leverage >= _HIGH_LEVERAGE or buffer_pct < _HIGH_BUFFER:
            return "HIGH"
        if leverage >= _MODERATE_LEVERAGE or buffer_pct < _MODERATE_BUFFER:
            return "MODERATE"
        return "LOW"

    def _rejection_check(
        self,
        entry_price: float,
        leverage: float,
        liq_price: float,
        buffer_pct: float,
        stop_loss: float,
        direction: str,
        max_loss: float,
    ) -> tuple[bool, str]:
        """
        Return (should_reject, reason).

        Rejection conditions (evaluated in priority order):
        1. Stop loss at or beyond liquidation price
        2. Buffer < 2%
        3. Leverage > 50
        4. Max loss > 80% of margin
        """
        # 1. Stop loss at or beyond liquidation
        if direction == "long" and stop_loss <= liq_price:
            return True, (
                f"Stop loss {stop_loss} is at or beyond the liquidation price "
                f"{liq_price:.4f} for a LONG position."
            )
        if direction == "short" and stop_loss >= liq_price:
            return True, (
                f"Stop loss {stop_loss} is at or beyond the liquidation price "
                f"{liq_price:.4f} for a SHORT position."
            )

        # 2. Buffer too small
        if buffer_pct < self.MIN_BUFFER_PCT:
            return True, (
                f"Liquidation buffer {buffer_pct:.2f}% is below the minimum "
                f"{self.MIN_BUFFER_PCT}%."
            )

        # 3. Leverage too high
        if leverage > self.MAX_LEVERAGE:
            return True, (
                f"Leverage {leverage}x exceeds the maximum allowed "
                f"{self.MAX_LEVERAGE}x."
            )

        # 4. Max loss too large
        if max_loss > self.MAX_LOSS_PCT:
            return True, (
                f"Max loss before stop {max_loss:.1f}% exceeds the maximum "
                f"allowed {self.MAX_LOSS_PCT}% of margin."
            )

        return False, ""
