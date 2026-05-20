"""
Risk Throttle Engine (Adaptive Cooldown).

Reduces signal aggressiveness after consecutive losses, high loss frequency
in the past hour, or volatility spikes.  Returns an immutable ThrottleState
describing the current risk level and any penalties applied.

Design rules:
  - All results are immutable (frozen dataclass)
  - No magic numbers — all thresholds are named class constants
  - reset() is the only way to clear accumulated loss state
  - Hour counter is checked and reset inside get_state() automatically
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThrottleState:
    """Immutable snapshot of current risk throttle level."""

    is_throttled: bool
    throttle_level: str        # "normal" | "cautious" | "defensive" | "halted"
    max_signals_per_cycle: int
    confidence_penalty: float
    reason: str


# ── Engine ────────────────────────────────────────────────────────────────────

class RiskThrottleEngine:
    """
    Reduces aggressiveness after consecutive losses, volatility spikes, or
    news chaos.

    Throttle levels:
      normal    — 0-2 consecutive losses; no penalty
      cautious  — 3-4 losses; confidence -5%, max 5 signals/cycle
      defensive — 5-7 losses (or 5+ losses in the last hour); -10%, max 2/cycle
      halted    — 8+ losses; blocks all new signals
    """

    # ── Loss thresholds ───────────────────────────────────────────────────────
    _CAUTIOUS_MIN_LOSSES: int = 3
    _DEFENSIVE_MIN_LOSSES: int = 5
    _HALTED_MIN_LOSSES: int = 8
    _HOURLY_DEFENSIVE_THRESHOLD: int = 5

    # ── Level parameters: (max_signals, confidence_penalty) ──────────────────
    _LEVEL_PARAMS: dict[str, tuple[int, float]] = {
        "normal":    (10,   0.0),
        "cautious":  (5,   -0.05),
        "defensive": (2,   -0.10),
        "halted":    (0,   -1.0),   # confidence_penalty irrelevant when halted
    }

    # ── Hour window ───────────────────────────────────────────────────────────
    _HOUR_WINDOW: timedelta = timedelta(hours=1)

    def __init__(self) -> None:
        self._consecutive_losses: int = 0
        self._losses_last_hour: int = 0
        self._last_reset: datetime = datetime.now(timezone.utc)

    # ── Public API ────────────────────────────────────────────────────────────

    def record_loss(self) -> None:
        """Increment both the consecutive and hourly loss counters."""
        self._consecutive_losses += 1
        self._losses_last_hour += 1

    def record_win(self) -> None:
        """Reset the consecutive loss counter (hourly counter unaffected)."""
        self._consecutive_losses = 0

    def get_state(self) -> ThrottleState:
        """
        Return the current ThrottleState based on accumulated loss data.

        Automatically resets the hourly counter if more than 60 minutes have
        elapsed since the last reset.
        """
        # Auto-reset the hourly counter if the window has passed
        now = datetime.now(timezone.utc)
        if now - self._last_reset >= self._HOUR_WINDOW:
            self._losses_last_hour = 0
            self._last_reset = now

        level = self._compute_level()
        max_signals, penalty = self._LEVEL_PARAMS[level]
        is_throttled = level != "normal"

        reason = self._build_reason(level)

        return ThrottleState(
            is_throttled=is_throttled,
            throttle_level=level,
            max_signals_per_cycle=max_signals,
            confidence_penalty=penalty,
            reason=reason,
        )

    def reset(self) -> None:
        """Fully reset all counters."""
        self._consecutive_losses = 0
        self._losses_last_hour = 0
        self._last_reset = datetime.now(timezone.utc)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_level(self) -> str:
        """Map current loss state to a throttle level string."""
        losses = self._consecutive_losses

        if losses >= self._HALTED_MIN_LOSSES:
            return "halted"

        if losses >= self._DEFENSIVE_MIN_LOSSES:
            return "defensive"

        # Also defensive if hourly losses are high
        if self._losses_last_hour >= self._HOURLY_DEFENSIVE_THRESHOLD:
            return "defensive"

        if losses >= self._CAUTIOUS_MIN_LOSSES:
            return "cautious"

        return "normal"

    def _build_reason(self, level: str) -> str:
        """Return a human-readable reason string for the current level."""
        if level == "halted":
            return (
                f"Trading halted: {self._consecutive_losses} consecutive losses"
            )
        if level == "defensive":
            hourly = self._losses_last_hour
            consecutive = self._consecutive_losses
            if hourly >= self._HOURLY_DEFENSIVE_THRESHOLD and consecutive < self._DEFENSIVE_MIN_LOSSES:
                return f"Defensive throttle: {hourly} losses in last hour"
            return f"Defensive throttle: {consecutive} consecutive losses"
        if level == "cautious":
            return f"Cautious throttle: {self._consecutive_losses} consecutive losses"
        return "Normal: no throttle active"
