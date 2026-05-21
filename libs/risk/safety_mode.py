"""
SafetyModeManager — system-wide emergency mode management.

Automatically transitions between NORMAL/SAFE/DEFENSIVE/PANIC/NO_TRADE
based on drawdown percentage, ATR volatility, and consecutive loss streaks.
Supports manual overrides and step-down recovery logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional

from libs.core.logging.logger import get_logger

logger = get_logger(__name__)


# ── Enums & Dataclasses ───────────────────────────────────────────────────────

class SafetyMode(IntEnum):
    """Ordered severity levels — higher value = more restrictive."""
    NORMAL = 0
    SAFE = 1
    DEFENSIVE = 2
    PANIC = 3
    NO_TRADE = 4


@dataclass(frozen=True)
class ModeTransition:
    """Immutable record of a mode change."""
    from_mode: SafetyMode
    to_mode: SafetyMode
    reason: str
    timestamp: str


# ── Manager ───────────────────────────────────────────────────────────────────

class SafetyModeManager:
    """
    Manages system-wide emergency modes.

    Drawdown, ATR volatility, and consecutive losses each trigger escalation.
    Recovery is gradual: one step down per update once drawdown normalises.
    Manual overrides (force_mode / clear_override) take precedence.
    """

    # Leverage map per mode
    _LEVERAGE: dict[SafetyMode, float] = {
        SafetyMode.NORMAL: 3.0,
        SafetyMode.SAFE: 2.0,
        SafetyMode.DEFENSIVE: 1.0,
        SafetyMode.PANIC: 0.0,
        SafetyMode.NO_TRADE: 0.0,
    }

    # Position-size multiplier per mode
    _SIZE_MULT: dict[SafetyMode, float] = {
        SafetyMode.NORMAL: 1.0,
        SafetyMode.SAFE: 0.50,
        SafetyMode.DEFENSIVE: 0.25,
        SafetyMode.PANIC: 0.0,
        SafetyMode.NO_TRADE: 0.0,
    }

    def __init__(
        self,
        drawdown_safe_pct: float = 5.0,
        drawdown_defensive_pct: float = 10.0,
        drawdown_panic_pct: float = 20.0,
        max_atr_pct: float = 15.0,
        max_consecutive_losses: int = 8,
        recovery_threshold: float = 5.0,
    ) -> None:
        self._safe_pct = drawdown_safe_pct
        self._defensive_pct = drawdown_defensive_pct
        self._panic_pct = drawdown_panic_pct
        self._max_atr_pct = max_atr_pct
        self._max_losses = max_consecutive_losses
        self._recovery_threshold = recovery_threshold

        self._computed_mode: SafetyMode = SafetyMode.NORMAL
        self._override: Optional[SafetyMode] = None
        self._history: list[ModeTransition] = []
        self._consecutive_losses: int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_mode(self) -> SafetyMode:
        """Returns the override if set, otherwise the computed mode."""
        return self._override if self._override is not None else self._computed_mode

    @property
    def mode_history(self) -> list[ModeTransition]:
        return list(self._history)

    # ── Core update logic ─────────────────────────────────────────────────────

    def update(
        self,
        drawdown_pct: float = 0.0,
        current_atr_pct: float = 0.0,
        consecutive_losses: Optional[int] = None,
    ) -> SafetyMode:
        """
        Evaluate current conditions and transition computed mode if necessary.

        Escalation rules (highest applicable wins):
        - Drawdown >= panic_pct                       → PANIC
        - Drawdown >= defensive_pct                   → DEFENSIVE
        - Drawdown >= safe_pct                        → SAFE
        - ATR > max_atr_pct * 2                       → PANIC
        - ATR > max_atr_pct                           → DEFENSIVE
        - consecutive_losses >= max_losses * 2        → PANIC
        - consecutive_losses >= max_losses            → DEFENSIVE

        Recovery (one step at a time):
        - When drawdown < safe_pct - recovery_threshold, step down one level.
        """
        if consecutive_losses is not None:
            self._consecutive_losses = consecutive_losses

        target = self._evaluate_target(drawdown_pct, current_atr_pct)
        new_mode = self._apply_recovery(drawdown_pct, target)
        self._record_transition_if_changed(new_mode, drawdown_pct, current_atr_pct)
        return self.current_mode

    def _evaluate_target(
        self, drawdown_pct: float, atr_pct: float
    ) -> SafetyMode:
        """Determine the target mode from current conditions."""
        target = SafetyMode.NORMAL

        # Drawdown escalation
        if drawdown_pct >= self._panic_pct:
            target = SafetyMode.PANIC
        elif drawdown_pct >= self._defensive_pct:
            target = max(target, SafetyMode.DEFENSIVE)
        elif drawdown_pct >= self._safe_pct:
            target = max(target, SafetyMode.SAFE)

        # ATR escalation
        if atr_pct > self._max_atr_pct * 2:
            target = max(target, SafetyMode.PANIC)
        elif atr_pct > self._max_atr_pct:
            target = max(target, SafetyMode.DEFENSIVE)

        # Consecutive-loss escalation
        losses = self._consecutive_losses
        if losses >= self._max_losses * 2:
            target = max(target, SafetyMode.PANIC)
        elif losses >= self._max_losses:
            target = max(target, SafetyMode.DEFENSIVE)

        return target

    def _apply_recovery(
        self, drawdown_pct: float, target: SafetyMode
    ) -> SafetyMode:
        """
        If conditions have normalised and target would be an upgrade (lower
        severity), allow only a single step down from the current computed mode.
        """
        recovery_level = self._safe_pct - self._recovery_threshold

        if drawdown_pct <= recovery_level and target < self._computed_mode:
            # Step down exactly one level — never jump multiple levels at once
            stepped = SafetyMode(max(0, self._computed_mode.value - 1))
            return stepped

        # Escalate freely; never de-escalate without recovery condition
        if target >= self._computed_mode:
            return target

        return self._computed_mode

    def _record_transition_if_changed(
        self, new_mode: SafetyMode, drawdown_pct: float, atr_pct: float
    ) -> None:
        if new_mode == self._computed_mode:
            return

        reason = self._build_reason(new_mode, drawdown_pct, atr_pct)
        transition = ModeTransition(
            from_mode=self._computed_mode,
            to_mode=new_mode,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._history.append(transition)
        logger.info(
            "safety_mode_transition",
            from_mode=self._computed_mode.name,
            to_mode=new_mode.name,
            reason=reason,
        )
        self._computed_mode = new_mode

    def _build_reason(
        self, new_mode: SafetyMode, drawdown_pct: float, atr_pct: float
    ) -> str:
        parts: list[str] = []
        if drawdown_pct >= self._panic_pct:
            parts.append(f"drawdown={drawdown_pct:.1f}% >= panic threshold {self._panic_pct}%")
        elif drawdown_pct >= self._defensive_pct:
            parts.append(f"drawdown={drawdown_pct:.1f}% >= defensive threshold {self._defensive_pct}%")
        elif drawdown_pct >= self._safe_pct:
            parts.append(f"drawdown={drawdown_pct:.1f}% >= safe threshold {self._safe_pct}%")
        if atr_pct > self._max_atr_pct * 2:
            parts.append(f"ATR={atr_pct:.1f}% > 2x max {self._max_atr_pct}%")
        elif atr_pct > self._max_atr_pct:
            parts.append(f"ATR={atr_pct:.1f}% > max {self._max_atr_pct}%")
        if self._consecutive_losses >= self._max_losses * 2:
            parts.append(f"consecutive_losses={self._consecutive_losses} >= 2x max {self._max_losses}")
        elif self._consecutive_losses >= self._max_losses:
            parts.append(f"consecutive_losses={self._consecutive_losses} >= max {self._max_losses}")
        if not parts:
            parts.append("recovery step-down")
        return "; ".join(parts)

    # ── Trading controls ──────────────────────────────────────────────────────

    def is_trading_allowed(self) -> bool:
        """True if current mode is below PANIC."""
        return self.current_mode < SafetyMode.PANIC

    def max_leverage(self) -> float:
        """Allowed leverage multiplier for current mode."""
        return self._LEVERAGE[self.current_mode]

    def position_size_multiplier(self) -> float:
        """Position size scaling factor for current mode."""
        return self._SIZE_MULT[self.current_mode]

    # ── Manual override ───────────────────────────────────────────────────────

    def force_mode(self, mode: SafetyMode) -> None:
        """Override the computed mode. Persists until clear_override() is called."""
        logger.warning("safety_mode_forced", mode=mode.name)
        self._override = mode

    def clear_override(self) -> None:
        """Remove any manual override; revert to computed mode."""
        logger.info("safety_mode_override_cleared", computed=self._computed_mode.name)
        self._override = None

    # ── Loss / win tracking ───────────────────────────────────────────────────

    def record_loss(self) -> None:
        """Increment consecutive-loss counter."""
        self._consecutive_losses += 1
        logger.debug("safety_mode_loss_recorded", consecutive_losses=self._consecutive_losses)

    def record_win(self) -> None:
        """Reset consecutive-loss counter on a win."""
        self._consecutive_losses = 0
        logger.debug("safety_mode_win_recorded")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a dashboard-ready summary of current safety state."""
        return {
            "current_mode": self.current_mode.name,
            "computed_mode": self._computed_mode.name,
            "override_active": self._override is not None,
            "override_mode": self._override.name if self._override is not None else None,
            "consecutive_losses": self._consecutive_losses,
            "trading_allowed": self.is_trading_allowed(),
            "max_leverage": self.max_leverage(),
            "position_size_multiplier": self.position_size_multiplier(),
            "total_transitions": len(self._history),
            "last_transition": (
                {
                    "from": self._history[-1].from_mode.name,
                    "to": self._history[-1].to_mode.name,
                    "reason": self._history[-1].reason,
                    "timestamp": self._history[-1].timestamp,
                }
                if self._history
                else None
            ),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_default_manager: Optional[SafetyModeManager] = None


def get_safety_manager() -> SafetyModeManager:
    """Return (or lazily create) the module-level singleton SafetyModeManager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SafetyModeManager()
    return _default_manager
