"""
PortfolioGuard — enforces exposure limits before a signal is accepted.

Rules (all configurable via GuardConfig):
  0. daily loss circuit breaker — halt all signals when daily realized loss >= threshold
  1. max_concurrent_signals    — total open BUY+SELL signals across all symbols
  2. max_per_symbol            — no more than N open signals for the same symbol
  3. max_same_direction        — limits how many BUY-only or SELL-only at once
  4. max_per_asset_class       — separate limits for crypto vs. stock exposure
  5. max_portfolio_heat_pct    — halt new signals when total % at risk >= threshold

Design rules:
  - Stateful but thread-safe (uses a simple lock).
  - Never raises — is_allowed() returns (False, reason) on any error.
  - register() and release() are idempotent.
  - Immutable GuardConfig; all guard state lives in PortfolioGuard instance.
  - No async I/O — purely in-memory.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from libs.core.config.settings import get_settings
from libs.core.logging.logger import get_logger
from libs.core.models.domain import AssetClass, SignalAction, SignalOutput

log = get_logger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GuardConfig:
    """Exposure limits enforced by PortfolioGuard."""
    max_concurrent_signals: int = 50     # total open signals (all symbols)
    max_per_symbol: int = 3             # open signals per symbol
    max_same_direction: int = 25        # max BUY-only or SELL-only at once
    max_per_asset_class: int = 30       # open signals per asset class
    max_age_hours: float = 24.0         # auto-expire positions older than this
    max_portfolio_heat_pct: float = 10.0  # halt new signals if total risk >= this %
    max_daily_loss_pct: float = 3.0     # halt if total realized loss >= this % of account in one day


# ── Internal position record ────────────────────────────────────────────────────

@dataclass
class _ActiveSignal:
    signal_id: UUID
    symbol: str
    asset_class: str        # AssetClass.value
    action: str             # SignalAction.value ("BUY" | "SELL")
    accepted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    risk_pct: float = 1.0   # per-signal risk as % of account


# ── Guard check result ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GuardDecision:
    """Return value of PortfolioGuard.is_allowed()."""
    allowed: bool
    reason: str = ""        # non-empty only when allowed=False

    @property
    def blocked(self) -> bool:
        return not self.allowed


# ── PortfolioGuard ─────────────────────────────────────────────────────────────

class PortfolioGuard:
    """
    Stateful exposure gate.  Call is_allowed() before emitting a signal.
    If allowed, call register() to track it.  Call release() when the
    signal resolves (TP/SL hit or manual close).

    Thread-safe: uses a threading.RLock.
    """

    def __init__(self, config: GuardConfig | None = None) -> None:
        self._config = config or GuardConfig()
        self._lock = threading.RLock()
        self._signals: dict[UUID, _ActiveSignal] = {}
        self._daily_loss_pct: float = 0.0   # accumulated realized loss today (% of account)
        self._loss_day: str = ""             # YYYY-MM-DD of current loss day

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_allowed(self, signal: SignalOutput) -> GuardDecision:
        """
        Check if a new signal is within exposure limits.

        Returns GuardDecision(allowed=True) when all rules pass.
        Returns GuardDecision(allowed=False, reason=...) on first violation.
        Never raises.
        """
        if signal.action == SignalAction.NO_TRADE:
            return GuardDecision(allowed=False, reason="NO_TRADE signal skipped")

        try:
            with self._lock:
                self._expire_old()
                return self._evaluate(signal)
        except Exception as exc:
            log.warning("portfolio_guard_error", error=str(exc))
            return GuardDecision(allowed=False, reason="Guard internal error — fail-closed")

    def register(self, signal: SignalOutput) -> None:
        """
        Mark a signal as active.  Must be called after is_allowed() returns True.
        Idempotent: registering the same signal_id twice is a no-op.
        """
        if signal.action == SignalAction.NO_TRADE:
            return
        try:
            risk_pct = get_settings().risk.max_risk_per_signal_pct
        except Exception:
            risk_pct = 1.0
        with self._lock:
            if signal.signal_id in self._signals:
                return
            self._signals[signal.signal_id] = _ActiveSignal(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                asset_class=signal.asset_class.value,
                action=signal.action.value,
                risk_pct=risk_pct,
            )
            log.debug(
                "portfolio_guard_registered",
                symbol=signal.symbol,
                action=signal.action.value,
                total_open=len(self._signals),
            )

    def release(self, signal_id: UUID) -> bool:
        """
        Remove a signal from the active set.
        Returns True if the signal was found and removed, False otherwise.
        Idempotent.
        """
        with self._lock:
            removed = self._signals.pop(signal_id, None)
            if removed:
                log.debug(
                    "portfolio_guard_released",
                    symbol=removed.symbol,
                    total_open=len(self._signals),
                )
                return True
            return False

    def release_by_symbol(self, symbol: str) -> int:
        """Remove all active signals for a given symbol. Returns count removed."""
        with self._lock:
            to_remove = [
                sig_id for sig_id, s in self._signals.items()
                if s.symbol == symbol
            ]
            for sig_id in to_remove:
                self._signals.pop(sig_id)
            if to_remove:
                log.debug(
                    "portfolio_guard_released_symbol",
                    symbol=symbol,
                    count=len(to_remove),
                )
            return len(to_remove)

    def record_loss(self, loss_pct: float) -> None:
        """Record a realized loss (as % of account). Resets automatically on new calendar day."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if self._loss_day != today:
                self._daily_loss_pct = 0.0
                self._loss_day = today
            self._daily_loss_pct += loss_pct
            log.info("daily_loss_recorded", loss_pct=loss_pct, total_today=self._daily_loss_pct)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._signals)

    def active_signals(self) -> list[_ActiveSignal]:
        with self._lock:
            return list(self._signals.values())

    def exposure_summary(self) -> dict:
        """Return per-asset-class, per-direction counts and risk metrics for observability."""
        with self._lock:
            sigs = list(self._signals.values())
            daily_loss = self._daily_loss_pct

        summary: dict = {
            "total": len(sigs),
            "buy": sum(1 for s in sigs if s.action == "BUY"),
            "sell": sum(1 for s in sigs if s.action == "SELL"),
            "crypto": sum(1 for s in sigs if s.asset_class == AssetClass.CRYPTO.value),
            "stock": sum(1 for s in sigs if s.asset_class == AssetClass.STOCK.value),
            "portfolio_heat_pct": sum(s.risk_pct for s in sigs),
            "daily_loss_pct": daily_loss,
        }
        return summary

    # ── Internal ───────────────────────────────────────────────────────────────

    def _expire_old(self) -> None:
        """Remove signals older than max_age_hours. Called inside lock."""
        now = datetime.now(timezone.utc)
        max_age_secs = self._config.max_age_hours * 3600
        expired = [
            sig_id
            for sig_id, s in self._signals.items()
            if (now - s.accepted_at).total_seconds() > max_age_secs
        ]
        for sig_id in expired:
            self._signals.pop(sig_id)
        if expired:
            log.debug("portfolio_guard_expired", count=len(expired))

    def _evaluate(self, signal: SignalOutput) -> GuardDecision:
        """Run all guard rules. Called inside lock with expired entries removed."""
        cfg = self._config
        sigs = list(self._signals.values())

        # Rule 0: daily loss circuit breaker
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._loss_day == today and self._daily_loss_pct >= cfg.max_daily_loss_pct:
            return GuardDecision(
                allowed=False,
                reason=f"GUARD_DAILY_LOSS: {self._daily_loss_pct:.1f}%/{cfg.max_daily_loss_pct}% daily loss limit hit",
            )

        # Rule 1: total concurrent cap
        if len(sigs) >= cfg.max_concurrent_signals:
            return GuardDecision(
                allowed=False,
                reason=f"GUARD_MAX_SIGNALS: {len(sigs)}/{cfg.max_concurrent_signals} open signals",
            )

        # Rule 2: per-symbol cap
        same_symbol = sum(1 for s in sigs if s.symbol == signal.symbol)
        if same_symbol >= cfg.max_per_symbol:
            return GuardDecision(
                allowed=False,
                reason=f"GUARD_MAX_PER_SYMBOL: {same_symbol}/{cfg.max_per_symbol} for {signal.symbol}",
            )

        # Rule 3: directional concentration
        same_dir = sum(1 for s in sigs if s.action == signal.action.value)
        if same_dir >= cfg.max_same_direction:
            return GuardDecision(
                allowed=False,
                reason=(
                    f"GUARD_DIRECTION: {same_dir}/{cfg.max_same_direction} "
                    f"{signal.action.value} signals open"
                ),
            )

        # Rule 4: per-asset-class cap
        same_ac = sum(1 for s in sigs if s.asset_class == signal.asset_class.value)
        if same_ac >= cfg.max_per_asset_class:
            return GuardDecision(
                allowed=False,
                reason=(
                    f"GUARD_ASSET_CLASS: {same_ac}/{cfg.max_per_asset_class} "
                    f"{signal.asset_class.value} signals open"
                ),
            )

        # Rule 5: portfolio heat cap
        try:
            cfg_risk_pct = get_settings().risk.max_risk_per_signal_pct
        except Exception:
            cfg_risk_pct = 1.0
        total_heat = sum(s.risk_pct for s in sigs)
        if total_heat + cfg_risk_pct >= cfg.max_portfolio_heat_pct:
            return GuardDecision(
                allowed=False,
                reason=f"GUARD_PORTFOLIO_HEAT: {total_heat:.1f}%/{cfg.max_portfolio_heat_pct}% at risk",
            )

        return GuardDecision(allowed=True)


# ── Singleton ──────────────────────────────────────────────────────────────────

_guard: PortfolioGuard | None = None
_guard_lock = threading.Lock()


def get_portfolio_guard(config: GuardConfig | None = None) -> PortfolioGuard:
    """Return the process-wide PortfolioGuard singleton."""
    global _guard
    with _guard_lock:
        if _guard is None:
            _guard = PortfolioGuard(config)
        return _guard


def reset_portfolio_guard(config: GuardConfig | None = None) -> PortfolioGuard:
    """Replace the singleton (useful for tests and reconfiguration)."""
    global _guard
    with _guard_lock:
        _guard = PortfolioGuard(config)
        return _guard
