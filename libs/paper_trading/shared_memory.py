"""SharedLossMemory — central loss database shared across all bots.

Every bot reports losses here. Every bot queries before taking a trade.
If the shared memory shows repeated failures for a pattern, all bots avoid it.

Rules:
  - Record every loss with: symbol, strategy, regime, action, patterns, R:R
  - Track failure counts per (symbol+action), (strategy+action), (regime+action)
  - Block conditions: 3+ losses on same key in rolling window
  - All bots check should_avoid() before opening a trade
  - Wins reset counters for their key
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from libs.core.logging.logger import get_logger

log = get_logger(__name__)

LOSS_THRESHOLD = 20      # block after N losses on same key (lenient for paper learning)
ROLLING_WINDOW = 50      # only look at last N records
WIN_RESET_COUNT = 1      # N wins on same key resets block


@dataclass(frozen=True)
class LossRecord:
    """Single recorded loss from any bot."""
    bot_name: str
    symbol: str
    action: str           # "BUY" or "SELL"
    strategy: str
    regime: str
    patterns: list[str]
    risk_reward: float
    loss_amount: float
    timestamp: datetime


@dataclass(frozen=True)
class WinRecord:
    """Single recorded win from any bot."""
    bot_name: str
    symbol: str
    action: str
    strategy: str
    regime: str
    timestamp: datetime


class SharedLossMemory:
    """Central memory shared by all bots. Thread-safe via atomic operations."""

    def __init__(self) -> None:
        self._losses: list[LossRecord] = []
        self._wins: list[WinRecord] = []
        self._adaptation_log: list[dict] = []

    def record_loss(
        self,
        bot_name: str,
        symbol: str,
        action: str,
        strategy: str,
        regime: str,
        patterns: list[str],
        risk_reward: float,
        loss_amount: float,
    ) -> None:
        """Record a loss from any bot."""
        rec = LossRecord(
            bot_name=bot_name, symbol=symbol, action=action,
            strategy=strategy, regime=regime, patterns=patterns,
            risk_reward=risk_reward, loss_amount=loss_amount,
            timestamp=datetime.now(timezone.utc),
        )
        self._losses.append(rec)
        # Trim to rolling window
        if len(self._losses) > ROLLING_WINDOW * 2:
            self._losses = self._losses[-ROLLING_WINDOW:]

        log.info(
            "shared_memory_loss",
            bot=bot_name, symbol=symbol, action=action,
            strategy=strategy, regime=regime, total_losses=len(self._losses),
        )

    def record_win(
        self,
        bot_name: str,
        symbol: str,
        action: str,
        strategy: str,
        regime: str,
    ) -> None:
        """Record a win — can reset blocked patterns."""
        rec = WinRecord(
            bot_name=bot_name, symbol=symbol, action=action,
            strategy=strategy, regime=regime,
            timestamp=datetime.now(timezone.utc),
        )
        self._wins.append(rec)
        if len(self._wins) > ROLLING_WINDOW * 2:
            self._wins = self._wins[-ROLLING_WINDOW:]

    def should_avoid(
        self,
        symbol: str,
        action: str,
        strategy: str,
        regime: str,
    ) -> tuple[bool, str]:
        """Check if this trade should be avoided based on shared loss history.

        Returns (should_avoid: bool, reason: str).
        """
        recent = self._losses[-ROLLING_WINDOW:]
        recent_wins = self._wins[-ROLLING_WINDOW:]

        # Check symbol+action failures
        sym_action_losses = sum(
            1 for r in recent if r.symbol == symbol and r.action == action
        )
        sym_action_wins = sum(
            1 for r in recent_wins if r.symbol == symbol and r.action == action
        )
        if sym_action_losses >= LOSS_THRESHOLD and sym_action_wins < WIN_RESET_COUNT:
            reason = f"Shared memory: {sym_action_losses} losses on {action} {symbol}"
            self._log_avoid(reason)
            return True, reason

        # Check strategy+action failures
        strat_action_losses = sum(
            1 for r in recent if r.strategy == strategy and r.action == action
        )
        strat_action_wins = sum(
            1 for r in recent_wins if r.strategy == strategy and r.action == action
        )
        if strat_action_losses >= LOSS_THRESHOLD and strat_action_wins < WIN_RESET_COUNT:
            reason = f"Shared memory: {strat_action_losses} losses on {action} via {strategy}"
            self._log_avoid(reason)
            return True, reason

        # Check regime+action failures — only block specific regime+strategy combos,
        # not entire regimes (too coarse for paper learning)
        regime_strat_losses = sum(
            1 for r in recent
            if r.regime == regime and r.action == action and r.strategy == strategy
        )
        regime_strat_wins = sum(
            1 for r in recent_wins
            if r.regime == regime and r.action == action and r.strategy == strategy
        )
        if regime_strat_losses >= LOSS_THRESHOLD and regime_strat_wins < WIN_RESET_COUNT:
            reason = f"Shared memory: {regime_strat_losses} losses on {action} {strategy} in {regime}"
            self._log_avoid(reason)
            return True, reason

        return False, ""

    def get_stats(self) -> dict:
        """Return memory stats for dashboard."""
        recent = self._losses[-ROLLING_WINDOW:]
        recent_wins = self._wins[-ROLLING_WINDOW:]

        # Count by key
        sym_counts: dict[str, int] = defaultdict(int)
        strat_counts: dict[str, int] = defaultdict(int)
        regime_counts: dict[str, int] = defaultdict(int)
        for r in recent:
            sym_counts[f"{r.action} {r.symbol}"] += 1
            strat_counts[f"{r.action} {r.strategy}"] += 1
            regime_counts[f"{r.action} {r.regime}"] += 1

        blocked = []
        for key, count in sym_counts.items():
            if count >= LOSS_THRESHOLD:
                blocked.append({"key": key, "type": "symbol", "losses": count})
        for key, count in strat_counts.items():
            if count >= LOSS_THRESHOLD:
                blocked.append({"key": key, "type": "strategy", "losses": count})
        for key, count in regime_counts.items():
            if count >= LOSS_THRESHOLD:
                blocked.append({"key": key, "type": "regime", "losses": count})

        return {
            "total_losses": len(recent),
            "total_wins": len(recent_wins),
            "blocked_patterns": blocked,
            "recent_adaptations": self._adaptation_log[-10:],
        }

    def _log_avoid(self, reason: str) -> None:
        self._adaptation_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        })
        log.info("shared_memory_avoid", reason=reason)


# ── Singleton ────────────────────────────────────────────────────────────────

_shared_memory: SharedLossMemory | None = None


def get_shared_memory() -> SharedLossMemory:
    global _shared_memory
    if _shared_memory is None:
        _shared_memory = SharedLossMemory()
    return _shared_memory
