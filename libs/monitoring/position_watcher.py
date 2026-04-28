"""
PositionWatcher — monitors open signals and alerts when price hits TP or SL.

This module is purely observational. It NEVER places or cancels orders.
It only watches prices and emits alert notifications.

Lifecycle:
  1. Call track(signal) when a BUY/SELL signal is emitted.
  2. run_loop() polls prices every `poll_interval` seconds.
  3. When TP1/TP2 or SL is hit, an alert is printed + EventBus event published.
  4. Tracked position is marked closed and removed from active set.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID

from libs.core.logging.logger import get_logger
from libs.core.models.domain import SignalAction, SignalOutput
from libs.core.events.bus import EventBus

log = get_logger(__name__)

# ── Event topics ─────────────────────────────────────────────────────────────
TOPIC_TP_HIT = "position.take_profit_hit"
TOPIC_SL_HIT = "position.stop_loss_hit"
TOPIC_TP2_HIT = "position.take_profit2_hit"


class ExitReason(str, Enum):
    TAKE_PROFIT_1 = "TP1"
    TAKE_PROFIT_2 = "TP2"
    STOP_LOSS = "SL"


@dataclass
class TrackedPosition:
    signal_id: UUID
    symbol: str
    asset_class: str
    action: SignalAction           # BUY or SELL
    entry_low: float
    entry_high: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None
    strategy_name: str
    confidence: float
    data_provider: str = ""        # "binance" | "binance_futures" | "alpaca"
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tp1_hit: bool = False
    closed: bool = False

    @property
    def entry_mid(self) -> float:
        return (self.entry_low + self.entry_high) / 2

    def check_exit(self, price: float) -> ExitReason | None:
        """Return exit reason if price has crossed a level, else None."""
        if self.action == SignalAction.BUY:
            if not self.tp1_hit and price >= self.take_profit_1:
                return ExitReason.TAKE_PROFIT_1
            if self.tp1_hit and self.take_profit_2 and price >= self.take_profit_2:
                return ExitReason.TAKE_PROFIT_2
            if price <= self.stop_loss:
                return ExitReason.STOP_LOSS
        elif self.action == SignalAction.SELL:
            if not self.tp1_hit and price <= self.take_profit_1:
                return ExitReason.TAKE_PROFIT_1
            if self.tp1_hit and self.take_profit_2 and price <= self.take_profit_2:
                return ExitReason.TAKE_PROFIT_2
            if price >= self.stop_loss:
                return ExitReason.STOP_LOSS
        return None


class PositionWatcher:
    """
    Watches open signals and alerts when price hits TP or SL.

    Usage:
        watcher = PositionWatcher(crypto_provider, stock_provider, event_bus)
        watcher.track(signal_output)
        asyncio.create_task(watcher.run_loop())
    """

    # A position older than this is auto-expired (market moved on)
    MAX_POSITION_AGE_HOURS: int = 24

    def __init__(
        self,
        crypto_provider=None,
        stock_provider=None,
        futures_provider=None,
        event_bus: EventBus | None = None,
        poll_interval: int = 30,
    ) -> None:
        self._crypto = crypto_provider
        self._stock = stock_provider
        self._futures = futures_provider
        self._bus = event_bus or EventBus()
        self._poll_interval = poll_interval
        self._positions: dict[UUID, TrackedPosition] = {}
        self._lock = asyncio.Lock()

    def _has_open_position(self, symbol: str, action: SignalAction) -> bool:
        """True if an active (non-closed) position for this symbol+action exists."""
        return any(
            p.symbol == symbol and p.action == action and not p.closed
            for p in self._positions.values()
        )

    def _expire_old_positions(self) -> None:
        """Mark positions older than MAX_POSITION_AGE_HOURS as closed."""
        now = datetime.now(timezone.utc)
        for pos in self._positions.values():
            if pos.closed:
                continue
            age_hours = (now - pos.opened_at).total_seconds() / 3600
            if age_hours > self.MAX_POSITION_AGE_HOURS:
                pos.closed = True
                log.info(
                    "position_expired",
                    symbol=pos.symbol,
                    age_hours=round(age_hours, 1),
                )

    def track(self, signal: SignalOutput) -> None:
        """Register a new signal for price monitoring.

        Deduplication: if an active position for the same symbol+action
        already exists, the new signal is ignored to avoid flooding the
        tracker every scanner cycle.
        """
        if signal.action == SignalAction.NO_TRADE:
            return
        # Deduplicate — one active position per symbol+side at a time
        if self._has_open_position(signal.symbol, signal.action):
            return
        pos = TrackedPosition(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            asset_class=signal.asset_class.value,
            action=signal.action,
            entry_low=signal.entry_zone_low,
            entry_high=signal.entry_zone_high,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            strategy_name=signal.strategy_name,
            confidence=signal.confidence,
            data_provider=signal.data_provider,
        )
        self._positions[signal.signal_id] = pos
        log.info(
            "position_tracked",
            symbol=signal.symbol,
            action=signal.action.value,
            entry=f"{signal.entry_zone_low:.4f}–{signal.entry_zone_high:.4f}",
            sl=signal.stop_loss,
            tp1=signal.take_profit_1,
            tp2=signal.take_profit_2,
        )

    def untrack(self, signal_id: UUID) -> None:
        """Manually remove a position from tracking."""
        self._positions.pop(signal_id, None)

    @property
    def active_count(self) -> int:
        return sum(1 for p in self._positions.values() if not p.closed)

    def active_positions(self) -> list[TrackedPosition]:
        return [p for p in self._positions.values() if not p.closed]

    async def run_loop(self, stop_event: asyncio.Event | None = None) -> None:
        """Poll prices for all open positions. Runs until stop_event is set."""
        log.info("position_watcher_started", poll_interval=self._poll_interval)
        while True:
            self._expire_old_positions()
            if stop_event and stop_event.is_set():
                break
            await self._check_all()
            await asyncio.sleep(self._poll_interval)

    async def _check_all(self) -> None:
        active = [p for p in self._positions.values() if not p.closed]
        if not active:
            return

        # Route each position to its original provider
        tasks = []
        for pos in active:
            if pos.asset_class == "stock":
                if self._stock:
                    tasks.append(self._check_position(pos, self._stock))
            elif pos.data_provider == "binance_futures":
                provider = self._futures or self._crypto
                if provider:
                    tasks.append(self._check_position(pos, provider))
            else:
                if self._crypto:
                    tasks.append(self._check_position(pos, self._crypto))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_position(self, pos: TrackedPosition, provider) -> None:
        try:
            price = await provider.get_latest_price(pos.symbol)
            if price is None:
                return
        except Exception as exc:
            log.debug("price_fetch_failed", symbol=pos.symbol, error=str(exc))
            return

        reason = pos.check_exit(price)
        if reason is None:
            return

        if reason == ExitReason.TAKE_PROFIT_1:
            pos.tp1_hit = True
            if pos.take_profit_2:
                # Partial exit — keep watching for TP2
                self._alert(pos, price, reason, closed=False)
            else:
                pos.closed = True
                self._alert(pos, price, reason, closed=True)
            await self._bus.publish(TOPIC_TP_HIT, self._payload(pos, price, reason))

        elif reason == ExitReason.TAKE_PROFIT_2:
            pos.closed = True
            self._alert(pos, price, reason, closed=True)
            await self._bus.publish(TOPIC_TP2_HIT, self._payload(pos, price, reason))

        elif reason == ExitReason.STOP_LOSS:
            pos.closed = True
            self._alert(pos, price, reason, closed=True)
            await self._bus.publish(TOPIC_SL_HIT, self._payload(pos, price, reason))

    def _alert(
        self,
        pos: TrackedPosition,
        price: float,
        reason: ExitReason,
        closed: bool,
    ) -> None:
        icon = {"TP1": "✅", "TP2": "🎯", "SL": "🛑"}[reason.value]
        status = "CLOSED" if closed else "PARTIAL EXIT (watching TP2)"
        duration = datetime.now(timezone.utc) - pos.opened_at
        hours, rem = divmod(int(duration.total_seconds()), 3600)
        mins = rem // 60

        msg = (
            f"\n{icon} POSITION ALERT — {reason.value} HIT\n"
            f"   Symbol   : {pos.symbol} ({pos.asset_class.upper()})\n"
            f"   Action   : {pos.action.value}\n"
            f"   Entry    : {pos.entry_mid:.4f}\n"
            f"   Current  : {price:.4f}\n"
            f"   Target   : {pos.take_profit_1:.4f}"
            + (f" / {pos.take_profit_2:.4f}" if pos.take_profit_2 else "") + "\n"
            f"   Stop     : {pos.stop_loss:.4f}\n"
            f"   Strategy : {pos.strategy_name}\n"
            f"   Open for : {hours}h {mins}m\n"
            f"   Status   : {status}\n"
        )
        print(msg, flush=True)
        log.info(
            "position_exit_alert",
            symbol=pos.symbol,
            reason=reason.value,
            price=price,
            entry=pos.entry_mid,
            closed=closed,
        )

    @staticmethod
    def _payload(pos: TrackedPosition, price: float, reason: ExitReason) -> dict:
        return {
            "signal_id": str(pos.signal_id),
            "symbol": pos.symbol,
            "asset_class": pos.asset_class,
            "action": pos.action.value,
            "exit_reason": reason.value,
            "exit_price": price,
            "entry_price": pos.entry_mid,
            "strategy": pos.strategy_name,
        }
