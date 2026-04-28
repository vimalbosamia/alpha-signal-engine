"""
SignalOutcomeTracker — checks if BUY/SELL signals hit TP1 or SL.

Every poll cycle (30-second interval), ALL pending signals are checked:
  - BUY  WIN  → price >= take_profit_1
  - BUY  LOSS → price <= stop_loss
  - SELL WIN  → price <= take_profit_1
  - SELL LOSS → price >= stop_loss

A signal remains PENDING until TP1 or SL is actually hit, or until
MAX_SIGNAL_AGE_HOURS elapses — at which point it is marked EXPIRED.

There is NO directional fallback; interim price between SL and TP1
does not resolve the signal.

Results are stored in `signal_outcomes` table.
Per-strategy win rates are maintained in `strategy_performance` table.

The tracker NEVER places or modifies orders. It is purely observational.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from libs.core.logging.logger import get_logger
from libs.core.models.domain import SignalAction, SignalOutput
from libs.data.storage.db import get_session_factory
from libs.data.storage.models import SignalOutcomeRecord
from libs.data.storage.repository import OutcomeRepository

log = get_logger(__name__)

# Outcome constants
OUTCOME_WIN = "WIN"
OUTCOME_LOSS = "LOSS"
OUTCOME_PENDING = "PENDING"
OUTCOME_EXPIRED = "EXPIRED"

# Muted strategies — populated at runtime by learning loop
_MUTED_STRATEGIES: set[str] = set()


def is_strategy_muted(strategy_name: str) -> bool:
    """Return True if this strategy has poor recent performance."""
    return strategy_name in _MUTED_STRATEGIES


async def load_muted_strategies_from_db() -> int:
    """Load muted strategies from DB into in-memory set. Returns count loaded.
    Call once at server startup before the pipeline runs."""
    try:
        async with get_session_factory()() as db_session:
            repo = OutcomeRepository(db_session)
            stats = await repo.get_strategy_stats()
        muted = {s["strategy"] for s in stats if s["muted"]}
        _MUTED_STRATEGIES.clear()
        _MUTED_STRATEGIES.update(muted)
        log.info("muted_strategies_loaded", count=len(muted), strategies=list(muted))
        return len(muted)
    except Exception as exc:
        log.warning("muted_strategies_load_failed", error=str(exc))
        return 0


@dataclass
class PendingCheck:
    outcome_id: str
    signal_id: str
    symbol: str
    asset_class: str
    strategy_name: str
    action: SignalAction
    entry_price: float
    stop_loss: float
    take_profit_1: float
    check_at: datetime      # UTC time when signal was scheduled (reference only; does not gate resolution)
    created_at: datetime
    ml_features: list[float] | None = None   # feature vector for ML training


class SignalOutcomeTracker:
    """
    Monitors BUY/SELL signals and records whether they were correct.

    Usage:
        tracker = SignalOutcomeTracker(crypto_provider, stock_provider)
        tracker.enqueue(signal_output)
        asyncio.create_task(tracker.run_loop())
    """

    MAX_SIGNAL_AGE_HOURS: int = 4   # expire unresolved signals after 4h

    def __init__(
        self,
        crypto_provider=None,
        stock_provider=None,
        check_after_minutes: int = 30,
        poll_interval: int = 30,
    ) -> None:
        self._crypto = crypto_provider
        self._stock = stock_provider
        self._check_after = check_after_minutes
        self._poll_interval = poll_interval
        self._queue: list[PendingCheck] = []
        self._lock = asyncio.Lock()

    def enqueue(self, signal: SignalOutput) -> None:
        """Register a BUY/SELL signal for outcome checking."""
        if signal.action == SignalAction.NO_TRADE:
            return

        # Extract ML features at signal time (before we know outcome)
        try:
            from libs.ml.features import extract as extract_features
            ml_features = extract_features(signal)
        except Exception:
            ml_features = None

        entry = (signal.entry_zone_low + signal.entry_zone_high) / 2.0
        now = datetime.now(timezone.utc)
        check_at = now + timedelta(minutes=self._check_after)

        pending = PendingCheck(
            outcome_id=str(uuid4()),
            signal_id=str(signal.signal_id),
            symbol=signal.symbol,
            asset_class=signal.asset_class.value,
            strategy_name=signal.strategy_name,
            action=signal.action,
            entry_price=entry,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            check_at=check_at,
            created_at=now,
            ml_features=ml_features,
        )
        self._queue.append(pending)

        # Immediately write PENDING record to DB
        asyncio.ensure_future(self._save_pending(pending))

        log.info(
            "outcome_enqueued",
            symbol=signal.symbol,
            strategy=signal.strategy_name,
            action=signal.action.value,
            check_at=check_at.isoformat(),
        )

    async def run_loop(self, stop_event: asyncio.Event | None = None) -> None:
        """Poll pending signals. Runs until stop_event is set."""
        log.info("outcome_tracker_started", check_after_minutes=self._check_after)
        while True:
            if stop_event and stop_event.is_set():
                break
            await self._process_due()
            await asyncio.sleep(self._poll_interval)

    async def _process_due(self) -> None:
        now = datetime.now(timezone.utc)
        resolved: list[str] = []

        # Check ALL pending signals every cycle — resolution is price-driven, not time-driven
        tasks = [self._resolve(p) for p in self._queue]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for p, resolved_flag in zip(self._queue, results):
            if resolved_flag is True:
                resolved.append(p.outcome_id)

        self._queue = [p for p in self._queue if p.outcome_id not in resolved]

        # Expire signals older than MAX_SIGNAL_AGE_HOURS where TP1/SL was never hit
        cutoff = now - timedelta(hours=self.MAX_SIGNAL_AGE_HOURS)
        stale = [p for p in self._queue if p.created_at < cutoff]
        for p in stale:
            await self._write_outcome(p, price=0.0, outcome=OUTCOME_EXPIRED, correct=False)
        self._queue = [p for p in self._queue if p.outcome_id not in {s.outcome_id for s in stale}]

    async def _resolve(self, pending: PendingCheck) -> bool:
        """Check if TP1 or SL has been hit.

        Returns True if the signal is now resolved (WIN or LOSS written to DB).
        Returns False if neither level has been hit — signal stays pending.
        Never raises; all exceptions are caught and logged.
        """
        provider = self._crypto if pending.asset_class == "crypto" else self._stock
        if provider is None:
            return False

        try:
            price = await provider.get_latest_price(pending.symbol)
        except Exception as exc:
            log.debug("outcome_price_fetch_failed", symbol=pending.symbol, error=str(exc))
            return False

        if price is None or price <= 0:
            return False

        outcome: str | None = None
        correct: bool = False

        if pending.action == SignalAction.BUY:
            if price >= pending.take_profit_1:
                outcome, correct = OUTCOME_WIN, True
            elif price <= pending.stop_loss:
                outcome, correct = OUTCOME_LOSS, False
        else:  # SELL
            if price <= pending.take_profit_1:
                outcome, correct = OUTCOME_WIN, True
            elif price >= pending.stop_loss:
                outcome, correct = OUTCOME_LOSS, False

        if outcome is None:
            # Price is between SL and TP1 — signal remains pending
            return False

        await self._write_outcome(pending, price, outcome, correct)

        log.info(
            "signal_outcome_resolved",
            symbol=pending.symbol,
            strategy=pending.strategy_name,
            action=pending.action.value,
            outcome=outcome,
            entry=pending.entry_price,
            check_price=price,
            correct=correct,
        )
        return True

    async def _write_outcome(
        self,
        pending: PendingCheck,
        price: float,
        outcome: str,
        correct: bool,
    ) -> None:
        import orjson

        ml_features_json = (
            orjson.dumps(pending.ml_features).decode()
            if pending.ml_features else None
        )

        record = SignalOutcomeRecord(
            id=pending.outcome_id,
            signal_id=pending.signal_id,
            symbol=pending.symbol,
            strategy_name=pending.strategy_name,
            action=pending.action.value,
            entry_price=pending.entry_price,
            check_price=price,
            stop_loss=pending.stop_loss,
            take_profit_1=pending.take_profit_1,
            outcome=outcome,
            direction_correct=correct,
            check_after_minutes=self._check_after,
            ml_features=ml_features_json,
            created_at=pending.created_at,
            checked_at=datetime.now(timezone.utc) if outcome != OUTCOME_PENDING else None,
        )
        try:
            async with get_session_factory()() as db_session:
                repo = OutcomeRepository(db_session)
                await repo.save_outcome(record)

                # Update strategy performance stats
                if outcome in (OUTCOME_WIN, OUTCOME_LOSS):
                    await repo.upsert_strategy_performance(
                        pending.strategy_name, won=correct
                    )
                    # Refresh muted set
                    stats = await repo.get_strategy_stats()
                    muted = {s["strategy"] for s in stats if s["muted"]}
                    _MUTED_STRATEGIES.clear()
                    _MUTED_STRATEGIES.update(muted)
                    if muted:
                        log.info("strategies_muted", muted=list(muted))

                    # Notify ML classifier of new outcome for potential retrain
                    if pending.ml_features:
                        try:
                            from libs.ml.signal_classifier import get_classifier
                            get_classifier().record_outcome(pending.ml_features, win=correct)
                        except Exception as ml_exc:
                            log.debug("ml_record_outcome_failed", error=str(ml_exc))
        except Exception as exc:
            log.warning("outcome_db_save_failed", error=str(exc))

    async def _save_pending(self, pending: PendingCheck) -> None:
        import orjson

        ml_features_json = (
            orjson.dumps(pending.ml_features).decode()
            if pending.ml_features else None
        )
        record = SignalOutcomeRecord(
            id=pending.outcome_id,
            signal_id=pending.signal_id,
            symbol=pending.symbol,
            strategy_name=pending.strategy_name,
            action=pending.action.value,
            entry_price=pending.entry_price,
            check_price=0.0,
            stop_loss=pending.stop_loss,
            take_profit_1=pending.take_profit_1,
            outcome=OUTCOME_PENDING,
            direction_correct=False,
            check_after_minutes=self._check_after,
            ml_features=ml_features_json,
            created_at=pending.created_at,
            checked_at=None,
        )
        try:
            async with get_session_factory()() as db_session:
                repo = OutcomeRepository(db_session)
                await repo.save_outcome(record)
        except Exception as exc:
            log.warning("pending_outcome_save_failed", error=str(exc))
