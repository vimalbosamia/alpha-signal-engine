"""
Unit tests for SignalOutcomeTracker — TP1/SL price-hit detection logic.

Tests verify:
- WIN is only recorded when price actually hits TP1
- LOSS is only recorded when price actually hits SL
- No directional fallback (price between SL and TP1 stays PENDING)
- Signals older than MAX_SIGNAL_AGE_HOURS are EXPIRED
- Missing provider returns False (stays pending)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from libs.core.models.domain import SignalAction
from libs.monitoring.outcome_tracker import (
    OUTCOME_EXPIRED,
    OUTCOME_LOSS,
    OUTCOME_WIN,
    PendingCheck,
    SignalOutcomeTracker,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _outcome_from_call(mock_call):
    """Extract (outcome, correct) from either positional or keyword call."""
    args = mock_call.call_args.args
    kwargs = mock_call.call_args.kwargs
    outcome = args[2] if len(args) > 2 else kwargs["outcome"]
    correct = args[3] if len(args) > 3 else kwargs["correct"]
    return outcome, correct


def make_pending(
    action: SignalAction = SignalAction.BUY,
    entry_price: float = 100.0,
    stop_loss: float = 95.0,
    take_profit_1: float = 110.0,
    asset_class: str = "crypto",
    created_at: datetime | None = None,
) -> PendingCheck:
    now = datetime.now(timezone.utc)
    return PendingCheck(
        outcome_id=str(uuid4()),
        signal_id=str(uuid4()),
        symbol="BTC/USDT",
        asset_class=asset_class,
        strategy_name="test_strategy",
        action=action,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        check_at=now + timedelta(minutes=30),
        created_at=created_at or now,
        ml_features=None,
    )


def make_tracker(price: float | None = None) -> tuple[SignalOutcomeTracker, AsyncMock]:
    """Return a tracker with a mock crypto provider and patched _write_outcome."""
    provider = MagicMock()
    provider.get_latest_price = AsyncMock(return_value=price)
    tracker = SignalOutcomeTracker(crypto_provider=provider, stock_provider=None)
    tracker._write_outcome = AsyncMock()
    return tracker, provider


# ── BUY signal tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_buy_tp1_hit_resolves_as_win():
    """BUY signal where price >= tp1 must resolve as WIN."""
    tracker, provider = make_tracker(price=110.0)  # exactly at TP1
    pending = make_pending(action=SignalAction.BUY, take_profit_1=110.0, stop_loss=95.0)

    result = await tracker._resolve(pending)

    assert result is True
    tracker._write_outcome.assert_awaited_once()
    outcome, correct = _outcome_from_call(tracker._write_outcome)
    assert outcome == OUTCOME_WIN
    assert correct is True


@pytest.mark.asyncio
async def test_buy_tp1_exceeded_resolves_as_win():
    """BUY signal where price > tp1 (gapped through) also resolves as WIN."""
    tracker, provider = make_tracker(price=115.0)
    pending = make_pending(action=SignalAction.BUY, take_profit_1=110.0, stop_loss=95.0)

    result = await tracker._resolve(pending)

    assert result is True
    outcome, correct = _outcome_from_call(tracker._write_outcome)
    assert outcome == OUTCOME_WIN
    assert correct is True


@pytest.mark.asyncio
async def test_buy_sl_hit_resolves_as_loss():
    """BUY signal where price <= sl must resolve as LOSS."""
    tracker, provider = make_tracker(price=95.0)  # exactly at SL
    pending = make_pending(action=SignalAction.BUY, take_profit_1=110.0, stop_loss=95.0)

    result = await tracker._resolve(pending)

    assert result is True
    outcome, correct = _outcome_from_call(tracker._write_outcome)
    assert outcome == OUTCOME_LOSS
    assert correct is False


@pytest.mark.asyncio
async def test_buy_sl_breached_resolves_as_loss():
    """BUY signal where price < sl (gapped through) also resolves as LOSS."""
    tracker, provider = make_tracker(price=90.0)
    pending = make_pending(action=SignalAction.BUY, take_profit_1=110.0, stop_loss=95.0)

    result = await tracker._resolve(pending)

    assert result is True
    outcome, correct = _outcome_from_call(tracker._write_outcome)
    assert outcome == OUTCOME_LOSS
    assert correct is False


# ── SELL signal tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sell_tp1_hit_resolves_as_win():
    """SELL signal where price <= tp1 must resolve as WIN."""
    # For a SELL, TP1 is below entry (price needs to fall)
    tracker, provider = make_tracker(price=90.0)  # exactly at TP1
    pending = make_pending(
        action=SignalAction.SELL,
        entry_price=100.0,
        take_profit_1=90.0,
        stop_loss=105.0,
    )

    result = await tracker._resolve(pending)

    assert result is True
    outcome, correct = _outcome_from_call(tracker._write_outcome)
    assert outcome == OUTCOME_WIN
    assert correct is True


@pytest.mark.asyncio
async def test_sell_sl_hit_resolves_as_loss():
    """SELL signal where price >= sl must resolve as LOSS."""
    tracker, provider = make_tracker(price=105.0)  # exactly at SL
    pending = make_pending(
        action=SignalAction.SELL,
        entry_price=100.0,
        take_profit_1=90.0,
        stop_loss=105.0,
    )

    result = await tracker._resolve(pending)

    assert result is True
    outcome, correct = _outcome_from_call(tracker._write_outcome)
    assert outcome == OUTCOME_LOSS
    assert correct is False


# ── No directional fallback ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pending_when_price_between_sl_tp1():
    """Price between SL and TP1 must NOT resolve — signal stays pending."""
    # Price above entry but below TP1 — old code would have called this WIN
    tracker, provider = make_tracker(price=105.0)
    pending = make_pending(
        action=SignalAction.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit_1=110.0,
    )

    result = await tracker._resolve(pending)

    assert result is False
    tracker._write_outcome.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_directional_fallback_buy():
    """Even with price > entry but < tp1, BUY must NOT resolve as WIN."""
    tracker, provider = make_tracker(price=108.0)  # above entry, below TP1
    pending = make_pending(
        action=SignalAction.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit_1=110.0,
    )

    result = await tracker._resolve(pending)

    assert result is False
    tracker._write_outcome.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_directional_fallback_sell():
    """Even with price < entry but > tp1, SELL must NOT resolve as WIN."""
    tracker, provider = make_tracker(price=92.0)  # below entry, above TP1 (tp1=90)
    pending = make_pending(
        action=SignalAction.SELL,
        entry_price=100.0,
        take_profit_1=90.0,
        stop_loss=105.0,
    )

    result = await tracker._resolve(pending)

    assert result is False
    tracker._write_outcome.assert_not_awaited()


# ── Expiry logic ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_expired_after_max_age():
    """Signals older than MAX_SIGNAL_AGE_HOURS must be marked EXPIRED via _process_due."""
    tracker, provider = make_tracker(price=105.0)  # price between SL and TP1

    old_created_at = datetime.now(timezone.utc) - timedelta(
        hours=SignalOutcomeTracker.MAX_SIGNAL_AGE_HOURS + 1
    )
    pending = make_pending(
        action=SignalAction.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit_1=110.0,
        created_at=old_created_at,
    )
    tracker._queue.append(pending)

    await tracker._process_due()

    # Signal must have been expired and removed from queue
    tracker._write_outcome.assert_awaited_once()
    call_args = tracker._write_outcome.call_args
    assert call_args.kwargs["outcome"] == OUTCOME_EXPIRED
    assert call_args.kwargs["correct"] is False
    assert tracker._queue == []


@pytest.mark.asyncio
async def test_recent_signal_not_expired():
    """A signal created 1 minute ago with price between SL/TP1 must NOT be expired."""
    tracker, provider = make_tracker(price=105.0)

    fresh_created = datetime.now(timezone.utc) - timedelta(minutes=1)
    pending = make_pending(
        action=SignalAction.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit_1=110.0,
        created_at=fresh_created,
    )
    tracker._queue.append(pending)

    await tracker._process_due()

    # No outcome written, signal stays in queue
    tracker._write_outcome.assert_not_awaited()
    assert len(tracker._queue) == 1


# ── Provider edge cases ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_provider_returns_false():
    """When provider is None, _resolve must return False without writing outcome."""
    tracker = SignalOutcomeTracker(crypto_provider=None, stock_provider=None)
    tracker._write_outcome = AsyncMock()

    pending = make_pending(action=SignalAction.BUY, asset_class="crypto")
    result = await tracker._resolve(pending)

    assert result is False
    tracker._write_outcome.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_exception_returns_false():
    """When price fetch raises, _resolve must return False (never re-raise)."""
    provider = MagicMock()
    provider.get_latest_price = AsyncMock(side_effect=RuntimeError("network error"))
    tracker = SignalOutcomeTracker(crypto_provider=provider)
    tracker._write_outcome = AsyncMock()

    pending = make_pending(action=SignalAction.BUY, asset_class="crypto")
    result = await tracker._resolve(pending)

    assert result is False
    tracker._write_outcome.assert_not_awaited()


@pytest.mark.asyncio
async def test_none_price_returns_false():
    """When provider returns None price, _resolve must return False."""
    tracker, _ = make_tracker(price=None)
    tracker._write_outcome = AsyncMock()

    pending = make_pending(action=SignalAction.BUY)
    result = await tracker._resolve(pending)

    assert result is False
    tracker._write_outcome.assert_not_awaited()


@pytest.mark.asyncio
async def test_zero_price_returns_false():
    """When provider returns 0.0 (invalid), _resolve must return False."""
    tracker, _ = make_tracker(price=0.0)
    tracker._write_outcome = AsyncMock()

    pending = make_pending(action=SignalAction.BUY)
    result = await tracker._resolve(pending)

    assert result is False
    tracker._write_outcome.assert_not_awaited()


# ── _process_due integration ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_due_removes_resolved_signals():
    """Resolved signals (TP1 hit) must be removed from the queue."""
    tracker, provider = make_tracker(price=115.0)  # above TP1

    pending = make_pending(action=SignalAction.BUY, take_profit_1=110.0, stop_loss=95.0)
    tracker._queue.append(pending)

    await tracker._process_due()

    assert tracker._queue == []
    tracker._write_outcome.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_due_keeps_unresolved_signals():
    """Unresolved signals (price between SL and TP1) must remain in the queue."""
    tracker, provider = make_tracker(price=105.0)

    pending = make_pending(action=SignalAction.BUY, entry_price=100.0, stop_loss=95.0, take_profit_1=110.0)
    tracker._queue.append(pending)

    await tracker._process_due()

    assert len(tracker._queue) == 1
    tracker._write_outcome.assert_not_awaited()
