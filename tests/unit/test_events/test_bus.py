"""Unit tests for EventBus."""
from __future__ import annotations

import pytest
import asyncio
from libs.core.events.bus import EventBus


@pytest.mark.asyncio
async def test_subscribe_and_publish_calls_handler():
    """subscribe + publish → handler called with correct payload."""
    bus = EventBus()
    received = []

    async def handler(payload: dict) -> None:
        received.append(payload)

    bus.subscribe(EventBus.SIGNAL_GENERATED, handler)
    await bus.publish(EventBus.SIGNAL_GENERATED, {"symbol": "AAPL", "action": "BUY"})

    assert len(received) == 1
    assert received[0] == {"symbol": "AAPL", "action": "BUY"}


@pytest.mark.asyncio
async def test_multiple_handlers_all_called():
    """Multiple handlers for same topic → all are called."""
    bus = EventBus()
    calls = []

    async def handler_a(payload: dict) -> None:
        calls.append("a")

    async def handler_b(payload: dict) -> None:
        calls.append("b")

    bus.subscribe(EventBus.SIGNAL_GENERATED, handler_a)
    bus.subscribe(EventBus.SIGNAL_GENERATED, handler_b)
    await bus.publish(EventBus.SIGNAL_GENERATED, {})

    assert "a" in calls
    assert "b" in calls
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_failing_handler_does_not_stop_others():
    """Handler that raises → other handlers are still called."""
    bus = EventBus()
    calls = []

    async def bad_handler(payload: dict) -> None:
        raise RuntimeError("boom")

    async def good_handler(payload: dict) -> None:
        calls.append("good")

    bus.subscribe(EventBus.DATA_QUALITY_BLOCKED, bad_handler)
    bus.subscribe(EventBus.DATA_QUALITY_BLOCKED, good_handler)

    # Should not raise
    await bus.publish(EventBus.DATA_QUALITY_BLOCKED, {"symbol": "AAPL"})

    assert calls == ["good"]


@pytest.mark.asyncio
async def test_unsubscribe_prevents_handler_calls():
    """unsubscribe → handler not called after removal."""
    bus = EventBus()
    calls = []

    async def handler(payload: dict) -> None:
        calls.append(payload)

    bus.subscribe(EventBus.SIGNAL_GENERATED, handler)
    bus.unsubscribe(EventBus.SIGNAL_GENERATED, handler)
    await bus.publish(EventBus.SIGNAL_GENERATED, {"symbol": "TSLA"})

    assert calls == []


@pytest.mark.asyncio
async def test_publish_to_topic_with_no_subscribers_no_exception():
    """publish to topic with no subscribers → no exception raised."""
    bus = EventBus()
    # Should complete without error
    await bus.publish("nonexistent.topic", {"data": "value"})


def test_handler_count_returns_correct_count():
    """handler_count returns the correct number of registered handlers."""
    bus = EventBus()

    async def h1(p): pass
    async def h2(p): pass
    async def h3(p): pass

    assert bus.handler_count(EventBus.SIGNAL_GENERATED) == 0

    bus.subscribe(EventBus.SIGNAL_GENERATED, h1)
    assert bus.handler_count(EventBus.SIGNAL_GENERATED) == 1

    bus.subscribe(EventBus.SIGNAL_GENERATED, h2)
    bus.subscribe(EventBus.SIGNAL_GENERATED, h3)
    assert bus.handler_count(EventBus.SIGNAL_GENERATED) == 3

    bus.unsubscribe(EventBus.SIGNAL_GENERATED, h2)
    assert bus.handler_count(EventBus.SIGNAL_GENERATED) == 2


@pytest.mark.asyncio
async def test_clear_removes_all_subscriptions():
    """clear() removes all subscriptions so no handlers are called."""
    bus = EventBus()
    calls = []

    async def handler(payload: dict) -> None:
        calls.append(payload)

    bus.subscribe(EventBus.SIGNAL_GENERATED, handler)
    bus.subscribe(EventBus.PROVIDER_ERROR, handler)
    bus.clear()

    await bus.publish(EventBus.SIGNAL_GENERATED, {})
    await bus.publish(EventBus.PROVIDER_ERROR, {})

    assert calls == []
    assert bus.handler_count(EventBus.SIGNAL_GENERATED) == 0
