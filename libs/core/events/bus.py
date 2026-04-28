from __future__ import annotations
import asyncio
from collections import defaultdict
from typing import Any, Callable, Awaitable
from libs.core.logging.logger import get_logger

log = get_logger(__name__)

# Handler type: async callable receiving event data dict
EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


class EventBus:
    """
    Simple in-process async pub/sub event bus.

    Events are identified by string topic names.
    Handlers are async callables that receive the event payload dict.

    Usage:
        bus = EventBus()
        bus.subscribe("signal.generated", my_handler)
        await bus.publish("signal.generated", {"symbol": "AAPL", ...})
    """

    # Well-known topic constants
    SIGNAL_GENERATED = "signal.generated"
    DATA_QUALITY_BLOCKED = "data.quality.blocked"
    PROVIDER_ERROR = "provider.error"
    PROVIDER_DISCONNECT = "provider.disconnect"

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Register an async handler for a topic."""
        self._handlers[topic].append(handler)
        log.debug("event_bus_subscribed", topic=topic, handler=handler.__name__)

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """Remove a handler. No-op if not registered."""
        self._handlers[topic] = [h for h in self._handlers[topic] if h is not handler]

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """
        Dispatch payload to all handlers subscribed to topic.
        Errors in individual handlers are logged but do not stop other handlers.
        """
        handlers = list(self._handlers.get(topic, []))
        if not handlers:
            return
        for handler in handlers:
            try:
                await handler(payload)
            except Exception as exc:
                log.warning(
                    "event_handler_error",
                    topic=topic,
                    handler=handler.__name__,
                    error=str(exc),
                )

    def handler_count(self, topic: str) -> int:
        return len(self._handlers.get(topic, []))

    def clear(self) -> None:
        """Remove all subscriptions (useful in tests)."""
        self._handlers.clear()
