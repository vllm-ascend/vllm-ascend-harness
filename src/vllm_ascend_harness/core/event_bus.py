"""Event bus for decoupled inter-module communication.

Provides a lightweight publish/subscribe mechanism so that the agent
loop, telemetry, and evaluator modules can communicate without hard
dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

EventHandler = Callable[[Any], None]


class EventBus:
    """In-process publish/subscribe event bus.

    Modules subscribe to named event topics; the bus dispatches
    published events to all registered handlers.

    Parameters
    ----------
    error_isolation:
        When ``True`` (default), exceptions raised by handlers are
        caught and logged so one failing handler does not break
        others.  When ``False``, the first exception propagates.
    """

    def __init__(self, error_isolation: bool = True) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._error_isolation = error_isolation

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Register *handler* for *topic*."""
        self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """Remove *handler* from *topic*."""
        handlers = self._handlers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, topic: str, event: Any = None) -> None:
        """Publish *event* to all subscribers of *topic*.

        When error isolation is enabled, handler exceptions are caught
        and stored in :attr:`last_errors`.  Otherwise the first
        exception propagates immediately.
        """
        self.last_errors: list[tuple[str, Exception]] = []
        for handler in self._handlers.get(topic, []):
            try:
                handler(event)
            except Exception as exc:
                if not self._error_isolation:
                    raise
                self.last_errors.append((handler.__name__, exc))

    def publish_async(self, topic: str, event: Any = None) -> None:
        """Publish *event* dispatching each handler in a background thread.

        Uses a :class:`~concurrent.futures.ThreadPoolExecutor` to run
        handlers concurrently.  This method blocks until all handlers
        finish.  Errors are isolated when :attr:`_error_isolation` is
        ``True``.
        """
        import concurrent.futures

        self.last_errors = []
        handlers = list(self._handlers.get(topic, []))
        if not handlers:
            return
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(handlers), 4)
        ) as pool:
            futures = {
                pool.submit(handler, event): handler for handler in handlers
            }
            for future in concurrent.futures.as_completed(futures):
                handler = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    if not self._error_isolation:
                        raise
                    self.last_errors.append((handler.__name__, exc))

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._handlers.clear()
