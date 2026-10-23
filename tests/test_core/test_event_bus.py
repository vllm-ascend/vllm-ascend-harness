"""Unit tests for EventBus."""

import pytest

from vllm_ascend_harness.core.event_bus import EventBus


def test_subscribe_and_publish():
    received: list = []
    bus = EventBus()
    bus.subscribe("test", received.append)
    bus.publish("test", "hello")
    assert received == ["hello"]


def test_unsubscribe():
    received: list = []
    bus = EventBus()
    handler = received.append
    bus.subscribe("test", handler)
    bus.unsubscribe("test", handler)
    bus.publish("test", "hello")
    assert received == []


def test_clear():
    received: list = []
    bus = EventBus()
    bus.subscribe("test", received.append)
    bus.clear()
    bus.publish("test", "hello")
    assert received == []


# --- Error isolation tests ---


def test_error_isolation_catches_handler_exception():
    received: list = []

    def bad_handler(event):
        raise RuntimeError("handler error")

    def good_handler(event):
        received.append(event)

    bus = EventBus(error_isolation=True)
    bus.subscribe("topic", bad_handler)
    bus.subscribe("topic", good_handler)
    bus.publish("topic", "data")
    assert received == ["data"]
    assert len(bus.last_errors) == 1


def test_no_error_isolation_propagates():
    def bad_handler(event):
        raise RuntimeError("handler error")

    bus = EventBus(error_isolation=False)
    bus.subscribe("topic", bad_handler)
    with pytest.raises(RuntimeError, match="handler error"):
        bus.publish("topic", "data")


# --- Async dispatch tests ---


def test_publish_async_dispatches_all():
    received: list = []
    bus = EventBus()
    bus.subscribe("topic", received.append)
    bus.publish_async("topic", "async-data")
    assert received == ["async-data"]


def test_publish_async_error_isolation():
    received: list = []

    def bad_handler(event):
        raise RuntimeError("async error")

    def good_handler(event):
        received.append(event)

    bus = EventBus(error_isolation=True)
    bus.subscribe("topic", bad_handler)
    bus.subscribe("topic", good_handler)
    bus.publish_async("topic", "data")
    assert received == ["data"]
    assert len(bus.last_errors) == 1


def test_publish_async_no_handlers():
    bus = EventBus()
    bus.publish_async("nonexistent", "data")
