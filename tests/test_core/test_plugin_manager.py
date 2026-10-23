"""Unit tests for PluginManager."""

import pytest

from vllm_ascend_harness.core.plugin_manager import PluginManager


class FakePlugin:
    """Minimal plugin for testing."""

    def __init__(self, name, init_kwargs=None):
        self.name = name
        self.init_kwargs = init_kwargs
        self.initialized = False

    def initialize(self, **kwargs):
        self.init_kwargs = kwargs
        self.initialized = True


class TestRegisterAndGet:
    def test_register_and_get(self):
        pm = PluginManager()
        plugin = FakePlugin("test-plugin")
        pm.register(plugin)
        assert pm.get("test-plugin") is plugin
        assert plugin.initialized is True

    def test_get_unknown_returns_none(self):
        pm = PluginManager()
        assert pm.get("nonexistent") is None

    def test_register_duplicate_raises(self):
        pm = PluginManager()
        pm.register(FakePlugin("dup"))
        with pytest.raises(ValueError, match="already registered"):
            pm.register(FakePlugin("dup"))

    def test_register_with_kwargs(self):
        pm = PluginManager()
        plugin = FakePlugin("kw-plugin")
        pm.register(plugin, key="value")
        assert plugin.init_kwargs == {"key": "value"}


class TestUnregister:
    def test_unregister(self):
        pm = PluginManager()
        pm.register(FakePlugin("to-remove"))
        pm.unregister("to-remove")
        assert pm.get("to-remove") is None

    def test_unregister_unknown_silent(self):
        pm = PluginManager()
        pm.unregister("nonexistent")

    def test_unregister_removes_factory(self):
        pm = PluginManager()
        pm.register_factory("lazy", lambda **kw: FakePlugin("lazy", kw))
        pm.unregister("lazy")
        assert pm.get("lazy") is None


class TestListPlugins:
    def test_list_empty(self):
        pm = PluginManager()
        assert pm.list_plugins() == []

    def test_list_with_registered(self):
        pm = PluginManager()
        pm.register(FakePlugin("a"))
        pm.register(FakePlugin("b"))
        assert set(pm.list_plugins()) == {"a", "b"}

    def test_list_includes_factories(self):
        pm = PluginManager()
        pm.register(FakePlugin("a"))
        pm.register_factory("b", lambda **kw: FakePlugin("b", kw))
        names = set(pm.list_plugins())
        assert names == {"a", "b"}


class TestRegisterFactory:
    def test_register_factory_lazy_instantiation(self):
        pm = PluginManager()
        call_count = [0]

        def factory(**kwargs):
            call_count[0] += 1
            return FakePlugin("lazy", kwargs)

        pm.register_factory("lazy", factory, key="val")
        assert call_count[0] == 0
        plugin = pm.get("lazy")
        assert plugin is not None
        assert plugin.name == "lazy"
        assert plugin.initialized is True
        assert plugin.init_kwargs == {"key": "val"}
        assert call_count[0] == 1

    def test_factory_cached_after_first_get(self):
        pm = PluginManager()

        def factory(**kwargs):
            return FakePlugin("cached", kwargs)

        pm.register_factory("cached", factory)
        first = pm.get("cached")
        second = pm.get("cached")
        assert first is second

    def test_register_factory_duplicate_raises(self):
        pm = PluginManager()
        pm.register_factory("dup", lambda **kw: FakePlugin("dup"))
        with pytest.raises(ValueError, match="already registered"):
            pm.register_factory("dup", lambda **kw: FakePlugin("dup"))

    def test_register_factory_after_register_raises(self):
        pm = PluginManager()
        pm.register(FakePlugin("existing"))
        with pytest.raises(ValueError, match="already registered"):
            pm.register_factory("existing", lambda **kw: FakePlugin("existing"))

    def test_factory_kwargs_passed(self):
        pm = PluginManager()

        def factory(**kwargs):
            p = FakePlugin("kw-test", kwargs)
            return p

        pm.register_factory("kw-test", factory, a=1, b=2)
        plugin = pm.get("kw-test")
        assert plugin.init_kwargs == {"a": 1, "b": 2}
