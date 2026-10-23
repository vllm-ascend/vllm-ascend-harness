"""Plugin manager with extensible registration interface.

Inspired by OpenHarness's plugin architecture.  Provides a central
registry for skills, adapters, and scorers to be discovered and
invoked at runtime.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol


class Plugin(Protocol):
    """Protocol that all plugins must satisfy."""

    name: str

    def initialize(self, **kwargs: Any) -> None: ...


class PluginManager:
    """Central plugin registry.

    Extension point: third-party modules register custom skills,
    model adapters, or scorers via :meth:`register`.  Registered
    plugins are discoverable by name through :meth:`get`.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._factories: dict[str, Callable[..., Plugin]] = {}
        self._factory_kwargs: dict[str, dict[str, Any]] = {}

    def register(self, plugin: Plugin, **kwargs: Any) -> None:
        """Register *plugin* and call its ``initialize`` hook."""
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin already registered: {plugin.name}")
        plugin.initialize(**kwargs)
        self._plugins[plugin.name] = plugin

    def unregister(self, name: str) -> None:
        """Remove a registered plugin by *name*."""
        self._plugins.pop(name, None)
        self._factories.pop(name, None)
        self._factory_kwargs.pop(name, None)

    def get(self, name: str) -> Plugin | None:
        """Look up a plugin by *name*.

        If only a factory is registered, instantiate it lazily on the
        first call and cache the result.
        """
        if name in self._plugins:
            return self._plugins[name]
        if name in self._factories:
            factory = self._factories[name]
            kwargs = self._factory_kwargs.get(name, {})
            plugin = factory(**kwargs)
            plugin.initialize(**kwargs)
            self._plugins[name] = plugin
            del self._factories[name]
            del self._factory_kwargs[name]
            return plugin
        return None

    def list_plugins(self) -> list[str]:
        """Return all registered plugin names (including lazy factories)."""
        return list(set(self._plugins.keys()) | set(self._factories.keys()))

    def register_factory(
        self, name: str, factory: Callable[..., Plugin], **kwargs: Any
    ) -> None:
        """Register a lazy factory for plugin *name*.

        The factory is called with *kwargs* on the first :meth:`get`
        invocation, and the resulting plugin is cached for subsequent
        lookups.
        """
        if name in self._plugins or name in self._factories:
            raise ValueError(f"Plugin already registered: {name}")
        self._factories[name] = factory
        self._factory_kwargs[name] = kwargs
